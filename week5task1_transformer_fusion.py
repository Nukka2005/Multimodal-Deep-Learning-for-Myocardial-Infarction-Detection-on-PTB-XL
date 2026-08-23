# Week 5, Task 1 — multimodal fusion with a Transformer sequence reader instead of the
# LSTM, trained from scratch. Self-contained.
# Requires ./ptb-xl/ptbxl_5class.csv (run week4task1_label_extraction.py once to create it).
# %%
import copy
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using', device)

# %%  labels -> fold split -> load 100 Hz signals + tabular features
df5 = pd.read_csv('./ptb-xl/ptbxl_5class.csv', index_col='ecg_id')
tr_df = df5[df5['strat_fold'] <= 8]
va_df = df5[df5['strat_fold'] == 9]
te_df = df5[df5['strat_fold'] == 10]

def load_split(split_df):
    X, y = [], []
    for _, row in split_df.iterrows():
        sig, _ = wfdb.rdsamp('./ptb-xl/' + row['filename_lr'])   # (1000, 12)
        X.append(sig.T)                                          # (12, 1000)
        y.append(row['label'])
    return np.array(X, dtype='float32'), np.array(y, dtype='int64')

X_tr, y_tr = load_split(tr_df)
X_va, y_va = load_split(va_df)
X_te, y_te = load_split(te_df)

weights5 = torch.tensor(
    compute_class_weight('balanced', classes=np.array([0, 1, 2, 3, 4]), y=y_tr),
    dtype=torch.float32).to(device)

features = ['age', 'sex', 'height', 'weight']
imputer = SimpleImputer(strategy='median')
X_tr_imp = imputer.fit_transform(tr_df[features])
X_va_imp = imputer.transform(va_df[features])
X_te_imp = imputer.transform(te_df[features])

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_imp)
X_va_scaled = scaler.transform(X_va_imp)
X_te_scaled = scaler.transform(X_te_imp)

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, X_sig, X_tab, y):
        self.X_sig = torch.tensor(X_sig, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_sig[idx], self.X_tab[idx], self.y[idx]

train_loader_fuse = DataLoader(MultimodalDataset(X_tr, X_tr_scaled, y_tr), batch_size=32, shuffle=True)
val_loader_fuse   = DataLoader(MultimodalDataset(X_va, X_va_scaled, y_va), batch_size=32, shuffle=False)
test_loader_fuse  = DataLoader(MultimodalDataset(X_te, X_te_scaled, y_te), batch_size=32, shuffle=False)

# %%  model
class TransformerFusionModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        # --- Signal Encoder Backbone (CNN) ---
        self.conv1 = nn.Conv1d(12, 32, 7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()

        # --- Transformer Sequence Reader ---
        self.pos_emb = nn.Parameter(torch.randn(1, 250, 128))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256, batch_first=True, dropout=0.3,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # --- Tabular Encoder Backbone ---
        self.fc_tab = nn.Linear(4, 16)

        # --- Fusion Head ---
        # 128 from Transformer + 16 from Tabular = 144
        self.drop = nn.Dropout(0.3)
        self.fc_fusion = nn.Linear(128 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        s = self.pool(self.relu(self.bn1(self.conv1(x_sig))))
        s = self.pool(self.relu(self.bn2(self.conv2(s))))
        s = self.relu(self.bn3(self.conv3(s)))

        s = s.permute(0, 2, 1)          # (B, 250, 128)
        s = s + self.pos_emb
        s = self.transformer(s)         # (B, 250, 128)
        emb_sig = s.mean(dim=1)         # (B, 128) global average pool over time

        emb_tab = self.relu(self.fc_tab(x_tab))   # (B, 16)

        emb_fused = torch.cat([emb_sig, emb_tab], dim=1)   # (B, 144)
        emb_fused = self.drop(emb_fused)
        return self.fc_fusion(emb_fused)

model_trans_fuse = TransformerFusionModel().to(device)
criterion_trans = nn.CrossEntropyLoss(weight=weights5)
optimizer_trans = torch.optim.Adam(model_trans_fuse.parameters(), lr=1e-3)

# %%  train (best-val checkpoint)
tr_losses_tfuse, va_losses_tfuse = [], []
best_val_tfuse, best_state_tfuse = float('inf'), None

for epoch in range(30):
    model_trans_fuse.train()
    running = 0.0
    for x_sig, x_tab, yb in train_loader_fuse:
        x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)

        optimizer_trans.zero_grad()
        loss = criterion_trans(model_trans_fuse(x_sig, x_tab), yb)
        loss.backward()
        optimizer_trans.step()
        running += loss.item() * yb.size(0)
    tr = running / len(train_loader_fuse.dataset)

    model_trans_fuse.eval()
    running = 0.0
    with torch.no_grad():
        for x_sig, x_tab, yb in val_loader_fuse:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            running += criterion_trans(model_trans_fuse(x_sig, x_tab), yb).item() * yb.size(0)
    va = running / len(val_loader_fuse.dataset)

    tr_losses_tfuse.append(tr); va_losses_tfuse.append(va)
    if va < best_val_tfuse:
        best_val_tfuse, best_state_tfuse = va, copy.deepcopy(model_trans_fuse.state_dict())

    print(f'epoch {epoch+1:2d}/30   train {tr:.4f}   val {va:.4f}')

model_trans_fuse.load_state_dict(best_state_tfuse)
print(f'best val loss: {best_val_tfuse:.4f}')

# %%  evaluate on the test set
model_trans_fuse.eval()
all_probs_tfuse = []
with torch.no_grad():
    for x_sig, x_tab, yb in test_loader_fuse:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        outputs = model_trans_fuse(x_sig, x_tab)
        probs = F.softmax(outputs, dim=1)
        all_probs_tfuse.extend(probs.cpu().numpy())

all_probs_tfuse = np.array(all_probs_tfuse)
macro_auroc_tfuse = roc_auc_score(y_te, all_probs_tfuse, multi_class='ovr', average='macro')
print(f'Transformer Fusion Macro AUROC: {macro_auroc_tfuse:.4f}')

# %%
torch.save(model_trans_fuse.state_dict(), 'week5task1_transformer_fusion.pt')
print('saved week5task1_transformer_fusion.pt')
# %%
import matplotlib.pyplot as plt

epochs = range(1, len(tr_losses_tfuse) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, tr_losses_tfuse, marker='o', linewidth=2, label='Training Loss')
plt.plot(epochs, va_losses_tfuse, marker='s', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Transformer Fusion — Training vs Validation Loss')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('week5task1_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
per_class_tfuse = roc_auc_score(y_te, all_probs_tfuse, multi_class='ovr', average=None)
classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

print('Per-class AUROC (Transformer Fusion):')
for c, a in zip(classes, per_class_tfuse):
    print(f'  {c:5s} {a:.4f}')
print(f'  macro AUROC {macro_auroc_tfuse:.4f}')
# %%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
y_true_bin = label_binarize(y_te, classes=[0, 1, 2, 3, 4])

plt.figure(figsize=(7, 7))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs_tfuse[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Transformer Fusion — One-vs-Rest ROC Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('week5task1_transformer_roc.png', dpi=150)
plt.show()
