# Week 4, Task 4 (alternative) — multimodal fusion built by transfer learning: reuse the
# pretrained CNN-LSTM signal encoder and tabular MLP encoder, freeze their trained weights
# as an initialization, and fine-tune a fusion head on top. Self-contained.
#
# Requires:
#   - ./ptb-xl/ptbxl_5class.csv        (week4task1_label_extraction.py)
#   - week4task1_cnn_lstm.pt            (week4task1_cnn_lstm.py)
#   - week4task3_tabular_mlp.pt         (week4task3_tabular_mlp.py)
# For the 4-way "ultimate" comparison plot it optionally loads week4task1_test_probs.npz,
# week4task3_test_probs.npz and week4task4_fusion_test_probs.npz; any missing curve is
# skipped with a warning rather than failing the script.
# %%
import copy
import os
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using', device)

SIG_CKPT = 'week4task1_cnn_lstm.pt'
TAB_CKPT = 'week4task3_tabular_mlp.pt'
for ckpt in (SIG_CKPT, TAB_CKPT):
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f'{ckpt} not found. Run week4task1_cnn_lstm.py and week4task3_tabular_mlp.py '
            'once each to produce the pretrained encoders this script transfers from.'
        )

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

# %%  pretrained encoder architectures (must match week4task1_cnn_lstm.py / week4task3_tabular_mlp.py)
class CNNLSTM(nn.Module):
    def __init__(self, n_classes=5, hidden_size=128, p_drop=0.3, bidirectional=True):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32,  kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64,  kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()
        self.lstm = nn.LSTM(128, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.drop = nn.Dropout(p_drop)
        self.fc   = nn.Linear(lstm_out, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        emb = torch.cat([h_n[-2], h_n[-1]], dim=1)
        emb = self.drop(emb)
        return self.fc(emb)

class TabularMLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, n_classes=5, p_drop=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

pretrained_sig_model = CNNLSTM(n_classes=5, hidden_size=128, bidirectional=True)
pretrained_sig_model.load_state_dict(torch.load(SIG_CKPT, map_location='cpu'))

pretrained_tab_model = TabularMLP()
pretrained_tab_model.load_state_dict(torch.load(TAB_CKPT, map_location='cpu'))

# %%  transfer fusion model: chop off both heads, concat embeddings, train a new head
class TransferFusionModel(nn.Module):
    def __init__(self, pretrained_sig_model, pretrained_tab_model, n_classes=5):
        super().__init__()
        # deepcopy so we never mutate the loaded pretrained encoders
        self.sig_encoder = copy.deepcopy(pretrained_sig_model)
        self.tab_encoder = copy.deepcopy(pretrained_tab_model)

        # replace the final Linear layers with Identity -> encoders return embeddings, not logits
        self.sig_encoder.fc = nn.Identity()    # CNN-LSTM's final layer was named 'fc'
        self.tab_encoder.fc2 = nn.Identity()   # TabularMLP's final layer was named 'fc2'

        self.drop = nn.Dropout(0.3)
        self.fc_fusion = nn.Linear(256 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        emb_sig = self.sig_encoder(x_sig)   # (B, 256)
        emb_tab = self.tab_encoder(x_tab)   # (B, 16)
        emb_fused = torch.cat([emb_sig, emb_tab], dim=1)   # (B, 272)
        emb_fused = self.drop(emb_fused)
        return self.fc_fusion(emb_fused)

model_transfer = TransferFusionModel(pretrained_sig_model, pretrained_tab_model).to(device)
criterion_transfer = nn.CrossEntropyLoss(weight=weights5)
optimizer_transfer = torch.optim.Adam(model_transfer.parameters(), lr=1e-4)   # small lr: mostly-trained backbone

# %%  train (best-val checkpoint) — shorter run since the encoders start pretrained
tr_losses_trans, va_losses_trans = [], []
best_val_trans, best_state_trans = float('inf'), None

for epoch in range(15):
    model_transfer.train()
    running = 0.0
    for x_sig, x_tab, yb in train_loader_fuse:
        x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)

        optimizer_transfer.zero_grad()
        loss = criterion_transfer(model_transfer(x_sig, x_tab), yb)
        loss.backward()
        optimizer_transfer.step()
        running += loss.item() * yb.size(0)
    tr = running / len(train_loader_fuse.dataset)

    model_transfer.eval()
    running = 0.0
    with torch.no_grad():
        for x_sig, x_tab, yb in val_loader_fuse:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            running += criterion_transfer(model_transfer(x_sig, x_tab), yb).item() * yb.size(0)
    va = running / len(val_loader_fuse.dataset)

    tr_losses_trans.append(tr); va_losses_trans.append(va)
    if va < best_val_trans:
        best_val_trans, best_state_trans = va, copy.deepcopy(model_transfer.state_dict())

    print(f'epoch {epoch+1:2d}/15   train {tr:.4f}   val {va:.4f}')

model_transfer.load_state_dict(best_state_trans)
print(f'best transfer fusion val loss: {best_val_trans:.4f}')

# %%  evaluate on the test set
model_transfer.eval()
all_pred_trans, all_probs_trans = [], []
with torch.no_grad():
    for x_sig, x_tab, yb in test_loader_fuse:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        outputs = model_transfer(x_sig, x_tab)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_pred_trans.extend(preds.cpu().numpy())
        all_probs_trans.extend(probs.cpu().numpy())

all_pred_trans = np.array(all_pred_trans)
all_probs_trans = np.array(all_probs_trans)

macro_auroc_trans = roc_auc_score(y_te, all_probs_trans, multi_class='ovr', average='macro')
print(f'Transfer Fusion Macro AUROC: {macro_auroc_trans:.4f}')

# %%  loss curves
epochs_trans = range(1, len(tr_losses_trans) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs_trans, tr_losses_trans, marker='o', linewidth=2, label='Training Loss')
plt.plot(epochs_trans, va_losses_trans, marker='s', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Cross-Entropy Loss')
plt.title('Trgansfer Fusion Model: Training and Validation Loss')
plt.xticks(epochs_trans)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('week4task4_transfer_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  ultimate 4-way ROC comparison (micro-average) — others loaded from disk if available
y_true_bin_trans = label_binarize(y_te, classes=[0, 1, 2, 3, 4])

plt.figure(figsize=(8, 8))

def plot_saved_curve(npz_path, label, color, missing_hint):
    if not os.path.exists(npz_path):
        print(f'warning: {npz_path} not found -> {missing_hint}')
        return
    data = np.load(npz_path)
    y_true_bin = label_binarize(data['y_true'], classes=[0, 1, 2, 3, 4])
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), data['probs'].ravel())
    plt.plot(fpr, tpr, label=f'{label} (Micro AUC = {auc(fpr, tpr):.3f})', color=color, linewidth=2)

plot_saved_curve('week4task1_test_probs.npz', 'Signal Only', 'blue',
                  'run week4task1_cnn_lstm.py to include this curve')
plot_saved_curve('week4task3_test_probs.npz', 'Tabular Only', 'orange',
                  'run week4task3_tabular_mlp.py to include this curve')
plot_saved_curve('week4task4_fusion_test_probs.npz', 'Fusion From Scratch', 'green',
                  'run week4task4_fusion.py to include this curve')

fpr_trans, tpr_trans, _ = roc_curve(y_true_bin_trans.ravel(), all_probs_trans.ravel())
plt.plot(fpr_trans, tpr_trans, label=f'Fusion Transfer (Micro AUC = {auc(fpr_trans, tpr_trans):.3f})',
         color='red', linewidth=2, linestyle='-.')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ultimate Multimodal Comparison: Micro-Average ROC Curves')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('week4task4_ultimate_comparison.png', dpi=150)
plt.show()

# %%
torch.save(model_transfer.state_dict(), 'week4task4_transfer_fusion.pt')
print('saved week4task4_transfer_fusion.pt')
