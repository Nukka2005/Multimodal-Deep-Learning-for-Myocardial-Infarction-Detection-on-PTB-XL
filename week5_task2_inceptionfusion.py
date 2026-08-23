# Week 5, Task 4 — multimodal fusion with a deeper, multi-scale Inception+Residual
# signal backbone, replacing the plain 3-block CNN. Combined with the BiLSTM sequence
# reader (kept the same, so this isolates the effect of the backbone change).
# Requires ./ptb-xl/ptbxl_5class.csv (run week4task1_label_extraction.py once to create it).
# %%
import copy
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

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

# %%  Inception block: parallel kernel sizes at one layer, concatenated
class InceptionBlock(nn.Module):
    """One layer, multiple kernel sizes in parallel -> concatenated."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 4          # 4 branches -> even split of out_ch
        self.b1 = nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1)
        self.b2 = nn.Conv1d(in_ch, branch_ch, kernel_size=5, padding=2)
        self.b3 = nn.Conv1d(in_ch, branch_ch, kernel_size=9, padding=4)
        self.b4 = nn.Conv1d(in_ch, branch_ch, kernel_size=1)   # 1x1 "bottleneck" branch
        self.bn = nn.BatchNorm1d(branch_ch * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.relu(self.bn(out))

# %%  deeper backbone: stacked Inception blocks with residual connections
class ResidualInceptionBackbone(nn.Module):
    """Signal backbone replacing the plain 3-block CNN. Same output shape: (B, 250, 128)."""
    def __init__(self, in_ch=12):
        super().__init__()
        self.stem   = InceptionBlock(in_ch, 64)     # 12 -> 64
        self.block1 = InceptionBlock(64, 64)        # residual pair at 64 channels
        self.block2 = InceptionBlock(64, 128)       # 64 -> 128
        self.block3 = InceptionBlock(128, 128)      # residual pair at 128 channels
        self.proj   = nn.Conv1d(64, 128, kernel_size=1)  # match channels for the skip
        self.pool   = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.stem(x)                # (B, 64, 1000)
        x = x + self.block1(x)          # residual: block1 learns a correction to x
        x = self.pool(x)                # (B, 64, 500)
        y = self.block2(x)              # (B, 128, 500)
        x = self.proj(x)                # (B, 128, 500) - lift channels to match y
        x = y + self.block3(y)          # residual pair at the deeper stage
        x = self.pool(x)                # (B, 128, 250)
        return x.permute(0, 2, 1)       # (B, 250, 128) - matches CNNBackbone's output

# %%  full fusion model: Inception backbone + BiLSTM sequence reader + tabular
class InceptionFusionModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.backbone = ResidualInceptionBackbone()
        self.lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=True)

        self.fc_tab = nn.Linear(4, 16)

        # 256 from bidirectional LSTM + 16 from tabular = 272
        self.drop = nn.Dropout(0.3)
        self.fc_fusion = nn.Linear(256 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        s = self.backbone(x_sig)              # (B, 250, 128)
        _, (h_n, _) = self.lstm(s)
        emb_sig = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, 256)

        emb_tab = F.relu(self.fc_tab(x_tab))  # (B, 16)

        emb_fused = torch.cat([emb_sig, emb_tab], dim=1)  # (B, 272)
        emb_fused = self.drop(emb_fused)
        return self.fc_fusion(emb_fused)

# %%  shape sanity check before training
model_inc_fuse = InceptionFusionModel().to(device)
test_out = model_inc_fuse(torch.randn(4, 12, 1000).to(device), torch.randn(4, 4).to(device))
print(test_out.shape)   # expect torch.Size([4, 5])

criterion_inc = nn.CrossEntropyLoss(weight=weights5)
optimizer_inc = torch.optim.Adam(model_inc_fuse.parameters(), lr=1e-3)

# %%  train (best-val checkpoint)
tr_losses_inc, va_losses_inc = [], []
best_val_inc, best_state_inc = float('inf'), None

for epoch in range(30):
    model_inc_fuse.train()
    running = 0.0
    for x_sig, x_tab, yb in train_loader_fuse:
        x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)

        optimizer_inc.zero_grad()
        loss = criterion_inc(model_inc_fuse(x_sig, x_tab), yb)
        loss.backward()
        optimizer_inc.step()
        running += loss.item() * yb.size(0)
    tr = running / len(train_loader_fuse.dataset)

    model_inc_fuse.eval()
    running = 0.0
    with torch.no_grad():
        for x_sig, x_tab, yb in val_loader_fuse:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            running += criterion_inc(model_inc_fuse(x_sig, x_tab), yb).item() * yb.size(0)
    va = running / len(val_loader_fuse.dataset)

    tr_losses_inc.append(tr); va_losses_inc.append(va)
    if va < best_val_inc:
        best_val_inc, best_state_inc = va, copy.deepcopy(model_inc_fuse.state_dict())

    print(f'epoch {epoch+1:2d}/30   train {tr:.4f}   val {va:.4f}')

model_inc_fuse.load_state_dict(best_state_inc)
print(f'best val loss: {best_val_inc:.4f}')

# %%  loss curve
epochs_inc = range(1, len(tr_losses_inc) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs_inc, tr_losses_inc, marker='o', linewidth=2, label='Training Loss')
plt.plot(epochs_inc, va_losses_inc, marker='s', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Cross-Entropy Loss')
plt.title('Inception+Residual Fusion — Training vs Validation Loss')
plt.xticks(epochs_inc)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('week5task4_inception_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  evaluate on the test set
model_inc_fuse.eval()
all_probs_inc = []
with torch.no_grad():
    for x_sig, x_tab, yb in test_loader_fuse:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        probs = F.softmax(model_inc_fuse(x_sig, x_tab), dim=1)
        all_probs_inc.extend(probs.cpu().numpy())

all_probs_inc = np.array(all_probs_inc)

per_class_inc = roc_auc_score(y_te, all_probs_inc, multi_class='ovr', average=None)
macro_auroc_inc = roc_auc_score(y_te, all_probs_inc, multi_class='ovr', average='macro')

classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
print('Per-class AUROC (Inception+Residual Fusion):')
for c, a in zip(classes, per_class_inc):
    print(f'  {c:5s} {a:.4f}')
print(f'  macro AUROC {macro_auroc_inc:.4f}')

# %%  per-class ROC curves
y_true_bin = label_binarize(y_te, classes=[0, 1, 2, 3, 4])

plt.figure(figsize=(7, 7))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs_inc[:, i])
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Inception+Residual Fusion — One-vs-Rest ROC Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('week5task4_inception_roc.png', dpi=150)
plt.show()

# %%
torch.save(model_inc_fuse.state_dict(), 'week5task4_inception_fusion.pt')
np.savez('week5task4_inception_test_probs.npz', probs=all_probs_inc, y_true=y_te)
print('saved week5task4_inception_fusion.pt and test probs')