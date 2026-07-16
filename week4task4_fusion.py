# Week 4, Task 4 — multimodal fusion (signal + tabular), trained from scratch. Self-contained.
# Requires ./ptb-xl/ptbxl_5class.csv (run week4task1_label_extraction.py once to create it).
# For the 3-way comparison plot it optionally loads week4task1_test_probs.npz and
# week4task3_test_probs.npz (produced by week4task1_cnn_lstm.py / week4task3_tabular_mlp.py);
# if either is missing, that curve is skipped with a warning rather than failing.
# Saves week4task4_fusion.pt (weights) and week4task4_fusion_test_probs.npz (test-set probs,
# consumed later by week4task4_transfer_fusion.py for the ultimate comparison plot).
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
print('train:', X_tr.shape, np.bincount(y_tr))

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

# %%  multimodal dataset + loaders
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
print('Multimodal DataLoaders ready.')

# %%  model
class FusionModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        # --- Signal Encoder Backbone ---
        self.conv1 = nn.Conv1d(12, 32, 7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()
        self.lstm  = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=True)

        # --- Tabular Encoder Backbone ---
        self.fc_tab = nn.Linear(4, 16)

        # --- Fusion Head ---
        # 256 from bidirectional LSTM + 16 from MLP = 272
        self.drop = nn.Dropout(0.3)
        self.fc_fusion = nn.Linear(256 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        s = self.pool(self.relu(self.bn1(self.conv1(x_sig))))
        s = self.pool(self.relu(self.bn2(self.conv2(s))))
        s = self.relu(self.bn3(self.conv3(s)))
        s = s.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(s)
        emb_sig = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, 256)

        emb_tab = self.relu(self.fc_tab(x_tab))           # (B, 16)

        emb_fused = torch.cat([emb_sig, emb_tab], dim=1)  # (B, 272)
        emb_fused = self.drop(emb_fused)
        return self.fc_fusion(emb_fused)

model_fuse = FusionModel().to(device)
criterion_fuse = nn.CrossEntropyLoss(weight=weights5)
optimizer_fuse = torch.optim.Adam(model_fuse.parameters(), lr=1e-3)

# %%  train (best-val checkpoint)
tr_losses_fuse, va_losses_fuse = [], []
best_val_fuse, best_state_fuse = float('inf'), None

for epoch in range(30):
    model_fuse.train()
    running = 0.0
    for x_sig, x_tab, yb in train_loader_fuse:
        x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)

        optimizer_fuse.zero_grad()
        loss = criterion_fuse(model_fuse(x_sig, x_tab), yb)
        loss.backward()
        optimizer_fuse.step()
        running += loss.item() * yb.size(0)
    tr = running / len(train_loader_fuse.dataset)

    model_fuse.eval()
    running = 0.0
    with torch.no_grad():
        for x_sig, x_tab, yb in val_loader_fuse:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            running += criterion_fuse(model_fuse(x_sig, x_tab), yb).item() * yb.size(0)
    va = running / len(val_loader_fuse.dataset)

    tr_losses_fuse.append(tr); va_losses_fuse.append(va)
    if va < best_val_fuse:
        best_val_fuse, best_state_fuse = va, copy.deepcopy(model_fuse.state_dict())

    print(f'epoch {epoch+1:2d}/30   train {tr:.4f}   val {va:.4f}')

model_fuse.load_state_dict(best_state_fuse)
print(f'best fusion val loss: {best_val_fuse:.4f}')

# %%  evaluate on the test set
model_fuse.eval()
all_pred_fuse, all_probs_fuse = [], []
with torch.no_grad():
    for x_sig, x_tab, yb in test_loader_fuse:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        outputs = model_fuse(x_sig, x_tab)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_pred_fuse.extend(preds.cpu().numpy())
        all_probs_fuse.extend(probs.cpu().numpy())

all_pred_fuse = np.array(all_pred_fuse)
all_probs_fuse = np.array(all_probs_fuse)

macro_auroc_fuse = roc_auc_score(y_te, all_probs_fuse, multi_class='ovr', average='macro')
print(f'Fusion Macro AUROC: {macro_auroc_fuse:.4f}')

# %%  3-way ROC comparison (micro-average) — signal-only / tabular-only loaded from disk if available
y_true_bin_fuse = label_binarize(y_te, classes=[0, 1, 2, 3, 4])

plt.figure(figsize=(8, 8))

if os.path.exists('week4task1_test_probs.npz'):
    sig = np.load('week4task1_test_probs.npz')
    y_true_bin_sig = label_binarize(sig['y_true'], classes=[0, 1, 2, 3, 4])
    fpr_sig, tpr_sig, _ = roc_curve(y_true_bin_sig.ravel(), sig['probs'].ravel())
    plt.plot(fpr_sig, tpr_sig, label=f'Signal Only (Micro AUC = {auc(fpr_sig, tpr_sig):.3f})', color='blue', linewidth=2)
else:
    print('warning: week4task1_test_probs.npz not found -> run week4task1_cnn_lstm.py to include the signal-only curve')

if os.path.exists('week4task3_test_probs.npz'):
    tab = np.load('week4task3_test_probs.npz')
    y_true_bin_tab = label_binarize(tab['y_true'], classes=[0, 1, 2, 3, 4])
    fpr_tab, tpr_tab, _ = roc_curve(y_true_bin_tab.ravel(), tab['probs'].ravel())
    plt.plot(fpr_tab, tpr_tab, label=f'Tabular Only (Micro AUC = {auc(fpr_tab, tpr_tab):.3f})', color='orange', linewidth=2)
else:
    print('warning: week4task3_test_probs.npz not found -> run week4task3_tabular_mlp.py to include the tabular-only curve')

fpr_fuse, tpr_fuse, _ = roc_curve(y_true_bin_fuse.ravel(), all_probs_fuse.ravel())
plt.plot(fpr_fuse, tpr_fuse, label=f'Fusion (Micro AUC = {auc(fpr_fuse, tpr_fuse):.3f})', color='green', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multimodal Comparison: Micro-Average ROC Curves')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('week4task4_multimodal_comparison.png', dpi=150)
plt.show()

# %%  persist weights + test-set probs for the transfer-learning fusion script
torch.save(model_fuse.state_dict(), 'week4task4_fusion.pt')
np.savez('week4task4_fusion_test_probs.npz', y_true=y_te, probs=all_probs_fuse)
print('saved week4task4_fusion.pt and week4task4_fusion_test_probs.npz')
