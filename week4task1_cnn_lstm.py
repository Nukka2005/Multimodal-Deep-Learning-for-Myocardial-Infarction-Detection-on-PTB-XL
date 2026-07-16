# Week 4, Task 1 — signal encoder: CNN-LSTM on 12-lead ECG, 5-class softmax. Self-contained.
# Requires ./ptb-xl/ptbxl_5class.csv (run week4task1_label_extraction.py once to create it).
# Saves week4task1_cnn_lstm.pt (weights) and week4task1_test_probs.npz (test-set probs,
# consumed later by week4task4_fusion.py / week4task4_transfer_fusion.py for comparison plots).
# %%
import copy
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

torch.manual_seed(42)
np.random.seed(42)

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using', device)

# %%  labels -> fold split -> load 100 Hz signals
df5 = pd.read_csv('./ptb-xl/ptbxl_5class.csv', index_col='ecg_id')
tr_df = df5[df5['strat_fold'] <= 8]
va_df = df5[df5['strat_fold'] == 9]
te_df = df5[df5['strat_fold'] == 10]
print(len(tr_df), len(va_df), len(te_df))

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

train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)), batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=32, shuffle=False)

# class weights now span all 5 classes -> HYP gets the biggest weight
weights5 = torch.tensor(
    compute_class_weight('balanced', classes=np.array([0, 1, 2, 3, 4]), y=y_tr),
    dtype=torch.float32).to(device)
print('class weights:', weights5)

# %%  model
class CNNLSTM(nn.Module):
    def __init__(self, n_classes=5, hidden_size=128, p_drop=0.3, bidirectional=True):
        super().__init__()
        # --- CNN feature extractor ---
        self.conv1 = nn.Conv1d(12, 32,  kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64,  kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()

        # --- LSTM sequence reader ---
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # --- classifier head ---
        lstm_out = hidden_size * (2 if bidirectional else 1)   # 256 if bidirectional
        self.drop = nn.Dropout(p_drop)
        self.fc   = nn.Linear(lstm_out, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # (B, 32, 500)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # (B, 64, 250)
        x = self.relu(self.bn3(self.conv3(x)))              # (B, 128, 250)

        x = x.permute(0, 2, 1)                               # (B, 250, 128)
        outputs, (h_n, c_n) = self.lstm(x)
        emb = torch.cat([h_n[-2], h_n[-1]], dim=1)           # (B, 256)

        emb = self.drop(emb)
        return self.fc(emb)

model = CNNLSTM(n_classes=5, hidden_size=128, bidirectional=True, p_drop=0.3).to(device)
out = model(torch.randn(8, 12, 1000).to(device))
print(out.shape)   # expect torch.Size([8, 5])

# %%  train (best-val checkpoint)
criterion = nn.CrossEntropyLoss(weight=weights5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

tr_losses, va_losses = [], []
best_val, best_state = float('inf'), None

for epoch in range(30):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    tr = running / len(train_loader.dataset)

    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            running += criterion(model(xb), yb).item() * xb.size(0)
    va = running / len(val_loader.dataset)

    tr_losses.append(tr); va_losses.append(va)
    if va < best_val:
        best_val, best_state = va, copy.deepcopy(model.state_dict())
    print(f'epoch {epoch+1:2d}/30   train {tr:.4f}   val {va:.4f}')

model.load_state_dict(best_state)
print(f'best val loss: {best_val:.4f}')

# %%  loss curves
epochs = range(1, len(tr_losses) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, tr_losses, marker='o', linewidth=2, label='Training Loss')
plt.plot(epochs, va_losses, marker='s', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Cross-Entropy Loss')
plt.title('CNN-LSTM Training and Validation Loss')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('week4task1_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  evaluate on the test set
model.eval()
all_true, all_pred, all_probs = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_true.extend(yb.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_probs = np.array(all_probs)

print('Test Accuracy:', accuracy_score(all_true, all_pred))
print('all_true shape:', all_true.shape)
print('all_probs shape:', all_probs.shape)

# %%  multiclass ROC (one-vs-rest)
y_true_bin = label_binarize(all_true, classes=[0, 1, 2, 3, 4])

plt.figure(figsize=(7, 7))
for i, cls in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('One-vs-Rest ROC Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('week4task1_multiclass_roc.png', dpi=150)
plt.show()

macro_auroc = roc_auc_score(all_true, all_probs, multi_class='ovr', average='macro')
print(f'Macro AUROC: {macro_auroc:.4f}')

# %%  persist weights + test-set probs for downstream fusion scripts
torch.save(model.state_dict(), 'week4task1_cnn_lstm.pt')
np.savez('week4task1_test_probs.npz', y_true=all_true, probs=all_probs)
print('saved week4task1_cnn_lstm.pt and week4task1_test_probs.npz')
