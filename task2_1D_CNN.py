# Task 2 — signal encoder: 1D CNN on the 12-lead ECG. Self-contained.
# %%
import copy
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, roc_curve

torch.manual_seed(42)
np.random.seed(42)

# %%  labels -> fold split -> load 100 Hz signals
df = pd.read_csv('./ptb-xl/ptbxl_labeled.csv', index_col='ecg_id')
train_df = df[df['strat_fold'] <= 8]
val_df   = df[df['strat_fold'] == 9]
test_df  = df[df['strat_fold'] == 10]
print(len(train_df), len(val_df), len(test_df))

def load_split(split_df):
    X, y = [], []
    for _, row in split_df.iterrows():
        sig, _ = wfdb.rdsamp('./ptb-xl/' + row['filename_lr'])   # (1000, 12)
        X.append(sig.T)                                          # (12, 1000)
        y.append(row['label'])
    return np.array(X, dtype='float32'), np.array(y, dtype='int64')

X_train, y_train = load_split(train_df)
X_val,   y_val   = load_split(val_df)
X_test,  y_test  = load_split(test_df)
print('train:', X_train.shape, np.bincount(y_train))

# %%  model
class ECGCNN(nn.Module):
    def __init__(self, n_classes=2, p_drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32,  kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64,  kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.drop  = nn.Dropout(p_drop)
        self.fc    = nn.Linear(128, n_classes)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # (B, 32, 500)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # (B, 64, 250)
        x = self.relu(self.bn3(self.conv3(x)))              # (B, 128, 250)
        x = self.gap(x).flatten(1)                          # (B, 128)
        x = self.drop(x)
        return self.fc(x)                                   # (B, 2) logits

# %%  tensors, loaders, class weights
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_val),   torch.tensor(y_val)),   batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(X_test),  torch.tensor(y_test)),  batch_size=32, shuffle=False)

weights = torch.tensor(
    compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train),
    dtype=torch.float32)
print('class weights:', weights)

# %%  train (best-val checkpoint)
model = ECGCNN(n_classes=2, p_drop=0.3)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []
best_val, best_state = float('inf'), None

for epoch in range(30):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_loader.dataset)

    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            running += criterion(model(xb), yb).item() * xb.size(0)
    val_loss = running / len(val_loader.dataset)

    train_losses.append(train_loss); val_losses.append(val_loss)
    if val_loss < best_val:
        best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
    print(f'epoch {epoch+1:2d}/30   train {train_loss:.4f}   val {val_loss:.4f}')

model.load_state_dict(best_state)
print(f'best val loss: {best_val:.4f}')

# %%  loss curves
plt.figure()
plt.plot(range(1, 31), train_losses, marker='o', label='train')
plt.plot(range(1, 31), val_losses,   marker='o', label='validation')
plt.xlabel('epoch'); plt.ylabel('loss')
plt.title('Task 2 - training vs validation loss')
plt.legend(); plt.grid(True)
plt.savefig('task2_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  evaluate on the test set
model.eval()
all_probs, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_probs.append(torch.softmax(model(xb), dim=1))
        all_true.append(yb)
all_probs = torch.cat(all_probs).numpy()
all_true  = torch.cat(all_true).numpy()

mi_probs = all_probs[:, 1]
auc = roc_auc_score(all_true, mi_probs)
print(f'Test AUROC (MI vs Normal): {auc:.4f}')

fpr, tpr, _ = roc_curve(all_true, mi_probs)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC (AUROC = {auc:.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Task 2 - ROC (MI vs Normal, test set)')
plt.legend(); plt.grid(True)
plt.savefig('task2_roc.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
torch.save(model.state_dict(), 'ecg_cnn_branch.pt')
print('saved ECG signal branch')