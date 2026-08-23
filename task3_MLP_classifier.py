# Task 3 — tabular encoder: MLP on demographics (age + sex). Self-contained.
# %%
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, roc_curve

torch.manual_seed(42)
np.random.seed(42)

# %%  features: z-score age with TRAIN stats only; sex as-is
df = pd.read_csv('./ptb-xl/ptbxl_labeled.csv', index_col='ecg_id')
train_df = df[df['strat_fold'] <= 8]
val_df   = df[df['strat_fold'] == 9]
test_df  = df[df['strat_fold'] == 10]

age_mean, age_std = train_df['age'].mean(), train_df['age'].std()

def make_xy(split_df):
    feats = split_df[['age', 'sex']].copy()
    feats['age'] = (feats['age'] - age_mean) / age_std
    return feats.values.astype('float32'), split_df['label'].values.astype('int64')

Xtr, ytr = make_xy(train_df)
Xva, yva = make_xy(val_df)
Xte, yte = make_xy(test_df)
print(Xtr.shape, Xva.shape, Xte.shape)

# %%  model
class TabularMLP(nn.Module):
    def __init__(self, in_features=2, n_classes=2, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(32, 16),          nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)

# %%  loaders + weights
train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)), batch_size=32, shuffle=False)

weights = torch.tensor(
    compute_class_weight('balanced', classes=np.array([0, 1]), y=ytr),
    dtype=torch.float32)

# %%  train (best-val checkpoint)
model_mlp = TabularMLP(in_features=2, n_classes=2)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-3)

tr_losses, va_losses = [], []
best_val, best_state = float('inf'), None

for epoch in range(50):
    model_mlp.train()
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model_mlp(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    tr_loss = running / len(train_loader.dataset)

    model_mlp.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            running += criterion(model_mlp(xb), yb).item() * xb.size(0)
    va_loss = running / len(val_loader.dataset)

    tr_losses.append(tr_loss); va_losses.append(va_loss)
    if va_loss < best_val:
        best_val, best_state = va_loss, copy.deepcopy(model_mlp.state_dict())
    if (epoch + 1) % 5 == 0:
        print(f'epoch {epoch+1:2d}/50   train {tr_loss:.4f}   val {va_loss:.4f}')

model_mlp.load_state_dict(best_state)
print(f'best val loss: {best_val:.4f}')

# %%  loss curves
plt.figure()
plt.plot(range(1, 51), tr_losses, label='train')
plt.plot(range(1, 51), va_losses, label='validation')
plt.xlabel('epoch'); plt.ylabel('loss')
plt.title('Task 3 - MLP training vs validation loss')
plt.legend(); plt.grid(True)
plt.savefig('task3_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  evaluate on the test set
model_mlp.eval()
all_probs, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_probs.append(torch.softmax(model_mlp(xb), dim=1))
        all_true.append(yb)
all_probs = torch.cat(all_probs).numpy()
all_true  = torch.cat(all_true).numpy()

mi_probs = all_probs[:, 1]
auc = roc_auc_score(all_true, mi_probs)
print(f'Test AUROC (tabular, MI vs Normal): {auc:.4f}')

fpr, tpr, _ = roc_curve(all_true, mi_probs)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC (AUROC = {auc:.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Task 3 - ROC (tabular: age + sex)')
plt.legend(); plt.grid(True)
plt.savefig('task3_roc.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
torch.save(model_mlp.state_dict(), 'tabular_mlp_branch.pt')
print('saved tabular branch')