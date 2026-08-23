# Week 4, Task 3 — tabular encoder: MLP on age/sex/height/weight, 5-class softmax. Self-contained.
# Requires ./ptb-xl/ptbxl_5class.csv (run week4task1_label_extraction.py once to create it).
# Saves week4task3_tabular_mlp.pt (weights) and week4task3_test_probs.npz (test-set probs,
# consumed later by week4task4_fusion.py / week4task4_transfer_fusion.py for comparison plots).
# %%
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

torch.manual_seed(42)
np.random.seed(42)

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using', device)

# %%  labels -> fold split
df5 = pd.read_csv('./ptb-xl/ptbxl_5class.csv', index_col='ecg_id')
tr_df = df5[df5['strat_fold'] <= 8]
va_df = df5[df5['strat_fold'] == 9]
te_df = df5[df5['strat_fold'] == 10]

y_tr = tr_df['label'].values.astype('int64')
y_va = va_df['label'].values.astype('int64')
y_te = te_df['label'].values.astype('int64')

weights5 = torch.tensor(
    compute_class_weight('balanced', classes=np.array([0, 1, 2, 3, 4]), y=y_tr),
    dtype=torch.float32).to(device)

# %%  tabular features: impute (median, fit on train) -> scale (fit on train)
features = ['age', 'sex', 'height', 'weight']

X_tr_tab_raw = tr_df[features].copy()
X_va_tab_raw = va_df[features].copy()
X_te_tab_raw = te_df[features].copy()

imputer = SimpleImputer(strategy='median')
X_tr_imp = imputer.fit_transform(X_tr_tab_raw)
X_va_imp = imputer.transform(X_va_tab_raw)
X_te_imp = imputer.transform(X_te_tab_raw)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_imp)
X_va_scaled = scaler.transform(X_va_imp)
X_te_scaled = scaler.transform(X_te_imp)

print('Tabular data ready. Shape:', X_tr_scaled.shape)

train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_tr)), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_va_scaled, dtype=torch.float32), torch.tensor(y_va)), batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32), torch.tensor(y_te)), batch_size=32, shuffle=False)

# %%  model
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

model_tab = TabularMLP().to(device)
criterion_tab = nn.CrossEntropyLoss(weight=weights5)
optimizer_tab = torch.optim.Adam(model_tab.parameters(), lr=1e-3)

# %%  train (best-val checkpoint)
tr_losses_tab, va_losses_tab = [], []
best_val_tab, best_state_tab = float('inf'), None

for epoch in range(30):
    model_tab.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_tab.zero_grad()
        loss = criterion_tab(model_tab(xb), yb)
        loss.backward()
        optimizer_tab.step()
        running += loss.item() * xb.size(0)
    tr = running / len(train_loader.dataset)

    model_tab.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            running += criterion_tab(model_tab(xb), yb).item() * xb.size(0)
    va = running / len(val_loader.dataset)

    tr_losses_tab.append(tr); va_losses_tab.append(va)
    if va < best_val_tab:
        best_val_tab, best_state_tab = va, copy.deepcopy(model_tab.state_dict())

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'epoch {epoch+1:2d}/30   train {tr:.4f}   val {va:.4f}')

model_tab.load_state_dict(best_state_tab)
print(f'best val loss: {best_val_tab:.4f}')

# %%  loss curves
epochs_tab = range(1, len(tr_losses_tab) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs_tab, tr_losses_tab, marker='o', linewidth=2, label='Training Loss')
plt.plot(epochs_tab, va_losses_tab, marker='s', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Cross-Entropy Loss')
plt.title('Tabular MLP: Training and Validation Loss')
plt.xticks(epochs_tab)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('week4task3_tabular_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  evaluate on the test set
model_tab.eval()
all_true_tab, all_pred_tab, all_probs_tab = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model_tab(xb)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_true_tab.extend(yb.numpy())
        all_pred_tab.extend(preds.cpu().numpy())
        all_probs_tab.extend(probs.cpu().numpy())

all_true_tab = np.array(all_true_tab)
all_pred_tab = np.array(all_pred_tab)
all_probs_tab = np.array(all_probs_tab)

macro_auroc_tab = roc_auc_score(all_true_tab, all_probs_tab, multi_class='ovr', average='macro')
print('Tabular Test Accuracy:', accuracy_score(all_true_tab, all_pred_tab))
print(f'Tabular Macro AUROC: {macro_auroc_tab:.4f}')

# %%  ROC (one-vs-rest)
y_true_bin_tab = label_binarize(all_true_tab, classes=[0, 1, 2, 3, 4])
plt.figure(figsize=(7, 7))
for i, cls in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin_tab[:, i], all_probs_tab[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Tabular Model: One-vs-Rest ROC Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('week4task3_tabular_roc.png', dpi=150)
plt.show()

# %%  persist weights + test-set probs for downstream fusion scripts
torch.save(model_tab.state_dict(), 'week4task3_tabular_mlp.pt')
np.savez('week4task3_test_probs.npz', y_true=all_true_tab, probs=all_probs_tab)
print('saved week4task3_tabular_mlp.pt and week4task3_test_probs.npz')
