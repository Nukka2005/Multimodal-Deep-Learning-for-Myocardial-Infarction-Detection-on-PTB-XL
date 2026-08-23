# ============================================================
# Week 6 - fixing the training curves: Transformer fusion
#
# Diagnosis from the old run: severe overfitting. Train loss fell to ~0.15
# while validation loss climbed past 2.5, best checkpoint around epoch 5-7.
# The Transformer (weak inductive bias, many params) is the culprit, not the CNN.
#
# This file trains the SAME architecture with regularization added, for 300
# epochs, and plots the old 30-epoch curve next to the new one.
#
# Needs: ptb-xl/ptbxl_5class.csv + cache/*.npy  (run week4.py once first)
# ============================================================
# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
# %%
import torch.backends.cuda as cuda_backend
# disable the fused attention kernels that trigger the misaligned-address bug
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)   # fall back to the plain, stable implementation

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# ------------------------------------------------------------
# ===== CHANGES AT A GLANCE (all marked  # <<< CHANGE  below) =====
#   1. EPOCHS 30 -> 300                     (see the real long-run behaviour)
#   2. Dropout 0.3 -> 0.5                   (more regularization on the flexible parts)
#   3. Transformer d_model 128, nhead 8->4, layers 2->1, ffn 256->128
#                                           (shrink the OVER-CAPACITY component)
#   4. Adam -> AdamW with weight_decay=1e-2 (real weight decay on all params)
#   5. Fixed lr -> CosineAnnealingLR + warmup (stop the rush to a bad minimum)
#   6. label_smoothing=0.1 in the loss      (discourage over-confident memorizing)
#   7. batch 32 -> 64                       (smoother gradients, fewer val spikes)
# ------------------------------------------------------------

EPOCHS = 300          # <<< CHANGE 1: was 30
BATCH  = 64           # <<< CHANGE 7: was 32
DROPOUT = 0.5         # <<< CHANGE 2: was 0.3
WEIGHT_DECAY = 1e-2   # <<< CHANGE 4: new
MAX_LR = 1e-3
WARMUP_EPOCHS = 10    # <<< CHANGE 5: new

# %% ===== data (unchanged from week4/week5) =====
df5 = pd.read_csv('./ptb-xl/ptbxl_5class.csv', index_col='ecg_id')
tr_df = df5[df5['strat_fold'] <= 8]
va_df = df5[df5['strat_fold'] == 9]
te_df = df5[df5['strat_fold'] == 10]

X_tr, y_tr = np.load('./cache/X_tr.npy'), np.load('./cache/y_tr.npy')
X_va, y_va = np.load('./cache/X_va.npy'), np.load('./cache/y_va.npy')
X_te, y_te = np.load('./cache/X_te.npy'), np.load('./cache/y_te.npy')

features = ['age', 'sex', 'height', 'weight']
imputer = SimpleImputer(strategy='median')
T_tr = imputer.fit_transform(tr_df[features])
T_va = imputer.transform(va_df[features])
T_te = imputer.transform(te_df[features])
scaler = StandardScaler()
T_tr = scaler.fit_transform(T_tr).astype('float32')
T_va = scaler.transform(T_va).astype('float32')
T_te = scaler.transform(T_te).astype('float32')

weights5 = torch.tensor(
    compute_class_weight('balanced', classes=np.arange(5), y=y_tr),
    dtype=torch.float32).to(device)


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, X_sig, X_tab, y):
        self.X_sig = torch.tensor(X_sig, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_sig[i], self.X_tab[i], self.y[i]


train_loader = DataLoader(MultimodalDataset(X_tr, T_tr, y_tr), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(MultimodalDataset(X_va, T_va, y_va), batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(MultimodalDataset(X_te, T_te, y_te), batch_size=BATCH, shuffle=False)


# %% ===== model (same shape as week5, capacity dialed down) =====
class TransformerFusionModel(nn.Module):
    def __init__(self, n_classes=5, p_drop=DROPOUT):
        super().__init__()
        # --- CNN backbone (UNCHANGED - it is not the overfitting culprit) ---
        self.conv1 = nn.Conv1d(12, 32, 7, padding=3); self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2); self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # --- Transformer (SHRUNK: this is the over-capacity part) ---
        self.pos_emb = nn.Parameter(torch.randn(1, 250, 128) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,                 # <<< CHANGE 3: was 8
            dim_feedforward=128,     # <<< CHANGE 3: was 256
            batch_first=True,
            dropout=p_drop,          # <<< CHANGE 2: 0.5 flows in here
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)  # <<< CHANGE 3: was 2

        # --- tabular branch (unchanged) ---
        self.fc_tab = nn.Linear(4, 16)

        # --- fusion head ---
        self.drop = nn.Dropout(p_drop)            # <<< CHANGE 2: 0.5
        self.fc_fusion = nn.Linear(128 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        s = self.pool(self.relu(self.bn1(self.conv1(x_sig))))
        s = self.pool(self.relu(self.bn2(self.conv2(s))))
        s = self.relu(self.bn3(self.conv3(s)))
        s = s.permute(0, 2, 1)                    # (B, 250, 128)
        s = s + self.pos_emb
        s = self.transformer(s)
        emb_sig = s.mean(dim=1)                   # (B, 128)
        emb_tab = F.relu(self.fc_tab(x_tab))      # (B, 16)
        fused = torch.cat([emb_sig, emb_tab], dim=1)
        return self.fc_fusion(self.drop(fused))


# %% ===== training with warmup + cosine schedule =====
def train(model, epochs):
    # <<< CHANGE 6: label_smoothing added
    criterion = nn.CrossEntropyLoss(weight=weights5, label_smoothing=0.1)
    # <<< CHANGE 4: AdamW + weight_decay (was plain Adam)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    # <<< CHANGE 5: linear warmup for WARMUP_EPOCHS, then cosine anneal to ~0
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    tr_losses, va_losses, lrs = [], [], []
    best_val, best_state, best_epoch = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x_sig, x_tab, yb in train_loader:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_sig, x_tab), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * yb.size(0)
        tr = running / len(train_loader.dataset)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for x_sig, x_tab, yb in val_loader:
                x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
                running += criterion(model(x_sig, x_tab), yb).item() * yb.size(0)
        va = running / len(val_loader.dataset)

        tr_losses.append(tr); va_losses.append(va)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()                          # <<< CHANGE 5: steps per EPOCH here

        # best-checkpointing kept (we disabled early STOPPING, not checkpointing)
        if va < best_val:
            best_val, best_epoch = va, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'epoch {epoch+1:3d}/{epochs}   train {tr:.4f}   val {va:.4f}   lr {lrs[-1]:.2e}')

    model.load_state_dict(best_state)
    print(f'best val {best_val:.4f} at epoch {best_epoch}')
    return tr_losses, va_losses, lrs, best_epoch


model = TransformerFusionModel().to(device)
print(model(torch.randn(2, 12, 1000).to(device), torch.randn(2, 4).to(device)).shape)
print('params:', sum(p.numel() for p in model.parameters()))

tr_losses, va_losses, lrs, best_epoch = train(model, EPOCHS)
torch.save(model.state_dict(), 'week6_transformer_fusion_fixed.pt')
np.savez('week6_transformer_curves.npz',
         tr=tr_losses, va=va_losses, lr=lrs, best_epoch=best_epoch)


# %% ===== evaluate =====
model.eval()
probs = []
with torch.no_grad():
    for x_sig, x_tab, yb in test_loader:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        probs.append(F.softmax(model(x_sig, x_tab), dim=1).cpu().numpy())
probs = np.concatenate(probs)

macro = roc_auc_score(y_te, probs, multi_class='ovr', average='macro')
per = roc_auc_score(y_te, probs, multi_class='ovr', average=None)
print(f'\nfixed Transformer fusion - macro AUROC {macro:.4f}')
for c, a in zip(CLASSES, per):
    print(f'  {c:5s} {a:.4f}')


# %% ===== NEW curve (300 epochs) =====
plt.figure(figsize=(9, 5))
ep = range(1, len(tr_losses) + 1)
plt.plot(ep, tr_losses, linewidth=1.5, label='train')
plt.plot(ep, va_losses, linewidth=1.5, label='validation')
plt.axvline(best_epoch, color='red', linestyle=':', alpha=0.7, label=f'best (epoch {best_epoch})')
plt.xlabel('epoch'); plt.ylabel('cross-entropy loss')
plt.title('Transformer fusion - FIXED (300 epochs, regularized)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('week6_transformer_fixed_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== OLD vs NEW side by side (for the meeting) =====
# paste your old 30-epoch numbers here, OR load them if you saved an npz last time.
# these are the values you reported from the week5 run:
old_note = ("Old run: 30 epochs, train fell to ~0.15 while validation climbed past 2.5, "
            "best epoch ~5-7. Severe divergence.")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# left: schematic of the old behaviour if you don't have the saved arrays
axes[0].set_title('BEFORE - 30 epochs, no extra regularization')
try:
    old = np.load('week5_transformer_curves.npz')   # only if you saved it last week
    axes[0].plot(range(1, len(old['tr']) + 1), old['tr'], label='train')
    axes[0].plot(range(1, len(old['va']) + 1), old['va'], label='validation')
except FileNotFoundError:
    axes[0].text(0.5, 0.5, old_note, ha='center', va='center', wrap=True,
                 transform=axes[0].transAxes, fontsize=10)
    axes[0].set_xlim(0, 30)
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)

# right: the new fixed run
axes[1].set_title('AFTER - 300 epochs, regularized')
axes[1].plot(ep, tr_losses, label='train')
axes[1].plot(ep, va_losses, label='validation')
axes[1].axvline(best_epoch, color='red', linestyle=':', alpha=0.7, label=f'best (epoch {best_epoch})')
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('loss')
axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('week6_transformer_before_after.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== LR schedule (confirms warmup then cosine) =====
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(lrs) + 1), lrs, linewidth=2)
plt.xlabel('epoch'); plt.ylabel('learning rate')
plt.title('Warmup + cosine schedule')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('week6_transformer_lr.png', dpi=150, bbox_inches='tight')
plt.show()