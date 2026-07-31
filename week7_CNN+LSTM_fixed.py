# ============================================================
# Week 7 - the healthy-curve exhibit: CNN + BiLSTM fusion, 300 epochs
#
# Purpose: show that a model with strong inductive bias (convolution's locality +
# the LSTM's built-in sequential ordering) trains CLEANLY on the same ~11.5k
# recordings where the Transformer overfits by epoch 14. Same regularization
# treatment as the Transformer runs, so the comparison is fair.
#
# Expected shape: train and validation both fall and stay close, validation
# stabilising rather than climbing away. This is the "curves that make sense" the
# assignment asks for, and the contrast that explains WHY the Transformer could not.
#
# Paths are set for Colab + Drive. Flip COLAB = False to run locally.
# ============================================================
# %%
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import google

torch.manual_seed(42)
np.random.seed(42)

# ---- paths: Colab (Drive) vs local ----
COLAB = False
if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/PTBXL/'   # <-- adjust if your Drive folder differs
else:
    ROOT = './'

DATA_DIR  = ROOT + 'ptb-xl/'
CACHE_DIR = ROOT + 'cache/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
if device.type == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

EPOCHS = 300
BATCH  = 64
DROPOUT = 0.5
WEIGHT_DECAY = 1e-2
MAX_LR = 1e-3
WARMUP_EPOCHS = 10

# augmentation (train only) - same knobs as the Transformer run
AUG_SHIFT  = 50
AUG_NOISE  = 0.02
AUG_SCALE  = 0.10
AUG_LEAD_P = 0.10

# %% ===== data =====
df5 = pd.read_csv(DATA_DIR + 'ptbxl_5class.csv', index_col='ecg_id')
tr_df = df5[df5['strat_fold'] <= 8]
va_df = df5[df5['strat_fold'] == 9]
te_df = df5[df5['strat_fold'] == 10]

X_tr, y_tr = np.load(CACHE_DIR + 'X_tr.npy'), np.load(CACHE_DIR + 'y_tr.npy')
X_va, y_va = np.load(CACHE_DIR + 'X_va.npy'), np.load(CACHE_DIR + 'y_va.npy')
X_te, y_te = np.load(CACHE_DIR + 'X_te.npy'), np.load(CACHE_DIR + 'y_te.npy')
print('signals:', X_tr.shape, X_va.shape, X_te.shape)

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


def augment_batch(x):
    """Train-only ECG augmentation on (B, 12, 1000)."""
    B, C, L = x.shape
    shifts = torch.randint(-AUG_SHIFT, AUG_SHIFT + 1, (B,), device=x.device)
    idx = (torch.arange(L, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)) % L
    x = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, C, -1))
    scale = 1.0 + (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * AUG_SCALE
    x = x * scale
    x = x + torch.randn_like(x) * AUG_NOISE
    lead_mask = (torch.rand(B, C, 1, device=x.device) > AUG_LEAD_P).float()
    return x * lead_mask


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


# %% ===== model: CNN + BiLSTM fusion =====
class CNNLSTMFusion(nn.Module):
    def __init__(self, n_classes=5, hidden=128, p_drop=DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, 7, padding=3); self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2); self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(128, hidden, num_layers=1, batch_first=True, bidirectional=True)

        self.fc_tab = nn.Linear(4, 16)
        self.drop = nn.Dropout(p_drop)
        self.fc_fusion = nn.Linear(hidden * 2 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        s = self.pool(self.relu(self.bn1(self.conv1(x_sig))))
        s = self.pool(self.relu(self.bn2(self.conv2(s))))
        s = self.relu(self.bn3(self.conv3(s)))
        s = s.permute(0, 2, 1)                       # (B, 250, 128)
        _, (h_n, _) = self.lstm(s)
        emb_sig = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, 256)
        emb_tab = F.relu(self.fc_tab(x_tab))             # (B, 16)
        fused = torch.cat([emb_sig, emb_tab], dim=1)
        return self.fc_fusion(self.drop(fused))


# %% ===== training (same treatment as the Transformer runs) =====
def train(model, epochs, augment=True):
    criterion = nn.CrossEntropyLoss(weight=weights5, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    tr_losses, va_losses, lrs = [], [], []
    best_val, best_state, best_epoch = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x_sig, x_tab, yb in train_loader:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            if augment:
                x_sig = augment_batch(x_sig)
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
        scheduler.step()

        if va < best_val:
            best_val, best_epoch = va, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'epoch {epoch+1:3d}/{epochs}   train {tr:.4f}   val {va:.4f}   lr {lrs[-1]:.2e}')

    model.load_state_dict(best_state)
    print(f'best val {best_val:.4f} at epoch {best_epoch}')
    return tr_losses, va_losses, lrs, best_epoch


model = CNNLSTMFusion().to(device)
print(model(torch.randn(2, 12, 1000).to(device), torch.randn(2, 4).to(device)).shape)

tr_losses, va_losses, lrs, best_epoch = train(model, EPOCHS, augment=True)
torch.save(model.state_dict(), ROOT + 'week7_cnn_lstm_fusion.pt')
np.savez(ROOT + 'week7_cnn_lstm_curves.npz', tr=tr_losses, va=va_losses, lr=lrs, best_epoch=best_epoch)


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
print(f'\nCNN+BiLSTM fusion - macro AUROC {macro:.4f}')
for c, a in zip(CLASSES, per):
    print(f'  {c:5s} {a:.4f}')


# %% ===== the healthy curve =====
plt.figure(figsize=(9, 5))
ep = range(1, len(tr_losses) + 1)
plt.plot(ep, tr_losses, linewidth=1.5, label='train')
plt.plot(ep, va_losses, linewidth=1.5, label='validation')
plt.axvline(best_epoch, color='red', linestyle=':', alpha=0.7, label=f'best (epoch {best_epoch})')
plt.xlabel('epoch'); plt.ylabel('cross-entropy loss')
plt.title('CNN + BiLSTM fusion - 300 epochs (strong inductive bias)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(ROOT + 'week7_cnn_lstm_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== the money shot: LSTM vs Transformer, side by side =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].set_title('Transformer fusion (weak inductive bias)')
try:
    t = np.load(ROOT + 'week7_transformer_curves.npz')
    axes[0].plot(range(1, len(t['tr']) + 1), t['tr'], label='train')
    axes[0].plot(range(1, len(t['va']) + 1), t['va'], label='validation')
    axes[0].axvline(int(t['best_epoch']), color='red', linestyle=':', alpha=0.7,
                    label=f"best (epoch {int(t['best_epoch'])})")
except FileNotFoundError:
    axes[0].text(0.5, 0.5, 'week7_transformer_curves.npz not found', ha='center', va='center',
                 transform=axes[0].transAxes)
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].set_title('CNN + BiLSTM fusion (strong inductive bias)')
axes[1].plot(ep, tr_losses, label='train')
axes[1].plot(ep, va_losses, label='validation')
axes[1].axvline(best_epoch, color='red', linestyle=':', alpha=0.7, label=f'best (epoch {best_epoch})')
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('loss')
axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)

# share y-limits so the contrast is honest, not a plotting artifact
lo = min(min(tr_losses), min(va_losses))
hi = max(max(va_losses), 2.1)
for ax in axes:
    ax.set_ylim(lo - 0.05, hi)

plt.tight_layout()
plt.savefig(ROOT + 'week7_lstm_vs_transformer_curves.png', dpi=150, bbox_inches='tight')
plt.show()