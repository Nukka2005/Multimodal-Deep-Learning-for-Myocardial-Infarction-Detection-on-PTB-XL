# ============================================================
# Week 8 - three-modality multi-label fusion
#
# Three action items in one file:
#   (1) MULTI-LABEL: keep records that belong to several classes (bigger dataset,
#       ~21k instead of 16k) and predict each class independently.
#   (2) NEW MODALITY: add the FFT magnitude (frequency domain) as a third branch.
#   (3) ARCHITECTURE: Inception blocks (parallel kernels 3/5/7/9) instead of the
#       plain fixed-kernel CNN, for multi-scale feature extraction.
#
# Branches:
#   raw signal (12,1000) -> 1D Inception -> BiLSTM        -> emb_time (128)
#   FFT magnitude (12,501) -> 1D Inception -> global pool  -> emb_freq (128)
#   demographics (4,) -> MLP                               -> emb_tab  (16)
#   concat -> classifier -> 5 logits  (sigmoid, multi-label)
#
# The FFT is computed inside the model from the (possibly augmented) raw signal,
# so there is nothing extra to cache.
#
# Loss: BCEWithLogitsLoss + pos_weight (multi-label needs independent binary
# decisions, not softmax). Metric: macro AUROC - and because this is now
# multi-label, it is DIRECTLY comparable to the Strodthoff 0.93 benchmark.
#
# Paths set for Colab + Drive. Flip COLAB=False for local.
# ============================================================

# %%
import ast
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc

torch.manual_seed(42)
np.random.seed(42)

COLAB = True
if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/PTBXL/'
else:
    ROOT = './'
DATA_DIR  = ROOT + 'ptb-xl/'
CACHE_DIR = ROOT + 'cache/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

EPOCHS = 300
BATCH  = 64
DROPOUT = 0.5
WEIGHT_DECAY = 1e-2
MAX_LR = 1e-3
WARMUP_EPOCHS = 10

AUG_SHIFT  = 50
AUG_NOISE  = 0.02
AUG_SCALE  = 0.10
AUG_LEAD_P = 0.10


# %% ===== (1) build the MULTI-LABEL dataset =====
meta = pd.read_csv(DATA_DIR + 'ptbxl_database.csv', index_col='ecg_id')
meta['scp_codes'] = meta['scp_codes'].apply(ast.literal_eval)

scp = pd.read_csv(DATA_DIR + 'scp_statements.csv', index_col=0)
scp = scp[scp['diagnostic'] == 1]

def to_superclasses(codes):
    return list({scp.loc[c, 'diagnostic_class'] for c in codes if c in scp.index})

meta['superclasses'] = meta['scp_codes'].apply(to_superclasses)

# KEEP records with 1 OR MORE superclasses (drop only the 0-superclass ones).
# This is the multi-label change: multi-class records are no longer discarded.
multi = meta[meta['superclasses'].apply(len) >= 1].copy()

def multihot(names):
    v = np.zeros(5, dtype='float32')
    for n in names:
        v[CLASS_IDX[n]] = 1.0
    return v

Y = np.stack(multi['superclasses'].apply(multihot).values)   # (N, 5)
multi = multi.assign(**{f'y_{c}': Y[:, i] for i, c in enumerate(CLASSES)})

print('total kept (multi-label):', len(multi), 'vs 16,244 single-label')
print('label counts per class (records can appear in several):')
print(Y.sum(0).astype(int), 'for', CLASSES)
print('records with >1 label:', int((Y.sum(1) > 1).sum()))

tr_df = multi[multi['strat_fold'] <= 8]
va_df = multi[multi['strat_fold'] == 9]
te_df = multi[multi['strat_fold'] == 10]

ycols = [f'y_{c}' for c in CLASSES]
y_tr = tr_df[ycols].values.astype('float32')
y_va = va_df[ycols].values.astype('float32')
y_te = te_df[ycols].values.astype('float32')
print('split:', len(tr_df), len(va_df), len(te_df))


# %% ===== load signals for the multi-label set (own cache, separate from single-label) =====
import os
os.makedirs(CACHE_DIR, exist_ok=True)

def load_signals(split_df):
    import wfdb
    X = []
    for i, (_, row) in enumerate(split_df.iterrows()):
        sig, _ = wfdb.rdsamp(DATA_DIR + row['filename_lr'])
        X.append(sig.T)
        if (i + 1) % 3000 == 0:
            print(f'  {i+1}/{len(split_df)}')
    return np.array(X, dtype='float32')

if os.path.exists(CACHE_DIR + 'ml_X_tr.npy'):
    X_tr = np.load(CACHE_DIR + 'ml_X_tr.npy')
    X_va = np.load(CACHE_DIR + 'ml_X_va.npy')
    X_te = np.load(CACHE_DIR + 'ml_X_te.npy')
    print('loaded ml signals from cache')
else:
    print('caching multi-label signals (a few minutes)...')
    X_tr = load_signals(tr_df); np.save(CACHE_DIR + 'ml_X_tr.npy', X_tr)
    X_va = load_signals(va_df); np.save(CACHE_DIR + 'ml_X_va.npy', X_va)
    X_te = load_signals(te_df); np.save(CACHE_DIR + 'ml_X_te.npy', X_te)
    print('cached')
print('signals:', X_tr.shape, X_va.shape, X_te.shape)


# %% ===== tabular (impute + scale on train only) =====
features = ['age', 'sex', 'height', 'weight']
imputer = SimpleImputer(strategy='median')
T_tr = imputer.fit_transform(tr_df[features])
T_va = imputer.transform(va_df[features])
T_te = imputer.transform(te_df[features])
scaler = StandardScaler()
T_tr = scaler.fit_transform(T_tr).astype('float32')
T_va = scaler.transform(T_va).astype('float32')
T_te = scaler.transform(T_te).astype('float32')

# multi-label class balance: pos_weight = (#neg / #pos) per class, for BCE
pos = y_tr.sum(0)
neg = len(y_tr) - pos
pos_weight = torch.tensor(neg / pos, dtype=torch.float32).to(device)
print('pos_weight:', pos_weight.cpu().numpy().round(2))


# %% ===== augmentation (train only) =====
def augment_batch(x):
    B, C, L = x.shape
    shifts = torch.randint(-AUG_SHIFT, AUG_SHIFT + 1, (B,), device=x.device)
    idx = (torch.arange(L, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)) % L
    x = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, C, -1))
    scale = 1.0 + (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * AUG_SCALE
    x = x * scale
    x = x + torch.randn_like(x) * AUG_NOISE
    lead_mask = (torch.rand(B, C, 1, device=x.device) > AUG_LEAD_P).float()
    return x * lead_mask


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, X, T, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)   # float for BCE

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.T[i], self.y[i]


train_loader = DataLoader(FusionDataset(X_tr, T_tr, y_tr), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(FusionDataset(X_va, T_va, y_va), batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(FusionDataset(X_te, T_te, y_te), batch_size=BATCH, shuffle=False)


# %% ===== (3) Inception block (parallel kernels) =====
class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 4 == 0
        b = out_ch // 4
        self.b1 = nn.Conv1d(in_ch, b, 3, padding=1)
        self.b2 = nn.Conv1d(in_ch, b, 5, padding=2)
        self.b3 = nn.Conv1d(in_ch, b, 7, padding=3)
        self.b4 = nn.Conv1d(in_ch, b, 9, padding=4)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.relu(self.bn(out))


# %% ===== (2)+(3) three-modality model =====
class ThreeModalFusion(nn.Module):
    def __init__(self, n_classes=5, p_drop=DROPOUT):
        super().__init__()
        self.pool = nn.MaxPool1d(2)

        # --- time branch: inception -> BiLSTM ---
        self.t_inc1 = InceptionBlock1D(12, 64)
        self.t_inc2 = InceptionBlock1D(64, 128)
        self.t_lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)  # -> 128

        # --- frequency branch: inception -> global pool (no LSTM) ---
        self.f_inc1 = InceptionBlock1D(12, 64)
        self.f_inc2 = InceptionBlock1D(64, 128)

        # --- tabular branch ---
        self.fc_tab = nn.Linear(4, 16)

        # --- fusion head: 128 (time) + 128 (freq) + 16 (tab) = 272 ---
        self.drop = nn.Dropout(p_drop)
        self.fc_fusion = nn.Linear(128 + 128 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        # time branch
        t = self.pool(self.t_inc1(x_sig))      # (B, 64, 500)
        t = self.pool(self.t_inc2(t))          # (B, 128, 250)
        t = t.permute(0, 2, 1)                 # (B, 250, 128)
        _, (h_n, _) = self.t_lstm(t)
        emb_time = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, 128)

        # frequency branch: FFT magnitude of the raw signal, log-compressed
        mag = torch.fft.rfft(x_sig, dim=2).abs()          # (B, 12, 501)
        mag = torch.log1p(mag)                            # compress dynamic range
        f = self.pool(self.f_inc1(mag))        # (B, 64, 250)
        f = self.pool(self.f_inc2(f))          # (B, 128, 125)
        emb_freq = F.adaptive_avg_pool1d(f, 1).flatten(1)  # (B, 128)

        # tabular branch
        emb_tab = F.relu(self.fc_tab(x_tab))   # (B, 16)

        fused = torch.cat([emb_time, emb_freq, emb_tab], dim=1)   # (B, 272)
        return self.fc_fusion(self.drop(fused))                   # (B, 5) logits


# %% ===== shape check =====
model = ThreeModalFusion().to(device)
out = model(torch.randn(4, 12, 1000).to(device), torch.randn(4, 4).to(device))
print('output:', out.shape, '(expect [4, 5])')
print('params:', sum(p.numel() for p in model.parameters()))


# %% ===== train (multi-label: BCEWithLogitsLoss) =====
def train(model, epochs):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # multi-label loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    tr_losses, va_losses = [], []
    best_val, best_state, best_epoch = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x_sig, x_tab, yb in train_loader:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
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
        scheduler.step()
        if va < best_val:
            best_val, best_epoch = va, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'epoch {epoch+1:3d}/{epochs}   train {tr:.4f}   val {va:.4f}')

    model.load_state_dict(best_state)
    print(f'best val {best_val:.4f} at epoch {best_epoch}')
    return tr_losses, va_losses, best_epoch


tr_losses, va_losses, best_epoch = train(model, EPOCHS)
torch.save(model.state_dict(), ROOT + 'week8_fft_multilabel_fusion.pt')
np.savez(ROOT + 'week8_curves.npz', tr=tr_losses, va=va_losses, best_epoch=best_epoch)


# %% ===== evaluate (sigmoid, macro AUROC - benchmark-comparable) =====
model.eval()
probs = []
with torch.no_grad():
    for x_sig, x_tab, yb in test_loader:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        probs.append(torch.sigmoid(model(x_sig, x_tab)).cpu().numpy())   # sigmoid, not softmax
probs = np.concatenate(probs)

macro = roc_auc_score(y_te, probs, average='macro')
per = roc_auc_score(y_te, probs, average=None)
print(f'\nThree-modal multi-label fusion - macro AUROC {macro:.4f}')
for c, a in zip(CLASSES, per):
    print(f'  {c:5s} {a:.4f}')
print('(multi-label -> directly comparable to Strodthoff benchmark ~0.93)')


# %% ===== curves =====
plt.figure(figsize=(9, 5))
ep = range(1, len(tr_losses) + 1)
plt.plot(ep, tr_losses, linewidth=1.5, label='train')
plt.plot(ep, va_losses, linewidth=1.5, label='validation')
plt.axvline(best_epoch, color='red', linestyle=':', alpha=0.7, label=f'best (epoch {best_epoch})')
plt.xlabel('epoch'); plt.ylabel('BCE loss')
plt.title('Three-modal multi-label fusion (signal + FFT + demographics)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(ROOT + 'week8_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== per-class ROC =====
plt.figure(figsize=(7, 7))
for i, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_te[:, i], probs[:, i])
    plt.plot(fpr, tpr, label=f'{c} (AUC = {auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Three-modal multi-label fusion - one-vs-rest ROC')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT + 'week8_roc.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== convergence comparison vs week7 CNN+BiLSTM (curve shape only) =====
# AUROC is not directly comparable (single vs multi label), but convergence behaviour is.
plt.figure(figsize=(9, 5))
plt.plot(ep, va_losses, label='week8 3-modal multi-label (val)')
try:
    w7 = np.load(ROOT + 'week7_cnn_lstm_curves.npz')
    plt.plot(range(1, len(w7['va']) + 1), w7['va'], label='week7 CNN+BiLSTM single-label (val)')
except FileNotFoundError:
    print('week7_cnn_lstm_curves.npz not on Drive - skipping overlay')
plt.xlabel('epoch'); plt.ylabel('validation loss')
plt.title('Convergence behaviour: week7 vs week8')
plt.legend(); plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(ROOT + 'week8_convergence_comparison.png', dpi=150, bbox_inches='tight')
plt.show()