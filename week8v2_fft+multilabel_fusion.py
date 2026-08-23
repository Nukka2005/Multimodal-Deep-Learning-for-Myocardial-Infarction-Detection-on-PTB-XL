# ============================================================
# Week 8 v2 - pushing convergence past 150 epochs
#
# Changes from v1 (all marked <<< NEW):
#   1. MixUp: blend pairs of samples + their labels -> impossible to memorize
#   2. Smaller model: inception 32/64 (was 64/128), LSTM hidden 32 (was 64)
#   3. DropPath: randomly skip entire inception blocks during training
#   4. Gradient clipping: cap gradient norm to prevent val spikes
#   5. Stronger augmentation: doubled all knobs
#   6. Lower LR (5e-4) + longer warmup (30 epochs)
#
# Same: multi-label, three modalities (signal + FFT + tabular), BCE loss,
#        300 epochs, best-val checkpoint, cosine schedule.
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

COLAB = False
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
MAX_LR = 5e-4             # <<< NEW: was 1e-3 (slower, more careful)
WARMUP_EPOCHS = 30         # <<< NEW: was 10 (longer ramp-up)
GRAD_CLIP = 1.0            # <<< NEW: max gradient norm
MIXUP_ALPHA = 0.4          # <<< NEW: Beta distribution parameter for MixUp
DROP_PATH_RATE = 0.1       # <<< NEW: probability of skipping an inception block

# stronger augmentation (all doubled from v1)            <<< NEW
AUG_SHIFT  = 100           # was 50
AUG_NOISE  = 0.05          # was 0.02
AUG_SCALE  = 0.20          # was 0.10
AUG_LEAD_P = 0.20          # was 0.10


# %% ===== multi-label dataset (same as v1) =====
meta = pd.read_csv(DATA_DIR + 'ptbxl_database.csv', index_col='ecg_id')
meta['scp_codes'] = meta['scp_codes'].apply(ast.literal_eval)

scp = pd.read_csv(DATA_DIR + 'scp_statements.csv', index_col=0)
scp = scp[scp['diagnostic'] == 1]

def to_superclasses(codes):
    return list({scp.loc[c, 'diagnostic_class'] for c in codes if c in scp.index})

meta['superclasses'] = meta['scp_codes'].apply(to_superclasses)
multi = meta[meta['superclasses'].apply(len) >= 1].copy()

def multihot(names):
    v = np.zeros(5, dtype='float32')
    for n in names:
        v[CLASS_IDX[n]] = 1.0
    return v

Y = np.stack(multi['superclasses'].apply(multihot).values)
print('total kept (multi-label):', len(multi))
print('label counts:', Y.sum(0).astype(int), CLASSES)
print('multi-label records:', int((Y.sum(1) > 1).sum()))

tr_df = multi[multi['strat_fold'] <= 8]
va_df = multi[multi['strat_fold'] == 9]
te_df = multi[multi['strat_fold'] == 10]
ycols = [f'y_{c}' for c in CLASSES]
multi = multi.assign(**{f'y_{c}': Y[:, i] for i, c in enumerate(CLASSES)})
tr_df = multi[multi['strat_fold'] <= 8]
va_df = multi[multi['strat_fold'] == 9]
te_df = multi[multi['strat_fold'] == 10]
y_tr = tr_df[ycols].values.astype('float32')
y_va = va_df[ycols].values.astype('float32')
y_te = te_df[ycols].values.astype('float32')
print('split:', len(tr_df), len(va_df), len(te_df))


# %% ===== load signals =====
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
    print('caching multi-label signals...')
    X_tr = load_signals(tr_df); np.save(CACHE_DIR + 'ml_X_tr.npy', X_tr)
    X_va = load_signals(va_df); np.save(CACHE_DIR + 'ml_X_va.npy', X_va)
    X_te = load_signals(te_df); np.save(CACHE_DIR + 'ml_X_te.npy', X_te)
print('signals:', X_tr.shape)


# %% ===== tabular =====
features = ['age', 'sex', 'height', 'weight']
imputer = SimpleImputer(strategy='median')
T_tr = scaler_fit = imputer.fit_transform(tr_df[features])
T_va = imputer.transform(va_df[features])
T_te = imputer.transform(te_df[features])
scaler = StandardScaler()
T_tr = scaler.fit_transform(T_tr).astype('float32')
T_va = scaler.transform(T_va).astype('float32')
T_te = scaler.transform(T_te).astype('float32')

pos = y_tr.sum(0)
neg = len(y_tr) - pos
pos_weight = torch.tensor(neg / pos, dtype=torch.float32).to(device)
print('pos_weight:', pos_weight.cpu().numpy().round(2))


# %% ===== augmentation + MixUp =====
def augment_batch(x):
    """Stronger ECG augmentation (train only)."""
    B, C, L = x.shape
    shifts = torch.randint(-AUG_SHIFT, AUG_SHIFT + 1, (B,), device=x.device)
    idx = (torch.arange(L, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)) % L
    x = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, C, -1))
    scale = 1.0 + (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * AUG_SCALE
    x = x * scale
    x = x + torch.randn_like(x) * AUG_NOISE
    lead_mask = (torch.rand(B, C, 1, device=x.device) > AUG_LEAD_P).float()
    return x * lead_mask


def mixup_batch(x_sig, x_tab, y):                        # <<< NEW
    """MixUp: blend pairs of (signal, tabular, label) with a random ratio.
    Makes memorization nearly impossible because no clean sample ever repeats."""
    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)        # draw mixing ratio
    lam = max(lam, 1.0 - lam)                             # keep lam >= 0.5 so one sample dominates
    idx = torch.randperm(x_sig.size(0), device=x_sig.device)   # random shuffle for pairing
    x_sig = lam * x_sig + (1 - lam) * x_sig[idx]
    x_tab = lam * x_tab + (1 - lam) * x_tab[idx]
    y     = lam * y     + (1 - lam) * y[idx]              # soft labels: "70% MI, 30% NORM"
    return x_sig, x_tab, y


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, X, T, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.T[i], self.y[i]


train_loader = DataLoader(FusionDataset(X_tr, T_tr, y_tr), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(FusionDataset(X_va, T_va, y_va), batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(FusionDataset(X_te, T_te, y_te), batch_size=BATCH, shuffle=False)


# %% ===== DropPath (stochastic depth) =====              <<< NEW
class DropPath(nn.Module):
    """During training, randomly skip this block entirely (output zeros).
    At eval, always pass through. Forces every block to be independently useful."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.size(0), 1, 1, device=x.device) > self.p).float()
        return x * keep / (1 - self.p)   # scale up to preserve expected value


# %% ===== inception block (smaller + droppath) =====
class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, drop_path=0.0):
        super().__init__()
        assert out_ch % 4 == 0
        b = out_ch // 4
        self.b1 = nn.Conv1d(in_ch, b, 3, padding=1)
        self.b2 = nn.Conv1d(in_ch, b, 5, padding=2)
        self.b3 = nn.Conv1d(in_ch, b, 7, padding=3)
        self.b4 = nn.Conv1d(in_ch, b, 9, padding=4)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.drop_path(self.relu(self.bn(out)))


# %% ===== three-modality model (SMALLER) =====
class ThreeModalFusionV2(nn.Module):
    def __init__(self, n_classes=5, p_drop=DROPOUT, dp=DROP_PATH_RATE):
        super().__init__()
        self.pool = nn.MaxPool1d(2)

        # --- time branch: inception (SMALLER: 32/64, was 64/128) + BiLSTM ---
        self.t_inc1 = InceptionBlock1D(12, 32, drop_path=dp)    # <<< NEW: was 64
        self.t_inc2 = InceptionBlock1D(32, 64, drop_path=dp)    # <<< NEW: was 128
        self.t_lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=True)  # <<< NEW: was 64 -> 64

        # --- frequency branch: inception (SMALLER) + global pool ---
        self.f_inc1 = InceptionBlock1D(12, 32, drop_path=dp)    # <<< NEW: was 64
        self.f_inc2 = InceptionBlock1D(32, 64, drop_path=dp)    # <<< NEW: was 128

        # --- tabular ---
        self.fc_tab = nn.Linear(4, 16)

        # --- fusion: 64 (time, 32 bidir) + 64 (freq) + 16 (tab) = 144 ---
        self.drop = nn.Dropout(p_drop)
        self.fc_fusion = nn.Linear(64 + 64 + 16, n_classes)    # <<< NEW: was 128+128+16=272

    def forward(self, x_sig, x_tab):
        # time branch
        t = self.pool(self.t_inc1(x_sig))        # (B, 32, 500)
        t = self.pool(self.t_inc2(t))            # (B, 64, 250)
        t = t.permute(0, 2, 1)                   # (B, 250, 64)
        _, (h_n, _) = self.t_lstm(t)
        emb_time = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, 64)

        # frequency branch: FFT magnitude, log-compressed
        mag = torch.fft.rfft(x_sig, dim=2).abs()          # (B, 12, 501)
        mag = torch.log1p(mag)
        f = self.pool(self.f_inc1(mag))          # (B, 32, 250)
        f = self.pool(self.f_inc2(f))            # (B, 64, 125)
        emb_freq = F.adaptive_avg_pool1d(f, 1).flatten(1)  # (B, 64)

        # tabular branch
        emb_tab = F.relu(self.fc_tab(x_tab))     # (B, 16)

        fused = torch.cat([emb_time, emb_freq, emb_tab], dim=1)   # (B, 144)
        return self.fc_fusion(self.drop(fused))


# %% ===== shape + param check =====
model = ThreeModalFusionV2().to(device)
out = model(torch.randn(4, 12, 1000).to(device), torch.randn(4, 4).to(device))
n_params = sum(p.numel() for p in model.parameters())
print('output:', out.shape)
print(f'params: {n_params:,} (v1 was ~{272*5 + 128*128*2 + 64*128*4*2:,}+)')
print(f'samples per param: {len(X_tr)/n_params:.4f}')


# %% ===== train (MixUp + gradient clipping + all regularization) =====
def train(model, epochs):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # no label_smoothing: MixUp handles it
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    tr_losses, va_losses = [], []
    best_val, best_state, best_epoch = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x_sig, x_tab, yb in train_loader:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            x_sig = augment_batch(x_sig)                           # augmentation
            x_sig, x_tab, yb = mixup_batch(x_sig, x_tab, yb)     # <<< NEW: MixUp

            optimizer.zero_grad()
            loss = criterion(model(x_sig, x_tab), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)   # <<< NEW: clip
            optimizer.step()
            running += loss.item() * x_sig.size(0)
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
torch.save(model.state_dict(), ROOT + 'week8v2_fusion.pt')
np.savez(ROOT + 'week8v2_curves.npz', tr=tr_losses, va=va_losses, best_epoch=best_epoch)


# %% ===== evaluate =====
model.eval()
probs = []
with torch.no_grad():
    for x_sig, x_tab, yb in test_loader:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        probs.append(torch.sigmoid(model(x_sig, x_tab)).cpu().numpy())
probs = np.concatenate(probs)

macro = roc_auc_score(y_te, probs, average='macro')
per = roc_auc_score(y_te, probs, average=None)
print(f'\nv2 three-modal multi-label - macro AUROC {macro:.4f}')
for c, a in zip(CLASSES, per):
    print(f'  {c:5s} {a:.4f}')
print(f'(benchmark: Strodthoff ~0.93, v1 was 0.9228)')


# %% ===== loss curves =====
plt.figure(figsize=(9, 5))
ep = range(1, len(tr_losses) + 1)
plt.plot(ep, tr_losses, linewidth=1.5, label='train (augmented + mixup)')
plt.plot(ep, va_losses, linewidth=1.5, label='validation (clean)')
plt.axvline(best_epoch, color='red', linestyle=':', alpha=0.7, label=f'best (epoch {best_epoch})')
plt.axvline(150, color='gray', linestyle='--', alpha=0.5, label='epoch 150 target')
plt.xlabel('epoch'); plt.ylabel('BCE loss')
plt.title('v2: smaller model + MixUp + DropPath + gradient clip')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(ROOT + 'week8v2_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== per-class ROC =====
plt.figure(figsize=(7, 7))
for i, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_te[:, i], probs[:, i])
    plt.plot(fpr, tpr, label=f'{c} (AUC = {auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('v2 three-modal multi-label - one-vs-rest ROC')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT + 'week8v2_roc.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== v1 vs v2 comparison (if v1 curves exist) =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

axes[0].set_title('v1: inception 64/128, LSTM h=64, mild aug')
try:
    v1 = np.load(ROOT + 'week8_curves.npz')
    ep1 = range(1, len(v1['tr']) + 1)
    axes[0].plot(ep1, v1['tr'], label='train')
    axes[0].plot(ep1, v1['va'], label='validation')
    axes[0].axvline(int(v1['best_epoch']), color='red', linestyle=':', alpha=0.7,
                    label=f"best (epoch {int(v1['best_epoch'])})")
except FileNotFoundError:
    axes[0].text(0.5, 0.5, 'week8_curves.npz not found', ha='center', va='center',
                 transform=axes[0].transAxes)
axes[0].axvline(150, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('BCE loss')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].set_title('v2: inception 32/64, LSTM h=32, MixUp + DropPath')
axes[1].plot(ep, tr_losses, label='train')
axes[1].plot(ep, va_losses, label='validation')
axes[1].axvline(best_epoch, color='red', linestyle=':', alpha=0.7, label=f'best (epoch {best_epoch})')
axes[1].axvline(150, color='gray', linestyle='--', alpha=0.5, label='epoch 150 target')
axes[1].set_xlabel('epoch')
axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(ROOT + 'week8_v1_vs_v2.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== full progression table =====
print('\n' + '='*70)
print('FULL PROGRESSION (all weeks)')
print('='*70)
runs = [
    ('Transformer no reg',          5,    0.9138, 'single'),
    ('Transformer + reg',           6,    None,   'single'),
    ('Transformer + reg + aug',     14,   None,   'single'),
    ('CNN+BiLSTM h=128',            47,   None,   'single'),
    ('CNN+BiLSTM h=64',             47,   0.9111, 'single'),
    ('3-modal v1 (w8)',             38,   0.9228, 'multi'),
    ('3-modal v2 (w8)',             best_epoch, macro, 'multi'),
]
print(f"{'model':<28}{'best ep':>8}{'AUROC':>8}  {'labels'}")
for name, ep, auc_val, lab in runs:
    a = f'{auc_val:.4f}' if auc_val else '   -'
    print(f'{name:<28}{ep:>8}{a:>8}  {lab}')
print(f"{'Strodthoff benchmark':<28}{'—':>8}{'0.930':>8}  multi")