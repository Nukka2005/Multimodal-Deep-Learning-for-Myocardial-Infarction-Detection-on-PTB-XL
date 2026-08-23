# %%
# ============================================================
# Week 5 Task 4 - xresnet1d101: the deep residual backbone that tops the
# PTB-XL benchmark (Strodthoff et al. 2021, 0.928 macro AUROC).
#
# Two models are trained:
#   1. signal only  -> directly comparable to the published benchmark number
#   2. fusion       -> signal + demographics, comparable to week4/week5 fusion
#
# Needs: ptb-xl/ptbxl_5class.csv  (run week4.py once to create it)
#        cache/*.npy              (week4.py writes these on first run)
# ============================================================
# %%
import os
import numpy as np
import pandas as pd
import wfdb

os.makedirs('./cache', exist_ok=True)

df5 = pd.read_csv('./ptb-xl/ptbxl_5class.csv', index_col='ecg_id')
tr_df = df5[df5['strat_fold'] <= 8]
va_df = df5[df5['strat_fold'] == 9]
te_df = df5[df5['strat_fold'] == 10]

def load_split(split_df, name):
    X, y = [], []
    for i, (_, row) in enumerate(split_df.iterrows()):
        sig, _ = wfdb.rdsamp('./ptb-xl/' + row['filename_lr'])
        X.append(sig.T)
        y.append(row['label'])
        if (i + 1) % 2000 == 0:
            print(f'  {name}: {i+1}/{len(split_df)}')
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int64')
    np.save(f'./cache/X_{name}.npy', X)
    np.save(f'./cache/y_{name}.npy', y)
    print(f'{name}: {X.shape}  {np.bincount(y)}')

for df, name in [(tr_df, 'tr'), (va_df, 'va'), (te_df, 'te')]:
    load_split(df, name)

print('cache built')
print(os.path.exists('./ptb-xl/ptbxl_5class.csv'))
# %%
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
if device.type == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))
    print('vram:', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
USE_AMP = device.type == 'cuda'   # mixed precision: ~2x faster, half the memory

# %% ===== data =====
df5 = pd.read_csv('./ptb-xl/ptbxl_5class.csv', index_col='ecg_id')
tr_df = df5[df5['strat_fold'] <= 8]
va_df = df5[df5['strat_fold'] == 9]
te_df = df5[df5['strat_fold'] == 10]

X_tr, y_tr = np.load('./cache/X_tr.npy'), np.load('./cache/y_tr.npy')
X_va, y_va = np.load('./cache/X_va.npy'), np.load('./cache/y_va.npy')
X_te, y_te = np.load('./cache/X_te.npy'), np.load('./cache/y_te.npy')
print('signals:', X_tr.shape, X_va.shape, X_te.shape)

# tabular: median impute + scale, fit on TRAIN only
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
print('class weights:', weights5.cpu().numpy().round(3))


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, X_sig, X_tab, y):
        self.X_sig = torch.tensor(X_sig, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_sig[i], self.X_tab[i], self.y[i]


BATCH = 32
sig_train = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=BATCH, shuffle=True)
sig_val   = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)), batch_size=BATCH, shuffle=False)
sig_test  = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=BATCH, shuffle=False)

fuse_train = DataLoader(MultimodalDataset(X_tr, T_tr, y_tr), batch_size=BATCH, shuffle=True)
fuse_val   = DataLoader(MultimodalDataset(X_va, T_va, y_va), batch_size=BATCH, shuffle=False)
fuse_test  = DataLoader(MultimodalDataset(X_te, T_te, y_te), batch_size=BATCH, shuffle=False)


# %% ===== building blocks =====
class ConvLayer(nn.Sequential):
    """conv -> batchnorm -> relu. zero_bn starts the block as an identity."""
    def __init__(self, ni, nf, ks=3, stride=1, act=True, zero_bn=False):
        conv = nn.Conv1d(ni, nf, ks, stride=stride, padding=ks // 2, bias=False)
        bn = nn.BatchNorm1d(nf)
        # zero-gamma trick: last BN in a block starts at 0 -> block outputs 0 -> pure identity
        nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
        layers = [conv, bn]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class Bottleneck(nn.Module):
    """squeeze -> work -> expand. 4x fewer params than a plain block of the same width,
    which is what makes 101 layers affordable."""
    expansion = 4

    def __init__(self, ni, nh, stride=1, ks=5):
        super().__init__()
        nf = nh * self.expansion
        self.convs = nn.Sequential(
            ConvLayer(ni, nh, ks=1),                              # squeeze channels
            ConvLayer(nh, nh, ks=ks, stride=stride),              # ResNet-B: stride here, not on the 1x1
            ConvLayer(nh, nf, ks=1, act=False, zero_bn=True),     # expand back, zero-gamma
        )
        # ResNet-D shortcut: avgpool then 1x1 conv, instead of a striding 1x1 that skips samples
        if ni != nf or stride != 1:
            pool = [nn.AvgPool1d(2, ceil_mode=True)] if stride != 1 else []
            self.shortcut = nn.Sequential(*pool, ConvLayer(ni, nf, ks=1, act=False))
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convs(x) + self.shortcut(x))


class XResNet1dBackbone(nn.Module):
    """Signal encoder. layers=[3,4,23,3] gives the 101-layer variant."""
    def __init__(self, layers, in_ch=12, ks=5):
        super().__init__()
        # ResNet-C stem: three narrow convs instead of one wide one
        self.stem = nn.Sequential(
            ConvLayer(in_ch, 32, ks=ks, stride=2),
            ConvLayer(32, 32, ks=ks),
            ConvLayer(32, 64, ks=ks),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        ni = 64
        stages = []
        for i, (nh, n_blocks) in enumerate(zip([64, 128, 256, 512], layers)):
            stride = 1 if i == 0 else 2
            blocks = [Bottleneck(ni if j == 0 else nh * 4, nh,
                                 stride if j == 0 else 1, ks=ks)
                      for j in range(n_blocks)]
            stages.append(nn.Sequential(*blocks))
            ni = nh * 4
        self.stages = nn.Sequential(*stages)
        self.out_dim = ni          # 2048

    def forward(self, x):
        x = self.stem(x)           # (B, 64, 250)
        x = self.stages(x)         # (B, 2048, 32)
        return F.adaptive_avg_pool1d(x, 1).flatten(1)   # (B, 2048)


class XResNet1d(nn.Module):
    """Signal only classifier."""
    def __init__(self, layers, in_ch=12, n_classes=5, p_drop=0.5):
        super().__init__()
        self.backbone = XResNet1dBackbone(layers, in_ch)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(self.backbone.out_dim, n_classes)

    def forward(self, x):
        return self.fc(self.drop(self.backbone(x)))


class XResNet1dFusion(nn.Module):
    """Signal + demographics. The 2048-dim signal embedding is projected down to 256
    so the 16-dim tabular embedding is not drowned out at the concat."""
    def __init__(self, layers, in_ch=12, n_tab=4, n_classes=5, p_drop=0.5, emb_dim=256):
        super().__init__()
        self.backbone = XResNet1dBackbone(layers, in_ch)
        self.sig_proj = nn.Sequential(
            nn.Linear(self.backbone.out_dim, emb_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_tab = nn.Linear(n_tab, 16)
        self.drop = nn.Dropout(p_drop)
        self.fc_fusion = nn.Linear(emb_dim + 16, n_classes)

    def forward(self, x_sig, x_tab):
        emb_sig = self.sig_proj(self.backbone(x_sig))       # (B, 256)
        emb_tab = F.relu(self.fc_tab(x_tab))                # (B, 16)
        fused = torch.cat([emb_sig, emb_tab], dim=1)        # (B, 272)
        return self.fc_fusion(self.drop(fused))


def xresnet1d101(**kwargs):
    return XResNet1d([3, 4, 23, 3], **kwargs)


def xresnet1d101_fusion(**kwargs):
    return XResNet1dFusion([3, 4, 23, 3], **kwargs)


def xresnet1d50(**kwargs):
    """Smaller fallback if 101 overfits badly."""
    return XResNet1d([3, 4, 6, 3], **kwargs)


# %% ===== shape + size check before committing to a long run =====
probe = xresnet1d101(n_classes=5).to(device)
with torch.no_grad():
    out = probe(torch.randn(2, 12, 1000).to(device))
n_params = sum(p.numel() for p in probe.parameters())
print('output:', out.shape)                          # expect [2, 5]
print(f'parameters: {n_params/1e6:.1f}M')
print(f'samples per parameter: {len(X_tr)/n_params:.5f}')   # this is the overfitting risk in one number
del probe
if device.type == 'cuda':
    torch.cuda.empty_cache()


# %% ===== training =====
def train(model, train_loader, val_loader, epochs=30, lr=1e-3, wd=1e-2, multimodal=False, tag=''):
    """One-cycle LR + AdamW + mixed precision. Restores the best-validation checkpoint."""
    criterion = nn.CrossEntropyLoss(weight=weights5, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    tr_losses, va_losses, lrs = [], [], []
    best_val, best_state, best_epoch = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            *inputs, yb = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = criterion(model(*inputs), yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()                       # one-cycle steps per batch, not per epoch
            running += loss.item() * yb.size(0)
        tr = running / len(train_loader.dataset)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                *inputs, yb = [b.to(device, non_blocking=True) for b in batch]
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    running += criterion(model(*inputs), yb).item() * yb.size(0)
        va = running / len(val_loader.dataset)

        tr_losses.append(tr); va_losses.append(va)
        lrs.append(scheduler.get_last_lr()[0])
        if va < best_val:
            best_val, best_epoch = va, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(f'{tag} epoch {epoch+1:2d}/{epochs}   train {tr:.4f}   val {va:.4f}   lr {lrs[-1]:.2e}')

    model.load_state_dict(best_state)
    print(f'{tag} best val {best_val:.4f} at epoch {best_epoch}')
    return tr_losses, va_losses, lrs, best_epoch


def predict(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            *inputs, _ = [b.to(device) for b in batch]
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                p = F.softmax(model(*inputs).float(), dim=1)
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


# %% ===== train signal only =====
model_sig = xresnet1d101(n_classes=5, p_drop=0.5).to(device)
tr_s, va_s, lr_s, best_ep_s = train(model_sig, sig_train, sig_val, epochs=30, tag='[sig]')
torch.save(model_sig.state_dict(), 'week5_xresnet1d101_signal.pt')

probs_sig = predict(model_sig, sig_test)
macro_sig = roc_auc_score(y_te, probs_sig, multi_class='ovr', average='macro')
per_sig = roc_auc_score(y_te, probs_sig, multi_class='ovr', average=None)
print(f'\nxresnet1d101 signal only - macro AUROC {macro_sig:.4f}')
for c, a in zip(CLASSES, per_sig):
    print(f'  {c:5s} {a:.4f}')


# %% ===== train fusion =====
if device.type == 'cuda':
    torch.cuda.empty_cache()

model_fuse = xresnet1d101_fusion(n_classes=5, n_tab=4, p_drop=0.5).to(device)
tr_f, va_f, lr_f, best_ep_f = train(model_fuse, fuse_train, fuse_val, epochs=30, multimodal=True, tag='[fuse]')
torch.save(model_fuse.state_dict(), 'week5_xresnet1d101_fusion.pt')

probs_fuse = predict(model_fuse, fuse_test)
macro_fuse = roc_auc_score(y_te, probs_fuse, multi_class='ovr', average='macro')
per_fuse = roc_auc_score(y_te, probs_fuse, multi_class='ovr', average=None)
print(f'\nxresnet1d101 fusion - macro AUROC {macro_fuse:.4f}')
for c, a in zip(CLASSES, per_fuse):
    print(f'  {c:5s} {a:.4f}')

np.savez('week5_xresnet_probs.npz',
         probs_sig=probs_sig, probs_fuse=probs_fuse, y_true=y_te)


# %% ===== loss curves =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (tr, va, best_ep, title) in zip(axes, [
        (tr_s, va_s, best_ep_s, 'xresnet1d101 signal only'),
        (tr_f, va_f, best_ep_f, 'xresnet1d101 fusion')]):
    ep = range(1, len(tr) + 1)
    ax.plot(ep, tr, marker='o', linewidth=2, label='train')
    ax.plot(ep, va, marker='s', linewidth=2, label='validation')
    ax.axvline(best_ep, color='red', linestyle=':', alpha=0.7,
               label=f'best checkpoint (epoch {best_ep})')
    ax.set_xlabel('epoch'); ax.set_ylabel('cross-entropy loss')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

plt.tight_layout()
plt.savefig('week5_xresnet_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== one-cycle LR schedule (shows the warmup then anneal) =====
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(lr_s) + 1), lr_s, linewidth=2)
plt.xlabel('epoch'); plt.ylabel('learning rate')
plt.title('One-cycle LR schedule')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('week5_xresnet_lr_schedule.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== ROC curves, one panel per model =====
y_bin = label_binarize(y_te, classes=list(range(5)))

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (probs, macro, title) in zip(axes, [
        (probs_sig, macro_sig, 'xresnet1d101 signal only'),
        (probs_fuse, macro_fuse, 'xresnet1d101 fusion')]):
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, linewidth=2, label=f'{cls} (AUC = {auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title}\nmacro AUROC = {macro:.4f}')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('week5_xresnet_roc.png', dpi=150, bbox_inches='tight')
plt.show()


# %% ===== per-class comparison against the earlier models =====
# loads whatever previous runs saved; skips silently if a file is missing
runs = {'xresnet101 signal': per_sig, 'xresnet101 fusion': per_fuse}

for fname, label in [('week5task4_inception_test_probs.npz', 'inception fusion'),
                     ('week5task1_transformer_test_probs.npz', 'transformer fusion')]:
    if os.path.exists(fname):
        d = np.load(fname)
        runs[label] = roc_auc_score(d['y_true'], d['probs'], multi_class='ovr', average=None)
    else:
        print(f'skipping {label}: {fname} not found')

x = np.arange(len(CLASSES))
width = 0.8 / len(runs)
plt.figure(figsize=(11, 5))
for i, (name, per) in enumerate(runs.items()):
    plt.bar(x + (i - len(runs) / 2 + 0.5) * width, per, width, label=name)
plt.xticks(x, CLASSES)
plt.ylabel('AUROC'); plt.ylim(0.5, 1.0)
plt.title('Per-class AUROC by architecture')
plt.legend(); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('week5_architecture_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'model':<22}{'macro':>8}   " + '  '.join(f'{c:>6}' for c in CLASSES))
for name, per in runs.items():
    print(f'{name:<22}{per.mean():>8.4f}   ' + '  '.join(f'{a:>6.3f}' for a in per))
print(f"\nStrodthoff et al. 2021 benchmark: xresnet1d101 = 0.928 (multi-label variant)")