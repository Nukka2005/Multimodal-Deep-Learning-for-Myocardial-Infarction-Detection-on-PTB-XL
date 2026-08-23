# %% [cell 1] setup
# week9 xai, pass 2: the week8 v2 three-modal fusion model.
# multi-label (bce), inputs are (signal, tabular). the FFT branch is derived from
# the signal INSIDE forward, so it is not a separate input. two analyses here:
#   A. signal saliency maps per class (time-domain, folds in freq processing)
#   B. modality contribution: signal vs demographics attribution mass (the
#      research-question analysis)
import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients

try:
    import google.colab  # noqa: F401
    COLAB = True
except ImportError:
    COLAB = False

if COLAB:
    ROOT = "/content/drive/MyDrive/PTBXL/"
else:
    _default = os.path.expanduser("~/Ebad/PTB_XL") + "/"
    if os.path.exists(_default + "week8v2_fusion.pt"):
        ROOT = _default
    else:
        _hits = glob.glob(os.path.expanduser("~") + "/**/week8v2_fusion.pt", recursive=True)
        if not _hits:
            raise FileNotFoundError("week8v2_fusion.pt not found under home. set ROOT manually.")
        ROOT = os.path.dirname(_hits[0]) + "/"
print("ROOT =", ROOT)

DATA_DIR = ROOT + "ptb-xl/"
CACHE_DIR = ROOT + "cache/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.enabled = False   # bilstm backward in eval needs native, not cudnn

SEQ_LEN = 1000
N_CLASSES = 5
N_STEPS = 50
THRESHOLD = 0.5        # sigmoid threshold for "predicted positive". tune if needed.
DROPOUT = 0.5
DROP_PATH_RATE = 0.1

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

OUT_DIR = ROOT + "xai_outputs_fusion"
os.makedirs(OUT_DIR, exist_ok=True)


# %% [cell 2] model embedded verbatim from week8v2, then load frozen weights
# constants inlined so no side effects. droppath/dropout are inert at eval.
class DropPath(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.size(0), 1, 1, device=x.device) > self.p).float()
        return x * keep / (1 - self.p)


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


class ThreeModalFusionV2(nn.Module):
    def __init__(self, n_classes=5, p_drop=DROPOUT, dp=DROP_PATH_RATE):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.t_inc1 = InceptionBlock1D(12, 32, drop_path=dp)
        self.t_inc2 = InceptionBlock1D(32, 64, drop_path=dp)
        self.t_lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
        self.f_inc1 = InceptionBlock1D(12, 32, drop_path=dp)
        self.f_inc2 = InceptionBlock1D(32, 64, drop_path=dp)
        self.fc_tab = nn.Linear(4, 16)
        self.drop = nn.Dropout(p_drop)
        self.fc_fusion = nn.Linear(64 + 64 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        t = self.pool(self.t_inc1(x_sig))
        t = self.pool(self.t_inc2(t))
        t = t.permute(0, 2, 1)
        _, (h_n, _) = self.t_lstm(t)
        emb_time = torch.cat([h_n[-2], h_n[-1]], dim=1)
        mag = torch.fft.rfft(x_sig, dim=2).abs()
        mag = torch.log1p(mag)
        f = self.pool(self.f_inc1(mag))
        f = self.pool(self.f_inc2(f))
        emb_freq = F.adaptive_avg_pool1d(f, 1).flatten(1)
        emb_tab = F.relu(self.fc_tab(x_tab))
        fused = torch.cat([emb_time, emb_freq, emb_tab], dim=1)
        return self.fc_fusion(self.drop(fused))


model = ThreeModalFusionV2()
ckpt_path = ROOT + "week8v2_fusion.pt"
state = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state["model_state"] if "model_state" in state else state)
model.to(DEVICE).eval()
print("model loaded, eval mode")


# %% [cell 3] multi-label test data, reconstructed from the week8v2 pipeline
meta = pd.read_csv(DATA_DIR + "ptbxl_database.csv", index_col="ecg_id")
meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)
scp = pd.read_csv(DATA_DIR + "scp_statements.csv", index_col=0)
scp = scp[scp["diagnostic"] == 1]


def to_superclasses(codes):
    return list({scp.loc[c, "diagnostic_class"] for c in codes if c in scp.index})


def multihot(names):
    v = np.zeros(5, dtype="float32")
    for n in names:
        v[CLASS_IDX[n]] = 1.0
    return v


meta["superclasses"] = meta["scp_codes"].apply(to_superclasses)
multi = meta[meta["superclasses"].apply(len) >= 1].copy()
Y = np.stack(multi["superclasses"].apply(multihot).values)
multi = multi.assign(**{f"y_{c}": Y[:, i] for i, c in enumerate(CLASSES)})

tr_df = multi[multi["strat_fold"] <= 8]
te_df = multi[multi["strat_fold"] == 10]
ycols = [f"y_{c}" for c in CLASSES]
y_te = te_df[ycols].values.astype("float32")

X_te = np.load(CACHE_DIR + "ml_X_te.npy")     # (N, 12, 1000) raw signal

features = ["age", "sex", "height", "weight"]
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
scaler.fit(imputer.fit_transform(tr_df[features]))
T_te = scaler.transform(imputer.transform(te_df[features])).astype("float32")

X_te_t = torch.tensor(X_te, dtype=torch.float32)
T_te_t = torch.tensor(T_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.float32)
print("test signals:", tuple(X_te_t.shape), "demo:", tuple(T_te_t.shape),
      "labels:", tuple(y_te_t.shape))
print("positives per class:", dict(zip(CLASSES, y_te_t.sum(0).int().tolist())))


# %% [cell 4] analysis A - signal saliency maps per class (multi-label)
# demographics held fixed. one recording feeds every class it is correctly
# positive for. nan guard: rfft.abs() near the zero baseline can be ill-defined;
# gausslegendre (captum default) avoids the exact endpoints, but we skip any nan
# attribution defensively.
ig = IntegratedGradients(model)

sal_sum = torch.zeros(N_CLASSES, 12, SEQ_LEN)
sal_cnt = torch.zeros(N_CLASSES, dtype=torch.long)
nan_skips = 0

model.eval()
for i in range(len(y_te_t)):
    sig = X_te_t[i:i + 1].to(DEVICE)
    tab = T_te_t[i:i + 1].to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(sig, tab))[0]      # (5,)
    pred = (prob >= THRESHOLD).float()
    for cls in range(N_CLASSES):
        if y_te_t[i, cls] == 1 and pred[cls] == 1:    # correctly positive for cls
            attr = ig.attribute(sig, baselines=torch.zeros_like(sig),
                                 target=cls, additional_forward_args=(tab,),
                                 n_steps=N_STEPS)
            a = attr.squeeze(0).abs().detach().cpu()
            if torch.isnan(a).any():
                nan_skips += 1
                continue
            sal_sum[cls] += a
            sal_cnt[cls] += 1

print("correctly classified per class:", dict(zip(CLASSES, sal_cnt.tolist())))
print("nan attributions skipped:", nan_skips)
avg_saliency = sal_sum / sal_cnt.clamp(min=1).view(-1, 1, 1)
np.save(os.path.join(OUT_DIR, "avg_saliency.npy"), avg_saliency.numpy())
np.save(os.path.join(OUT_DIR, "sal_counts.npy"), sal_cnt.numpy())


# %% [cell 5] heatmaps + top 5% bars (same as pass 1)
avg = np.load(os.path.join(OUT_DIR, "avg_saliency.npy"))
cnts = np.load(os.path.join(OUT_DIR, "sal_counts.npy"))
fig, axes = plt.subplots(N_CLASSES, 1, figsize=(16, 20))
for i, cname in enumerate(CLASSES):
    s = avg[i]
    im = axes[i].imshow(s, aspect='auto', cmap='hot')
    axes[i].set_yticks(range(12)); axes[i].set_yticklabels(LEAD_NAMES)
    axes[i].set_title(f'saliency map - {cname} (n={int(cnts[i])})')
    axes[i].set_xlabel('time step')
    fig.colorbar(im, ax=axes[i], fraction=0.02)
    ti = s.sum(axis=0)
    thr = np.percentile(ti, 95)
    for t in np.where(ti >= thr)[0]:
        axes[i].axvline(x=t, color='cyan', alpha=0.35, linewidth=0.6)
plt.tight_layout()
fp = os.path.join(OUT_DIR, "saliency_maps_all_classes.png")
plt.savefig(fp, dpi=150); plt.show()
print("saved", fp)


# %% [cell 6] confound-corrected lead + interior time (same as pass 1 cell 8)
avg = np.load(os.path.join(OUT_DIR, "avg_saliency.npy"))
MARGIN = 50
print("interior peak time step (edges excluded):")
for i, cname in enumerate(CLASSES):
    ti = avg[i].sum(axis=0).copy()
    ti[:MARGIN] = 0; ti[-MARGIN:] = 0
    print(f"  {cname:5s} peak step {int(ti.argmax())} (~{int(ti.argmax())/SEQ_LEN*10:.2f}s)")

lead_mean = avg.mean(axis=2)
relative = lead_mean - lead_mean.mean(axis=0)
print("\nclass-specific top leads (elevation above global baseline):")
for i, cname in enumerate(CLASSES):
    order = np.argsort(relative[i])[::-1]
    tops = [(LEAD_NAMES[j], round(float(relative[i][j]), 5)) for j in order[:3]]
    print(f"  {cname:5s} {tops}")


# %% [cell 7] analysis B - modality contribution: signal vs demographics
# attribute BOTH inputs as a tuple (both interpolated from a zero baseline), so
# completeness holds across modalities and the split is meaningful. report each
# modality's share of total attribution magnitude, per class.
# caveat: signal has 12000 cells vs demographics 4, so raw mass favors signal by
# dimension. the informative read is whether demographics' share is ~0 (ignored)
# or clearly non-zero despite its tiny dimension (used). we also report per-
# element mean to control for that asymmetry.
ig2 = IntegratedGradients(model)
sig_mass = np.zeros(N_CLASSES)
tab_mass = np.zeros(N_CLASSES)
cnt = np.zeros(N_CLASSES)
CAP = 200   # cap correctly-classified recordings per class to keep this quick

model.eval()
for i in range(len(y_te_t)):
    if (cnt >= CAP).all():
        break
    sig = X_te_t[i:i + 1].to(DEVICE)
    tab = T_te_t[i:i + 1].to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(sig, tab))[0]
    pred = (prob >= THRESHOLD).float()
    for cls in range(N_CLASSES):
        if y_te_t[i, cls] == 1 and pred[cls] == 1 and cnt[cls] < CAP:
            a_sig, a_tab = ig2.attribute(
                (sig, tab),
                baselines=(torch.zeros_like(sig), torch.zeros_like(tab)),
                target=cls, n_steps=N_STEPS)
            if torch.isnan(a_sig).any() or torch.isnan(a_tab).any():
                continue
            sig_mass[cls] += float(a_sig.abs().sum())
            tab_mass[cls] += float(a_tab.abs().sum())
            cnt[cls] += 1

sig_mean = sig_mass / np.maximum(cnt, 1)
tab_mean = tab_mass / np.maximum(cnt, 1)
print("\nmodality contribution per class (n capped at", CAP, "):")
print(f"{'class':6s}{'demo share %':>14}{'demo/elem':>14}{'sig/elem':>14}")
for i, cname in enumerate(CLASSES):
    share = 100 * tab_mean[i] / (sig_mean[i] + tab_mean[i] + 1e-12)
    demo_per_elem = tab_mean[i] / 4
    sig_per_elem = sig_mean[i] / (12 * SEQ_LEN)
    print(f"{cname:6s}{share:>14.2f}{demo_per_elem:>14.6f}{sig_per_elem:>14.6f}")