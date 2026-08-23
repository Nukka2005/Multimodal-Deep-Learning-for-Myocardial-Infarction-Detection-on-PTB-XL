# %% [cell 1] setup
# week9 explainable ai - integrated gradients saliency maps
# frozen week7 cnn+bilstm fusion (signal + 4 demographics, single-label, 5-class).
# self-contained: the model class is embedded below so importing NEVER re-runs
# the week7 training script. no training this week. we attribute the signal branch;
# demographics is held fixed (pass 2).

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients

try:
    import google.colab  # noqa: F401
    COLAB = True
except ImportError:
    COLAB = False

import glob

if COLAB:
    ROOT = "/content/drive/MyDrive/PTBXL/"
else:
    # auto-resolve: use the default if the checkpoint is there, else search home.
    # stops re-running cell 1 from reverting a hand-fixed path.
    _default = os.path.expanduser("~/Ebad/MIT_BIH") + "/"
    if os.path.exists(_default + "week7_cnn_lstm_fusion.pt"):
        ROOT = _default
    else:
        _hits = glob.glob(os.path.expanduser("~") + "/**/week7_cnn_lstm_fusion.pt",
                          recursive=True)
        if not _hits:
            raise FileNotFoundError(
                "week7_cnn_lstm_fusion.pt not found under home. set ROOT manually.")
        ROOT = os.path.dirname(_hits[0]) + "/"
print("ROOT =", ROOT)

DATA_DIR = ROOT + "ptb-xl/"
CACHE_DIR = ROOT + "cache/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# captum needs a backward pass through the bilstm, but cudnn's fused rnn backward
# refuses to run in eval mode. disable cudnn so the native lstm (same math) is
# used. keeps model.eval() intact, so dropout stays off and bn uses running stats.
torch.backends.cudnn.enabled = False

SEQ_LEN = 1000          # 100 hz ptb-xl, confirmed from week7 source
N_CLASSES = 5
N_STEPS = 50

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']   # matches week7 CLASSES order

OUT_DIR = ROOT + "xai_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# baseline note: signal is NOT normalized in the week7 pipeline (only tabular is).
# so zero baseline = flat line = absence of signal. state this in task 5.
print(f"colab={COLAB} device={DEVICE} seq_len={SEQ_LEN}")


# %% [cell 2] model class embedded verbatim from week7, then load frozen weights
# embedded on purpose so this file has no side effects. eval() disables dropout,
# so p_drop value is irrelevant at inference.
class CNNLSTMFusion(nn.Module):
    def __init__(self, n_classes=5, hidden=128, p_drop=0.5):
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
        s = s.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(s)
        emb_sig = torch.cat([h_n[-2], h_n[-1]], dim=1)
        emb_tab = F.relu(self.fc_tab(x_tab))
        fused = torch.cat([emb_sig, emb_tab], dim=1)
        return self.fc_fusion(self.drop(fused))


model = CNNLSTMFusion(n_classes=N_CLASSES, hidden=64)   # hidden=64 matches the checkpoint
ckpt_path = ROOT + "week7_cnn_lstm_fusion.pt"
state = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state["model_state"] if "model_state" in state else state)
model.to(DEVICE).eval()
# do NOT wrap ig.attribute in torch.no_grad().
print("model loaded, eval mode")


# %% [cell 3] test data, reconstructed from the week7 pipeline
# refit imputer + scaler on train tabular exactly as week7 did, then transform
# test. signal loaded raw (no normalization), matching training.
df5 = pd.read_csv(DATA_DIR + "ptbxl_5class.csv", index_col="ecg_id")
tr_df = df5[df5["strat_fold"] <= 8]
te_df = df5[df5["strat_fold"] == 10]

X_te = np.load(CACHE_DIR + "X_te.npy")    # (N, 12, 1000) raw signal
y_te = np.load(CACHE_DIR + "y_te.npy")

features = ["age", "sex", "height", "weight"]
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
scaler.fit(imputer.fit_transform(tr_df[features]))                       # fit on train
T_te = scaler.transform(imputer.transform(te_df[features])).astype("float32")

X_te_t = torch.tensor(X_te, dtype=torch.float32)
T_te_t = torch.tensor(T_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.int64)
print("test signals:", tuple(X_te_t.shape), "demo:", tuple(T_te_t.shape))


# %% [cell 4] integrated gradients on the signal branch, aggregate per class
# running mean of ABSOLUTE attributions. demographics passed as additional arg,
# held fixed, not attributed. captum expands it across integration steps.
ig = IntegratedGradients(model)

sal_sum = torch.zeros(N_CLASSES, 12, SEQ_LEN)
sal_cnt = torch.zeros(N_CLASSES, dtype=torch.long)

model.eval()
for i in range(len(y_te_t)):
    signal = X_te_t[i:i + 1].to(DEVICE)      # (1, 12, 1000)
    demo = T_te_t[i:i + 1].to(DEVICE)        # (1, 4)  held fixed
    y = int(y_te_t[i])

    with torch.no_grad():
        pred = model(signal, demo).argmax(dim=1).item()
    if pred != y:                            # correctly classified only
        continue

    attr = ig.attribute(
        signal,
        baselines=torch.zeros_like(signal),
        target=y,
        additional_forward_args=(demo,),
        n_steps=N_STEPS,
    )
    sal_sum[y] += attr.squeeze(0).abs().detach().cpu()
    sal_cnt[y] += 1

print("correctly classified per class:", dict(zip(CLASS_NAMES, sal_cnt.tolist())))
avg_saliency = sal_sum / sal_cnt.clamp(min=1).view(-1, 1, 1)   # (5, 12, 1000)


# %% [cell 5] persist so cell 6/7 never recompute
np.save(os.path.join(OUT_DIR, "avg_saliency.npy"), avg_saliency.numpy())
np.save(os.path.join(OUT_DIR, "sal_counts.npy"), sal_cnt.numpy())
print("saved to", OUT_DIR)


# %% [cell 6] heatmaps + top 5% salient time bars
avg = np.load(os.path.join(OUT_DIR, "avg_saliency.npy"))
cnts = np.load(os.path.join(OUT_DIR, "sal_counts.npy"))

fig, axes = plt.subplots(N_CLASSES, 1, figsize=(16, 20))
for i, cname in enumerate(CLASS_NAMES):
    s = avg[i]                                  # (12, 1000), already magnitude
    im = axes[i].imshow(s, aspect='auto', cmap='hot')
    axes[i].set_yticks(range(12))
    axes[i].set_yticklabels(LEAD_NAMES)
    axes[i].set_title(f'saliency map - {cname} (n={int(cnts[i])})')
    axes[i].set_xlabel('time step')
    fig.colorbar(im, ax=axes[i], fraction=0.02)

    time_importance = s.sum(axis=0)             # (1000,)
    thr = np.percentile(time_importance, 95)
    for t in np.where(time_importance >= thr)[0]:
        axes[i].axvline(x=t, color='cyan', alpha=0.35, linewidth=0.6)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "saliency_maps_all_classes.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print("saved", fig_path)


# %% [cell 7] task 5 helpers - quantify instead of eyeballing
# caveat: the 10s strip holds ~10 unaligned beats, so peak time-step is an
# ABSOLUTE window, not an ecg phase. lead ranking is robust, phase claims are not.
for i, cname in enumerate(CLASS_NAMES):
    s = avg[i]
    lead_rank = np.argsort(s.mean(axis=1))[::-1]
    top_leads = [LEAD_NAMES[j] for j in lead_rank[:3]]
    peak_t = int(s.sum(axis=0).argmax())
    secs = peak_t / SEQ_LEN * 10.0
    print(f"{cname:5s} | top leads {top_leads} | peak step {peak_t} (~{secs:.2f}s)")


# %% [cell 8] separate clinical signal from two confounds (runs on cached array)
# confound 1: strip-edge saliency. the bilstm reads out from TERMINAL hidden
#   states (forward@t=end, backward@t=start), so boundary samples get outsized
#   gradient. architecture, not physiology. exclude a margin before locating
#   salient time windows.
# confound 2: dominant-lead effect. v2/ii top every class = global energy. subtract
#   the across-class mean lead profile to reveal per-class emphasis above baseline.
avg = np.load(os.path.join(OUT_DIR, "avg_saliency.npy"))   # (5, 12, 1000)
MARGIN = 50   # 0.5s each side, drops the terminal-state edge band

print("interior peak time step (edges excluded):")
for i, cname in enumerate(CLASS_NAMES):
    ti = avg[i].sum(axis=0).copy()
    ti[:MARGIN] = 0
    ti[-MARGIN:] = 0
    peak_t = int(ti.argmax())
    print(f"  {cname:5s} peak step {peak_t} (~{peak_t/SEQ_LEN*10:.2f}s)")

lead_mean = avg.mean(axis=2)                  # (5, 12) mean saliency per lead/class
baseline_profile = lead_mean.mean(axis=0)     # (12,) global lead profile
relative = lead_mean - baseline_profile       # (5, 12) elevation above global

print("\nclass-specific top leads (elevation above global baseline):")
for i, cname in enumerate(CLASS_NAMES):
    order = np.argsort(relative[i])[::-1]
    tops = [(LEAD_NAMES[j], round(float(relative[i][j]), 5)) for j in order[:3]]
    print(f"  {cname:5s} {tops}")