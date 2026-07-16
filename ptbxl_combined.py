# %%
import os
import ast
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

# %%
df = pd.read_csv('./ptb-xl/ptbxl_database.csv', index_col='ecg_id')
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
print(df.shape)
print(df.columns.tolist())
df.head()

# %%
print(df['age'].describe())
print(df['sex'].value_counts())
print(df['filename_hr'].iloc[0])
print(df['report'].iloc[300])

# %%
signal, meta = wfdb.rdsamp('./ptb-xl/records500/00000/00001_hr')
print(signal.shape)        # (5000, 12)
print(meta['sig_name'])    # the 12 lead names, in order

# %%
signal_T = signal.T   # (12, 5000) -> one row per lead

lead_names = meta['sig_name']
fig, axes = plt.subplots(12, 1, figsize=(14, 18), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(signal_T[i], linewidth=0.8)
    ax.set_ylabel(lead_names[i], fontsize=9, rotation=0, labelpad=20)
    ax.set_yticks([])
plt.xlabel('Sample (500 Hz)')
plt.suptitle('PTB-XL recording 00001 — 12 leads')
plt.tight_layout()
plt.savefig('ptbxl_sample.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
print(type(df['scp_codes'].iloc[0]))
print(df['scp_codes'].iloc[0])

# %%
scp = pd.read_csv('./ptb-xl/scp_statements.csv', index_col=0)
print(scp.shape)

scp = scp[scp['diagnostic'] == 1]   # keep only diagnostic codes
mi_codes = set(scp[scp['diagnostic_class'] == 'MI'].index)
print('MI codes:', mi_codes)
scp[['description', 'diagnostic_class']].head(20)

# %%
def to_superclasses(scp_dict):
    classes = []
    for code in scp_dict.keys():
        if code in scp.index:
            classes.append(scp.loc[code, 'diagnostic_class'])
    return list(set(classes))

df['diagnostic_superclass'] = df['scp_codes'].apply(to_superclasses)

print(df['diagnostic_superclass'].iloc[0])
df[['scp_codes', 'diagnostic_superclass']].head(10)

# %%
counts = df['diagnostic_superclass'].explode().value_counts()
print(counts)

counts.plot(kind='bar')
plt.ylabel('number of recordings')
plt.title('PTB-XL — recordings per diagnostic superclass')
plt.tight_layout()
plt.savefig('ptbxl_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
multi = df['diagnostic_superclass'].apply(len)
print((multi > 1).sum(), 'recordings have 2+ superclasses')
print((multi == 0).sum(), 'recordings have none')
df[df['diagnostic_superclass'].apply(len) >= 2][['scp_codes', 'diagnostic_superclass']].head()

# %%
def max_mi_conf(scp_dict):
    mi = ['IMI', 'AMI', 'ASMI', 'ALMI', 'ILMI', 'IPMI', 'IPLMI', 'LMI', 'PMI']
    return max([scp_dict.get(c, 0) for c in mi], default=0)

strong_mi = df[df['scp_codes'].apply(max_mi_conf) == 100]
print(strong_mi.shape[0], 'records with a 100%-confidence MI code')
print(strong_mi.index[:5].tolist())

# %%
norm_id = 3
mi_id   = strong_mi.index[0]

norm_path = './ptb-xl/' + df.loc[norm_id, 'filename_hr']
mi_path   = './ptb-xl/' + df.loc[mi_id,   'filename_hr']

norm_sig, _ = wfdb.rdsamp(norm_path)
mi_sig,   _ = wfdb.rdsamp(mi_path)

lead_ii = 1   # lead II is column index 1
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(norm_sig[:, lead_ii], linewidth=0.8)
axes[0].set_title(f'NORM — ecg_id {norm_id} (Lead II)')
axes[1].plot(mi_sig[:, lead_ii], linewidth=0.8, color='firebrick')
axes[1].set_title(f'MI — ecg_id {mi_id} (Lead II)')
axes[1].set_xlabel('Sample (500 Hz)')
plt.tight_layout()
plt.savefig('ptbxl_norm_vs_mi.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
print(df.loc[177, 'scp_codes'])
print(df.loc[177, 'report'])

# %%
# why STEMI/NSTEMI is not usable: STE_ is present in only 9 records, all at 0.0,
# and infarction_stadium1 encodes infarct age, not ST-elevation. -> binary MI vs Normal.
print(df['infarction_stadium1'].value_counts(dropna=False))

has_mi  = df['scp_codes'].apply(lambda c: any(k in mi_codes for k in c))
has_ste = df['scp_codes'].apply(lambda c: c.get('STE_', 0) >= 50)
print('records with an MI code:', has_mi.sum())
print('records with STE_ >= 50:', has_ste.sum())

# %%
def assign_label(row):
    codes   = row['scp_codes']
    has_mi  = any(c in mi_codes for c in codes)
    is_norm = codes.get('NORM', 0) >= 80

    if is_norm and not has_mi:
        return 0   # Normal
    if has_mi:
        return 1   # MI
    return None    # exclude ambiguous

df['label'] = df.apply(assign_label, axis=1)

# keep df intact; put the filtered set in a new frame so re-running is safe
labeled = df.dropna(subset=['label']).copy()
labeled['label'] = labeled['label'].astype(int)

print(labeled['label'].map({0: 'Normal', 1: 'MI'}).value_counts())

# %%
labeled.to_csv('./ptb-xl/ptbxl_labeled.csv')
print('saved', len(labeled), 'labeled recordings')
# %%
import pandas as pd, numpy as np, wfdb

df = pd.read_csv('./ptb-xl/ptbxl_labeled.csv', index_col='ecg_id')

train_df = df[df['strat_fold'] <= 8]
val_df   = df[df['strat_fold'] == 9]
test_df  = df[df['strat_fold'] == 10]

print(len(train_df), len(val_df), len(test_df))

def load_split(split_df):
    X, y = [], []
    for ecg_id, row in split_df.iterrows():
        sig, _ = wfdb.rdsamp('./ptb-xl/' + row['filename_lr'])  # (1000, 12)
        X.append(sig.T)                                          # (12, 1000)
        y.append(row['label'])
    return np.array(X, dtype='float32'), np.array(y, dtype='int64')
# %%
X_train, y_train = load_split(train_df)
X_val,   y_val   = load_split(val_df)
X_test,  y_test  = load_split(test_df)

print('train:', X_train.shape, np.bincount(y_train))
print('val:  ', X_val.shape,   np.bincount(y_val))
print('test: ', X_test.shape,  np.bincount(y_test))
# %%
import torch
import torch.nn as nn

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
        x = self.gap(x)                                     # (B, 128, 1)
        x = x.flatten(1)                                    # (B, 128)
        x = self.drop(x)                                    # (B, 128)
        x = self.fc(x)                                      # (B, 2)
        return x
# %%
model = ECGCNN()
dummy = torch.randn(8, 12, 1000)   # 8 fake recordings, 12 leads, 1000 samples
out = model(dummy)
print(out.shape)                   # expect torch.Size([8, 2])
# %%
import torch, numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

Xtr = torch.tensor(X_train, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.long)      # CrossEntropy wants int64/long
Xva = torch.tensor(X_val,   dtype=torch.float32)
yva = torch.tensor(y_val,   dtype=torch.long)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=32, shuffle=False)

# class weights: up-weight the rarer class so the model can't ignore MI
weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
weights = torch.tensor(weights, dtype=torch.float32)
print('class weights:', weights)
# %%
import copy

torch.manual_seed(42)
model = ECGCNN(n_classes=2, p_drop=0.3)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []
best_val = float('inf')
best_state = None

for epoch in range(30):
    # ---- train ----
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_loader.dataset)

    # ---- validate ----
    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            loss = criterion(out, yb)
            running += loss.item() * xb.size(0)
    val_loss = running / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # ---- checkpoint the best model ----
    if val_loss < best_val:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())

    print(f'epoch {epoch+1:2d}/30   train {train_loss:.4f}   val {val_loss:.4f}')

model.load_state_dict(best_state)   # restore the best checkpoint
print(f'best val loss: {best_val:.4f}')
# %%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(1, 31), train_losses, marker='o', label='train')
plt.plot(range(1, 31), val_losses,   marker='o', label='validation')
plt.xlabel('epoch'); plt.ylabel('loss')
plt.title('Task 2 — training vs validation loss')
plt.legend(); plt.grid(True)
plt.savefig('task2_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

Xte = torch.tensor(X_test, dtype=torch.float32)
yte = torch.tensor(y_test, dtype=torch.long)
test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=32, shuffle=False)

model.eval()
all_probs, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        probs  = torch.softmax(logits, dim=1)   # (B, 2) -> class probabilities
        all_probs.append(probs)
        all_true.append(yb)

all_probs = torch.cat(all_probs).numpy()   # (N, 2)
all_true  = torch.cat(all_true).numpy()    # (N,)

# probability of the MI class (column 1) is what we score
mi_probs = all_probs[:, 1]
auc = roc_auc_score(all_true, mi_probs)
print(f'Test AUROC (MI vs Normal): {auc:.4f}')

fpr, tpr, _ = roc_curve(all_true, mi_probs)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC (AUROC = {auc:.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Task 2 — ROC (MI vs Normal, test set)')
plt.legend(); plt.grid(True)
plt.savefig('task2_roc.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
torch.save(model.state_dict(), 'ecg_cnn_branch.pt')
print('saved ECG signal branch')

# %%
import pandas as pd, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv('./ptb-xl/ptbxl_labeled.csv', index_col='ecg_id')

train_df = df[df['strat_fold'] <= 8]
val_df   = df[df['strat_fold'] == 9]
test_df  = df[df['strat_fold'] == 10]

# z-score age using TRAIN statistics only (never peek at val/test)
age_mean = train_df['age'].mean()
age_std  = train_df['age'].std()

def make_xy(split_df):
    feats = split_df[['age', 'sex']].copy()
    feats['age'] = (feats['age'] - age_mean) / age_std
    X = feats.values.astype('float32')          # (N, 2)
    y = split_df['label'].values.astype('int64')
    return X, y

Xtr, ytr = make_xy(train_df)
Xva, yva = make_xy(val_df)
Xte, yte = make_xy(test_df)

print(Xtr.shape, Xva.shape, Xte.shape)   # (~11500, 2) ...
# %%
from sklearn.utils.class_weight import compute_class_weight

train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)), batch_size=32, shuffle=False)

weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=ytr)
weights = torch.tensor(weights, dtype=torch.float32)
# %%
class TabularMLP(nn.Module):
    def __init__(self, in_features=2, n_classes=2, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)

model_mlp = TabularMLP()
print(model_mlp(torch.randn(8, 2)).shape)   # expect torch.Size([8, 2])
# %%
import copy

torch.manual_seed(42)
model_mlp = TabularMLP()
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
        best_val = va_loss
        best_state = copy.deepcopy(model_mlp.state_dict())
    if (epoch + 1) % 5 == 0:
        print(f'epoch {epoch+1:2d}/50   train {tr_loss:.4f}   val {va_loss:.4f}')

model_mlp.load_state_dict(best_state)
print(f'best val loss: {best_val:.4f}')
# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

plt.figure()
plt.plot(range(1, 51), tr_losses, label='train')
plt.plot(range(1, 51), va_losses, label='validation')
plt.xlabel('epoch'); plt.ylabel('loss')
plt.title('Task 3 — MLP training vs validation loss')
plt.legend(); plt.grid(True)
plt.savefig('task3_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
model_mlp.eval()
all_probs, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        probs = torch.softmax(model_mlp(xb), dim=1)
        all_probs.append(probs); all_true.append(yb)
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
plt.title('Task 3 — ROC (tabular: age + sex)')
plt.legend(); plt.grid(True)
plt.savefig('task3_roc.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
torch.save(model_mlp.state_dict(), 'tabular_mlp_branch.pt')
print('saved tabular branch')
# %%  ===== THIS WEEK — Task 1: 5-class single-label extraction =====
import ast
import pandas as pd
import matplotlib.pyplot as plt

# reload raw metadata fresh -> this cell stands alone, no earlier cells needed
meta = pd.read_csv('./ptb-xl/ptbxl_database.csv', index_col='ecg_id')
meta['scp_codes'] = meta['scp_codes'].apply(ast.literal_eval)

# build the 5 diagnostic superclasses
#  (not present in the raw csv)
scp5 = pd.read_csv('./ptb-xl/scp_statements.csv', index_col=0)
scp5 = scp5[scp5['diagnostic'] == 1]

def get_superclasses(scp_dict):
    return list({scp5.loc[c, 'diagnostic_class'] for c in scp_dict if c in scp5.index})

meta['superclasses'] = meta['scp_codes'].apply(get_superclasses)

# keep ONLY recordings with exactly one superclass (mutually exclusive -> softmax)
single = meta[meta['superclasses'].apply(len) == 1].copy()
single['superclass'] = single['superclasses'].apply(lambda lst: lst[0])

label_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
single['label'] = single['superclass'].map(label_map)

print(single['superclass'].value_counts())
print('kept', len(single), 'of', len(meta), 'recordings')

# distribution bar chart (Task 1 deliverable)
single['superclass'].value_counts().plot(kind='bar')
plt.ylabel('recordings')
plt.title('Task 1 - single-label class distribution')
plt.tight_layout()
plt.savefig('task1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# save under a NEW name so last week's binary ptbxl_labeled.csv is untouched
single.to_csv('./ptb-xl/ptbxl_5class.csv')
print('saved ptbxl_5class.csv')
# %%
import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, n_classes=5, hidden_size=128, p_drop=0.3, bidirectional=True):
        super().__init__()
        # --- CNN feature extractor (same as last week) ---
        self.conv1 = nn.Conv1d(12, 32,  kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64,  kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()

        # --- LSTM sequence reader ---
        self.lstm = nn.LSTM(
            input_size=128,            # 128 features per timestep (CNN channels)
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,          # expect (batch, seq, features)
            bidirectional=bidirectional,
        )

        # --- classifier head ---
        lstm_out = hidden_size * (2 if bidirectional else 1)   # 256 if bidirectional
        self.drop = nn.Dropout(p_drop)
        self.fc   = nn.Linear(lstm_out, n_classes)

    def forward(self, x):
        # CNN
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # (B, 32, 500)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # (B, 64, 250)
        x = self.relu(self.bn3(self.conv3(x)))              # (B, 128, 250)

        # bridge: (B, channels, length) -> (B, length, features)
        x = x.permute(0, 2, 1)                              # (B, 250, 128)

        # LSTM
        outputs, (h_n, c_n) = self.lstm(x)
        # h_n shape: (num_directions, B, hidden). Grab both directions' last states.
        emb = torch.cat([h_n[-2], h_n[-1]], dim=1)          # (B, 256)

        emb = self.drop(emb)
        return self.fc(emb) 
# %%
model = CNNLSTM(n_classes=5, hidden_size=128, bidirectional=True)
out = model(torch.randn(8, 12, 1000))
print(out.shape)   # expect torch.Size([8, 5])                                # (B, 5)
# %%
import numpy as np, pandas as pd, wfdb, torch, copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

df5 = pd.read_csv('./ptb-xl/ptbxl_5class.csv', index_col='ecg_id')
tr_df = df5[df5['strat_fold'] <= 8]
va_df = df5[df5['strat_fold'] == 9]
te_df = df5[df5['strat_fold'] == 10]
print(len(tr_df), len(va_df), len(te_df))

def load_split(split_df):
    X, y = [], []
    for _, row in split_df.iterrows():
        sig, _ = wfdb.rdsamp('./ptb-xl/' + row['filename_lr'])   # (1000, 12)
        X.append(sig.T)                                          # (12, 1000)
        y.append(row['label'])
    return np.array(X, dtype='float32'), np.array(y, dtype='int64')

X_tr, y_tr = load_split(tr_df)
X_va, y_va = load_split(va_df)
X_te, y_te = load_split(te_df)
print('train:', X_tr.shape, np.bincount(y_tr))   # 5 counts now

train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)), batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=32, shuffle=False)

# class weights now span all 5 classes -> HYP gets the biggest weight
weights5 = torch.tensor(
    compute_class_weight('balanced', classes=np.array([0,1,2,3,4]), y=y_tr),
    dtype=torch.float32)
print('class weights:', weights5)
# %%
print(torch.cuda.is_available())        # True only if a CUDA GPU + CUDA PyTorch are set up
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using', device)

print(next(model.parameters()).device)
print(xb.device)
print(yb.device)
print(weights5.device)
# %%
model = CNNLSTM(
    n_classes=5,
    hidden_size=128,
    bidirectional=True,
    p_drop=0.3
).to(device)

weights5 = weights5.to(device)

criterion = nn.CrossEntropyLoss(weight=weights5)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# %%
for epoch in range(30):

    model.train()

    running = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        outputs = model(xb)

        loss = criterion(outputs, yb)

        loss.backward()

        optimizer.step()

        running += loss.item() * xb.size(0)

    tr = running / len(train_loader.dataset)    

    model.eval()

    running = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            outputs = model(xb)

            running += criterion(outputs, yb).item() * xb.size(0)
    va = running / len(val_loader.dataset)

    tr_losses.append(tr); va_losses.append(va)
    if va < best_val:
        best_val, best_state = va, copy.deepcopy(model.state_dict())
    print(f'epoch {epoch+1:2d}/30   train {tr:.4f}   val {va:.4f}')

model.load_state_dict(best_state)
print(f'best val loss: {best_val:.4f}')
# %%
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

model.eval()

all_true = []
all_pred = []
all_probs = []

with torch.no_grad():
    for xb, yb in test_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        outputs = model(xb)

        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_true.extend(yb.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_probs = np.array(all_probs)

print("Test Accuracy:", accuracy_score(all_true, all_pred))
print("all_true shape:", all_true.shape)
print("all_probs shape:", all_probs.shape)
# %%
import matplotlib.pyplot as plt

epochs = range(1, len(tr_losses) + 1)

plt.figure(figsize=(8,5))

plt.plot(epochs, tr_losses, marker='o', linewidth=2, label='Training Loss')
plt.plot(epochs, va_losses, marker='s', linewidth=2, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('CNN-LSTM Training and Validation Loss')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('week4task1_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

y_true_bin = label_binarize(all_true, classes=[0,1,2,3,4])

plt.figure(figsize=(7,7))

for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.3f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('One-vs-Rest ROC Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('week4task1_multiclass_roc.png', dpi=150)
plt.show()
# %%
from sklearn.metrics import roc_auc_score

macro_auroc = roc_auc_score(
    all_true,
    all_probs,
    multi_class='ovr',
    average='macro'
)

print(f"Macro AUROC: {macro_auroc:.4f}")
# %% ===== TASK 3: Tabular Encoder (MLP) =====
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

features = ['age', 'sex', 'height', 'weight']

# 1. Extract tabular features
X_tr_tab_raw = tr_df[features].copy()
X_va_tab_raw = va_df[features].copy()
X_te_tab_raw = te_df[features].copy()

# 2. Impute missing values (fit on train, transform all)
imputer = SimpleImputer(strategy='median')
X_tr_imp = imputer.fit_transform(X_tr_tab_raw)
X_va_imp = imputer.transform(X_va_tab_raw)
X_te_imp = imputer.transform(X_te_tab_raw)

# 3. Normalize continuous features (age, height, weight -> indices 0, 2, 3)
# We don't necessarily need to scale 'sex', but scaling all features is standard and harmless
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_imp)
X_va_scaled = scaler.transform(X_va_imp)
X_te_scaled = scaler.transform(X_te_imp)

# 4. Convert to PyTorch DataLoaders
train_loader_tab = DataLoader(TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_tr)), batch_size=32, shuffle=True)
val_loader_tab   = DataLoader(TensorDataset(torch.tensor(X_va_scaled, dtype=torch.float32), torch.tensor(y_va)), batch_size=32, shuffle=False)
test_loader_tab  = DataLoader(TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32), torch.tensor(y_te)), batch_size=32, shuffle=False)

print("Tabular data ready. Shape:", X_tr_scaled.shape)

# %%
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

# Initialize
model_tab = TabularMLP().to(device)
criterion_tab = nn.CrossEntropyLoss(weight=weights5) # Reuse weights5 from Task 2
optimizer_tab = torch.optim.Adam(model_tab.parameters(), lr=1e-3)

tr_losses_tab, va_losses_tab = [], []
best_val_tab, best_state_tab = float('inf'), None

# Train
for epoch in range(30):
    model_tab.train()
    running = 0.0
    for xb, yb in train_loader_tab:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_tab.zero_grad()
        loss = criterion_tab(model_tab(xb), yb)
        loss.backward()
        optimizer_tab.step()
        running += loss.item() * xb.size(0)
    tr = running / len(train_loader_tab.dataset)    

    model_tab.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader_tab:
            xb, yb = xb.to(device), yb.to(device)
            running += criterion_tab(model_tab(xb), yb).item() * xb.size(0)
    va = running / len(val_loader_tab.dataset)

    tr_losses_tab.append(tr); va_losses_tab.append(va)
    if va < best_val_tab:
        best_val_tab, best_state_tab = va, copy.deepcopy(model_tab.state_dict())
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'epoch {epoch+1:2d}/30   train {tr:.4f}   val {va:.4f}')

model_tab.load_state_dict(best_state_tab)
print(f'best val loss: {best_val_tab:.4f}')
# %%
import matplotlib.pyplot as plt

# Generate x-axis values based on the number of epochs trained
epochs_tab = range(1, len(tr_losses_tab) + 1)

plt.figure(figsize=(8,5))

# Plot both training and validation losses for the tabular model
plt.plot(epochs_tab, tr_losses_tab, marker='o', linewidth=2, label='Training Loss')
plt.plot(epochs_tab, va_losses_tab, marker='s', linewidth=2, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Tabular MLP: Training and Validation Loss')
plt.xticks(epochs_tab)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('week4task3_tabular_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
model_tab.eval()
all_true_tab, all_pred_tab, all_probs_tab = [], [], []

with torch.no_grad():
    for xb, yb in test_loader_tab:
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

print("Tabular Test Accuracy:", accuracy_score(all_true_tab, all_pred_tab))
print(f"Tabular Macro AUROC: {macro_auroc_tab:.4f}")

# Plot Tabular ROC
y_true_bin_tab = label_binarize(all_true_tab, classes=[0,1,2,3,4])
plt.figure(figsize=(7,7))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin_tab[:, i], all_probs_tab[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.3f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Tabular Model: One-vs-Rest ROC Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('week4task3_tabular_roc.png', dpi=150)
plt.show()
# %% ===== TASK 4: Multimodal Fusion =====
class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, X_sig, X_tab, y):
        self.X_sig = torch.tensor(X_sig, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_sig[idx], self.X_tab[idx], self.y[idx]

train_loader_fuse = DataLoader(MultimodalDataset(X_tr, X_tr_scaled, y_tr), batch_size=32, shuffle=True)
val_loader_fuse   = DataLoader(MultimodalDataset(X_va, X_va_scaled, y_va), batch_size=32, shuffle=False)
test_loader_fuse  = DataLoader(MultimodalDataset(X_te, X_te_scaled, y_te), batch_size=32, shuffle=False)

print("Multimodal DataLoaders ready.")
# %%
class FusionModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        # --- Signal Encoder Backbone ---
        self.conv1 = nn.Conv1d(12, 32, 7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()
        self.lstm  = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=True)
        
        # --- Tabular Encoder Backbone ---
        self.fc_tab = nn.Linear(4, 16)
        
        # --- Fusion Head ---
        # 256 from bidirectional LSTM + 16 from MLP = 272
        self.drop = nn.Dropout(0.3)
        self.fc_fusion = nn.Linear(256 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        # 1. Process Signal
        s = self.pool(self.relu(self.bn1(self.conv1(x_sig))))
        s = self.pool(self.relu(self.bn2(self.conv2(s))))
        s = self.relu(self.bn3(self.conv3(s)))
        s = s.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(s)
        emb_sig = torch.cat([h_n[-2], h_n[-1]], dim=1) # (B, 256)
        
        # 2. Process Tabular
        emb_tab = self.relu(self.fc_tab(x_tab)) # (B, 16)
        
        # 3. Fuse and Classify
        emb_fused = torch.cat([emb_sig, emb_tab], dim=1) # (B, 272)
        emb_fused = self.drop(emb_fused)
        return self.fc_fusion(emb_fused)

model_fuse = FusionModel().to(device)
criterion_fuse = nn.CrossEntropyLoss(weight=weights5)
optimizer_fuse = torch.optim.Adam(model_fuse.parameters(), lr=1e-3)

tr_losses_fuse, va_losses_fuse = [], []
best_val_fuse, best_state_fuse = float('inf'), None

for epoch in range(30):
    model_fuse.train()
    running = 0.0
    for x_sig, x_tab, yb in train_loader_fuse:
        x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
        
        optimizer_fuse.zero_grad()
        loss = criterion_fuse(model_fuse(x_sig, x_tab), yb)
        loss.backward()
        optimizer_fuse.step()
        running += loss.item() * yb.size(0)
    tr = running / len(train_loader_fuse.dataset)    

    model_fuse.eval()
    running = 0.0
    with torch.no_grad():
        for x_sig, x_tab, yb in val_loader_fuse:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            running += criterion_fuse(model_fuse(x_sig, x_tab), yb).item() * yb.size(0)
    va = running / len(val_loader_fuse.dataset)

    tr_losses_fuse.append(tr); va_losses_fuse.append(va)
    if va < best_val_fuse:
        best_val_fuse, best_state_fuse = va, copy.deepcopy(model_fuse.state_dict())
    
    print(f'epoch {epoch+1:2d}/30   train {tr:.4f}   val {va:.4f}')

model_fuse.load_state_dict(best_state_fuse)
print(f'best fusion val loss: {best_val_fuse:.4f}')
# %%
model_fuse.eval()
all_pred_fuse, all_probs_fuse = [], []

with torch.no_grad():
    for x_sig, x_tab, yb in test_loader_fuse:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        outputs = model_fuse(x_sig, x_tab)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_pred_fuse.extend(preds.cpu().numpy())
        all_probs_fuse.extend(probs.cpu().numpy())

all_pred_fuse = np.array(all_pred_fuse)
all_probs_fuse = np.array(all_probs_fuse)

macro_auroc_fuse = roc_auc_score(all_true, all_probs_fuse, multi_class='ovr', average='macro')

print(f"Signal-Only Macro AUROC: {macro_auroc:.4f}")
print(f"Tabular-Only Macro AUROC: {macro_auroc_tab:.4f}")
print(f"Fusion Macro AUROC: {macro_auroc_fuse:.4f}")

# --- 3-Way ROC Curve Comparison (Micro-Average) ---
plt.figure(figsize=(8,8))

# 1. Signal Model (From Task 2)
fpr_sig, tpr_sig, _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
plt.plot(fpr_sig, tpr_sig, label=f'Signal Only (Micro AUC = {auc(fpr_sig, tpr_sig):.3f})', color='blue', linewidth=2)

# 2. Tabular Model (From Task 3)
fpr_tab, tpr_tab, _ = roc_curve(y_true_bin_tab.ravel(), all_probs_tab.ravel())
plt.plot(fpr_tab, tpr_tab, label=f'Tabular Only (Micro AUC = {auc(fpr_tab, tpr_tab):.3f})', color='orange', linewidth=2)

# 3. Fusion Model (Task 4)
fpr_fuse, tpr_fuse, _ = roc_curve(y_true_bin.ravel(), all_probs_fuse.ravel())
plt.plot(fpr_fuse, tpr_fuse, label=f'Fusion (Micro AUC = {auc(fpr_fuse, tpr_fuse):.3f})', color='green', linewidth=2)

plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multimodal Comparison: Micro-Average ROC Curves')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('week4task4_multimodal_comparison.png', dpi=150)
plt.show()

# %% ===== TASK 4 ALTERNATIVE: Transfer Learning Fusion =====
import copy
import torch.nn as nn

class TransferFusionModel(nn.Module):
    def __init__(self, pretrained_sig_model, pretrained_tab_model, n_classes=5):
        super().__init__()
        
        # 1. Deepcopy the pre-trained models so we don't accidentally alter 
        # the original Task 2 and Task 3 models in memory.
        self.sig_encoder = copy.deepcopy(pretrained_sig_model)
        self.tab_encoder = copy.deepcopy(pretrained_tab_model)
        
        # 2. Chop off the heads! 
        # We replace the final Linear layers with nn.Identity().
        # Now, calling these encoders will return the embeddings, not the logits.
        self.sig_encoder.fc = nn.Identity()       # Task 2 model's final layer was named 'fc'
        self.tab_encoder.fc2 = nn.Identity()      # Task 3 model's final layer was named 'fc2'
        
        # 3. Build the new Joint Classifier Head
        self.drop = nn.Dropout(0.3)
        self.fc_fusion = nn.Linear(256 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        # Pass inputs through the pre-trained backbones
        emb_sig = self.sig_encoder(x_sig)  # Returns (B, 256)
        emb_tab = self.tab_encoder(x_tab)  # Returns (B, 16)
        
        # Concatenate and classify
        emb_fused = torch.cat([emb_sig, emb_tab], dim=1) # (B, 272)
        emb_fused = self.drop(emb_fused)
        return self.fc_fusion(emb_fused)

# Instantiate the model using your existing trained models
model_transfer = TransferFusionModel(model, model_tab).to(device)
criterion_transfer = nn.CrossEntropyLoss(weight=weights5)

# Notice the learning rate is smaller here (1e-4 instead of 1e-3). 
optimizer_transfer = torch.optim.Adam(model_transfer.parameters(), lr=1e-4)

tr_losses_trans, va_losses_trans = [], []
best_val_trans, best_state_trans = float('inf'), None

# Training Loop (Slightly shorter epochs since it's already mostly trained)
for epoch in range(15):
    model_transfer.train()
    running = 0.0
    for x_sig, x_tab, yb in train_loader_fuse:
        x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
        
        optimizer_transfer.zero_grad()
        loss = criterion_transfer(model_transfer(x_sig, x_tab), yb)
        loss.backward()
        optimizer_transfer.step()
        running += loss.item() * yb.size(0)
    tr = running / len(train_loader_fuse.dataset)    

    model_transfer.eval()
    running = 0.0
    with torch.no_grad():
        for x_sig, x_tab, yb in val_loader_fuse:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            running += criterion_transfer(model_transfer(x_sig, x_tab), yb).item() * yb.size(0)
    va = running / len(val_loader_fuse.dataset)

    tr_losses_trans.append(tr); va_losses_trans.append(va)
    if va < best_val_trans:
        best_val_trans, best_state_trans = va, copy.deepcopy(model_transfer.state_dict())
    
    print(f'epoch {epoch+1:2d}/15   train {tr:.4f}   val {va:.4f}')

model_transfer.load_state_dict(best_state_trans)
print(f'best transfer fusion val loss: {best_val_trans:.4f}')
# %%
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

model_transfer.eval()
all_pred_trans, all_probs_trans = [], []

with torch.no_grad():
    for x_sig, x_tab, yb in test_loader_fuse:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        outputs = model_transfer(x_sig, x_tab)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_pred_trans.extend(preds.cpu().numpy())
        all_probs_trans.extend(probs.cpu().numpy())

all_pred_trans = np.array(all_pred_trans)
all_probs_trans = np.array(all_probs_trans)

macro_auroc_trans = roc_auc_score(all_true, all_probs_trans, multi_class='ovr', average='macro')

print(f"Transfer Fusion Macro AUROC: {macro_auroc_trans:.4f}")

# %%
import matplotlib.pyplot as plt

# Generate x-axis values based on the 15 epochs
epochs_trans = range(1, len(tr_losses_trans) + 1)

plt.figure(figsize=(8,5))

plt.plot(epochs_trans, tr_losses_trans, marker='o', linewidth=2, label='Training Loss')
plt.plot(epochs_trans, va_losses_trans, marker='s', linewidth=2, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Transfer Fusion Model: Training and Validation Loss')
plt.xticks(epochs_trans)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('week4task4_transfer_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

# 1. Signal Model (From Task 2)
fpr_sig, tpr_sig, _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
plt.plot(fpr_sig, tpr_sig, label=f'Signal Only (Micro AUC = {auc(fpr_sig, tpr_sig):.3f})', color='blue', linewidth=2)

# 2. Tabular Model (From Task 3)
fpr_tab, tpr_tab, _ = roc_curve(y_true_bin_tab.ravel(), all_probs_tab.ravel())
plt.plot(fpr_tab, tpr_tab, label=f'Tabular Only (Micro AUC = {auc(fpr_tab, tpr_tab):.3f})', color='orange', linewidth=2)

# 3. Fusion Model (From Scratch)
fpr_fuse, tpr_fuse, _ = roc_curve(y_true_bin.ravel(), all_probs_fuse.ravel())
plt.plot(fpr_fuse, tpr_fuse, label=f'Fusion From Scratch (Micro AUC = {auc(fpr_fuse, tpr_fuse):.3f})', color='green', linewidth=2)

# 4. Fusion Model (Transfer Learning)
fpr_trans, tpr_trans, _ = roc_curve(y_true_bin.ravel(), all_probs_trans.ravel())
plt.plot(fpr_trans, tpr_trans, label=f'Fusion Transfer (Micro AUC = {auc(fpr_trans, tpr_trans):.3f})', color='red', linewidth=2, linestyle='-.')

plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ultimate Multimodal Comparison: Micro-Average ROC Curves')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('week4task4_ultimate_comparison.png', dpi=150)
plt.show()
# %%
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class TransformerFusionModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        # --- Signal Encoder Backbone (CNN) ---
        self.conv1 = nn.Conv1d(12, 32, 7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()
        
        # --- Transformer Sequence Reader ---
        # 1. Learnable Positional Embedding (1 batch, 250 steps, 128 features)
        self.pos_emb = nn.Parameter(torch.randn(1, 250, 128))
        
        # 2. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,          # Must match the CNN output features
            nhead=8,              # 8 attention heads
            dim_feedforward=256,  # Hidden layer size inside the transformer
            batch_first=True,     # Expects (batch, seq, feature)
            dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # --- Tabular Encoder Backbone ---
        self.fc_tab = nn.Linear(4, 16)
        
        # --- Fusion Head ---
        # 128 from Transformer + 16 from Tabular = 144
        self.drop = nn.Dropout(0.3)
        self.fc_fusion = nn.Linear(128 + 16, n_classes)

    def forward(self, x_sig, x_tab):
        # 1. Process Signal
        s = self.pool(self.relu(self.bn1(self.conv1(x_sig))))
        s = self.pool(self.relu(self.bn2(self.conv2(s))))
        s = self.relu(self.bn3(self.conv3(s)))
        
        # Bridge: (B, 128, 250) -> (B, 250, 128)
        s = s.permute(0, 2, 1)
        
        # Inject position information
        s = s + self.pos_emb
        
        # Read sequence with Self-Attention
        s = self.transformer(s)  # Output shape is still (B, 250, 128)
        
        # Aggregate the 250 time steps into a single 128-dim summary vector
        # Global Average Pooling across the time dimension (dim=1)
        emb_sig = s.mean(dim=1)  # Shape: (B, 128)
        
        # 2. Process Tabular
        emb_tab = self.relu(self.fc_tab(x_tab)) # Shape: (B, 16)
        
        # 3. Fuse and Classify
        emb_fused = torch.cat([emb_sig, emb_tab], dim=1) # Shape: (B, 144)
        emb_fused = self.drop(emb_fused)
        return self.fc_fusion(emb_fused)

# --- Initialization & Training ---
model_trans_fuse = TransformerFusionModel().to(device)
criterion_trans = nn.CrossEntropyLoss(weight=weights5)
optimizer_trans = torch.optim.Adam(model_trans_fuse.parameters(), lr=1e-3)

tr_losses_tfuse, va_losses_tfuse = [], []
best_val_tfuse, best_state_tfuse = float('inf'), None

for epoch in range(30):
    model_trans_fuse.train()
    running = 0.0
    for x_sig, x_tab, yb in train_loader_fuse:
        x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
        
        optimizer_trans.zero_grad()
        loss = criterion_trans(model_trans_fuse(x_sig, x_tab), yb)
        loss.backward()
        optimizer_trans.step()
        running += loss.item() * yb.size(0)
    tr = running / len(train_loader_fuse.dataset)    

    model_trans_fuse.eval()
    running = 0.0
    with torch.no_grad():
        for x_sig, x_tab, yb in val_loader_fuse:
            x_sig, x_tab, yb = x_sig.to(device), x_tab.to(device), yb.to(device)
            running += criterion_trans(model_trans_fuse(x_sig, x_tab), yb).item() * yb.size(0)
    va = running / len(val_loader_fuse.dataset)

    tr_losses_tfuse.append(tr); va_losses_tfuse.append(va)
    if va < best_val_tfuse:
        best_val_tfuse, best_state_tfuse = va, copy.deepcopy(model_trans_fuse.state_dict())
    
    print(f'epoch {epoch+1:2d}/30   train {tr:.4f}   val {va:.4f}')

model_trans_fuse.load_state_dict(best_state_tfuse)

# --- Evaluation ---
model_trans_fuse.eval()
all_probs_tfuse = []

with torch.no_grad():
    for x_sig, x_tab, yb in test_loader_fuse:
        x_sig, x_tab = x_sig.to(device), x_tab.to(device)
        outputs = model_trans_fuse(x_sig, x_tab)
        probs = F.softmax(outputs, dim=1)
        all_probs_tfuse.extend(probs.cpu().numpy())

import numpy as np
all_probs_tfuse = np.array(all_probs_tfuse)

macro_auroc_tfuse = roc_auc_score(all_true, all_probs_tfuse, multi_class='ovr', average='macro')
print(f"\nTransformer Fusion Macro AUROC: {macro_auroc_tfuse:.4f}")
# %%