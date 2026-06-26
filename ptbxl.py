# %%
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

# %%
df = pd.read_csv('./ptb-xl/ptbxl_database.csv', index_col='ecg_id')
print(df.shape)
print(df.columns.tolist())
df.head()
# %%
print(df['age'].describe())          # age range and median
print(df['sex'].value_counts())      # how many male vs female (0/1 encoding)
print(df['filename_hr'].iloc[0])     # the path to recording #1's 500 Hz signal
print(df['report'].iloc[300])          # the cardiologist's text for recording #1

# %%
import wfdb
import matplotlib.pyplot as plt

signal, meta = wfdb.rdsamp('./ptb-xl/records500/00000/00001_hr')
print(signal.shape)   # (5000, 12)
print(meta['sig_name'])   # the 12 lead names, in order
# %%
signal_T = signal.T   # (12, 5000) -> one row per lead

lead_names = meta['sig_name']   # use the real names from the file
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

# %%# %%
preds = (all_probs >= 0.5).astype(int)
print(f'recall (sensitivity): {recall_score(all_true, preds):.3f}')
print(f'precision:            {precision_score(all_true, preds):.3f}')
# %%
rec = '208'   # 208 has plenty of both N and V beats
vrec = wfdb.rdrecord(f'mitdb/{rec}')
vann = wfdb.rdann(f'mitdb/{rec}', 'atr')
vsig = vrec.p_signal[:, 0]

plt.figure(figsize=(14, 3))
plt.plot(vsig[:3600], linewidth=0.8)   # first 10 seconds
plt.title(f'MIT-BIH record {rec} — MLII, first 10 s')
plt.xlabel('Sample (360 Hz)')
plt.ylabel('mV')
plt.tight_layout()
plt.savefig('mitbih_strip.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
rec = '208'   # 208 has plenty of both N and V beats
vrec = wfdb.rdrecord(f'mitdb/{rec}')
vann = wfdb.rdann(f'mitdb/{rec}', 'atr')
vsig = vrec.p_signal[:, 0]

plt.figure(figsize=(14, 3))
plt.plot(vsig[:3600], linewidth=0.8)   # first 10 seconds
plt.title(f'MIT-BIH record {rec} — MLII, first 10 s')
plt.xlabel('Sample (360 Hz)')
plt.ylabel('mV')
plt.tight_layout()
plt.savefig('mitbih_strip.png', dpi=150, bbox_inches='tight')
plt.show()
df.loc[1, ['baseline_drift', 'static_noise', 'burst_noise', 'report']]
# %%
import ast

print(type(df['scp_codes'].iloc[0]))   # <class 'str'>  -- it's text!
print(df['scp_codes'].iloc[0])         # looks like a dict but is a string

df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

print(type(df['scp_codes'].iloc[0]))   # <class 'dict'>  -- now a real dict
print(df['scp_codes'].iloc[0])
# %%
scp = pd.read_csv('./ptb-xl/scp_statements.csv', index_col=0)
print(scp.shape)
print(scp.columns.tolist())

# just the diagnostic codes (some codes describe rhythm or waveform form, not a diagnosis)
diagnostic = scp[scp['diagnostic'] == 1]
print(diagnostic.shape)
diagnostic[['description', 'diagnostic_class']].head(20)

# %%
# codebook: keep only diagnostic codes, grab their superclass mapping
scp = pd.read_csv('./ptb-xl/scp_statements.csv', index_col=0)
scp = scp[scp['diagnostic'] == 1]

def to_superclasses(scp_dict):
    classes = []
    for code in scp_dict.keys():
        if code in scp.index:
            classes.append(scp.loc[code, 'diagnostic_class'])
    return list(set(classes))   # set() removes duplicates

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
# find MI records with a strong (100%) confidence code
def max_mi_conf(scp_dict):
    mi_codes = ['IMI', 'AMI', 'ASMI', 'ALMI', 'ILMI', 'IPMI', 'IPLMI', 'LMI', 'PMI']
    return max([scp_dict.get(c, 0) for c in mi_codes], default=0)

strong_mi = df[df['scp_codes'].apply(max_mi_conf) == 100]
print(strong_mi.shape[0], 'records with a 100%-confidence MI code')
print(strong_mi.index[:5].tolist())   # pick one of these ecg_ids
# %%
norm_id = 3                       # a normal record
mi_id   = strong_mi.index[0]      # first strong-MI record

# the metadata stores each record's file path — read it straight from there
norm_path = './ptb-xl/' + df.loc[norm_id, 'filename_hr']
mi_path   = './ptb-xl/' + df.loc[mi_id,   'filename_hr']

norm_sig, _ = wfdb.rdsamp(norm_path)
mi_sig,   _ = wfdb.rdsamp(mi_path)

# lead II is column index 1
lead_ii = 1

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
