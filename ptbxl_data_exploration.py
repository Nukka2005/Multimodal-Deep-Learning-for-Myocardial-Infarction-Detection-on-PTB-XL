# Exploration: understand PTB-XL (12-lead plot, superclass mix, NORM vs MI).
# %%
import ast
import wfdb
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('./ptb-xl/ptbxl_database.csv', index_col='ecg_id')
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
print(df.shape)
print(df.columns.tolist())
df.head()

# %%
print(df['age'].describe())
print(df['sex'].value_counts())
print(df['report'].iloc[300])

# %%  one recording, all 12 leads
signal, meta = wfdb.rdsamp('./ptb-xl/records500/00000/00001_hr')
signal_T = signal.T
fig, axes = plt.subplots(12, 1, figsize=(14, 18), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(signal_T[i], linewidth=0.8)
    ax.set_ylabel(meta['sig_name'][i], fontsize=9, rotation=0, labelpad=20)
    ax.set_yticks([])
plt.xlabel('Sample (500 Hz)')
plt.suptitle('PTB-XL recording 00001 — 12 leads')
plt.tight_layout()
plt.savefig('ptbxl_sample.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  diagnostic superclass distribution (column is NOT in the raw csv; build it)
scp = pd.read_csv('./ptb-xl/scp_statements.csv', index_col=0)
scp = scp[scp['diagnostic'] == 1]

def to_superclasses(scp_dict):
    return list({scp.loc[c, 'diagnostic_class'] for c in scp_dict if c in scp.index})

df['diagnostic_superclass'] = df['scp_codes'].apply(to_superclasses)
counts = df['diagnostic_superclass'].explode().value_counts()
print(counts)
counts.plot(kind='bar')
plt.ylabel('number of recordings')
plt.title('PTB-XL — recordings per diagnostic superclass')
plt.tight_layout()
plt.savefig('ptbxl_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  how many recordings carry more than one superclass
multi = df['diagnostic_superclass'].apply(len)
print((multi > 1).sum(), 'recordings have 2+ superclasses')
print((multi == 0).sum(), 'recordings have none')

# %%  NORM vs MI, lead II
norm_id, mi_id = 3, 8
norm_sig, _ = wfdb.rdsamp('./ptb-xl/' + df.loc[norm_id, 'filename_hr'])
mi_sig,   _ = wfdb.rdsamp('./ptb-xl/' + df.loc[mi_id,   'filename_hr'])
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(norm_sig[:, 1], linewidth=0.8)
axes[0].set_title(f'NORM — ecg_id {norm_id} (Lead II)')
axes[1].plot(mi_sig[:, 1], linewidth=0.8, color='firebrick')
axes[1].set_title(f'MI — ecg_id {mi_id} (Lead II)')
axes[1].set_xlabel('Sample (500 Hz)')
plt.tight_layout()
plt.savefig('ptbxl_norm_vs_mi.png', dpi=150, bbox_inches='tight')
plt.show()