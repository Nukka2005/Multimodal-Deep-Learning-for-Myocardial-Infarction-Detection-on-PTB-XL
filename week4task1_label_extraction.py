# Week 4, Task 1 — label extraction: 5-class diagnostic superclass -> ptbxl_5class.csv. Self-contained.
# %%
import ast
import pandas as pd
import matplotlib.pyplot as plt

# %%  reload raw metadata fresh -> this file stands alone, no earlier cells needed
meta = pd.read_csv('./ptb-xl/ptbxl_database.csv', index_col='ecg_id')
meta['scp_codes'] = meta['scp_codes'].apply(ast.literal_eval)

# build the 5 diagnostic superclasses (not present in the raw csv)
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

# %%  distribution bar chart (Task 1 deliverable)
single['superclass'].value_counts().plot(kind='bar')
plt.ylabel('recordings')
plt.title('Task 1 - single-label class distribution')
plt.tight_layout()
plt.savefig('task1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  save under a NEW name so the binary-task ptbxl_labeled.csv is untouched
single.to_csv('./ptb-xl/ptbxl_5class.csv')
print('saved ptbxl_5class.csv')
