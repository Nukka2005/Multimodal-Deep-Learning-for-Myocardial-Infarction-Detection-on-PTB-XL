# Task 1 — label extraction: binary MI vs Normal -> ptbxl_labeled.csv
# %%
import ast
import pandas as pd

# %%
df = pd.read_csv('./ptb-xl/ptbxl_database.csv', index_col='ecg_id')
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

scp = pd.read_csv('./ptb-xl/scp_statements.csv', index_col=0)
scp = scp[scp['diagnostic'] == 1]
mi_codes = set(scp[scp['diagnostic_class'] == 'MI'].index)   # all 14 MI codes
print('MI codes:', mi_codes)

# %%
# STEMI/NSTEMI is not usable: STE_ appears in only 9 records, all at 0.0,
# and infarction_stadium1 encodes infarct age, not ST-elevation -> binary MI vs Normal.
print(df['infarction_stadium1'].value_counts(dropna=False))
print('records with an MI code:', df['scp_codes'].apply(lambda c: any(k in mi_codes for k in c)).sum())
print('records with STE_ >= 50:', df['scp_codes'].apply(lambda c: c.get('STE_', 0) >= 50).sum())

# %%
def assign_label(row):
    codes  = row['scp_codes']
    has_mi = any(c in mi_codes for c in codes)
    if codes.get('NORM', 0) >= 80 and not has_mi:
        return 0   # Normal
    if has_mi:
        return 1   # MI
    return None    # exclude ambiguous

df['label'] = df.apply(assign_label, axis=1)
labeled = df.dropna(subset=['label']).copy()   # new frame -> re-running is safe
labeled['label'] = labeled['label'].astype(int)
print(labeled['label'].map({0: 'Normal', 1: 'MI'}).value_counts())

labeled.to_csv('./ptb-xl/ptbxl_labeled.csv')
print('saved', len(labeled), 'labeled recordings')