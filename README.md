# Multimodal Myocardial Infarction Detection on PTB-XL

Summer research project at Habib University, supervised by **Dr. Zafar Iqbal** through the Office of Research.

This repository builds a diagnostic classifier for 12 lead electrocardiograms that combines two very different kinds of evidence: the raw electrical signal from the heart, and the patient metadata that sits alongside it. Every component here was written from scratch in PyTorch rather than assembled from prebuilt pipelines, because the point was to understand each piece, not just run it.

---

## What this project is asking

An ECG carries the actual electrical fingerprint of an infarction. A patient's age and sex carry something quite different: population level risk. Neither is the whole picture. The question driving this work is whether combining them in one model beats either one alone, and if so, where the gain comes from.

The dataset is [PTB-XL](https://physionet.org/content/ptb-xl/), 21,799 clinical 12 lead recordings with expert SCP code annotations, patient demographics, and a prebuilt stratified fold assignment that keeps every patient inside a single fold.

---

## Results

### Binary task: MI versus Normal

| Model | Input | Test AUROC |
|---|---|---|
| 1D CNN | 12 lead signal | **0.972** |
| MLP | age, sex | 0.798 |

The gap here is the finding. Demographics carry a real signal (MI incidence climbs steeply with age, and 0.798 is well clear of chance), but they cannot diagnose a specific infarction. That evidence lives in the waveform. The two models are wrong in different ways, which is exactly the precondition that makes fusing them worth doing.

### Five class task: NORM / MI / STTC / CD / HYP

| Model | Macro AUROC |
|---|---|
| CNN + Transformer encoder, fused with demographics | **0.9138** |
| Published benchmark, best single models | ~0.93 |

Per class breakdown for the Transformer fusion model:

| Class | AUROC |
|---|---|
| NORM | 0.8958 |
| MI | 0.9075 |
| STTC | 0.9302 |
| CD | 0.9020 |
| HYP | 0.9337 |

---

## Three findings worth reading

### 1. The STEMI versus NSTEMI split is not possible on this dataset

The project originally aimed at a three class scheme: Normal, STEMI, NSTEMI. That distinction is clinically real and rests entirely on one question, whether ST elevation is present. PTB-XL has an `STE_` code for exactly that.

It appears in **9 records out of 21,799, and its value is 0.0 in every single one.** Not a single recording in the dataset confidently affirms ST elevation.

The obvious fallback, `infarction_stadium1`, does not rescue it either. That field encodes the *healing stage* of an infarction (Stadium I, II, III, meaning roughly how long ago it happened), not whether ST elevation was present at the time. Timing and morphology are different axes. Dressing one up as the other would have produced class names that quietly lied about what the labels meant.

So the task was redefined as binary MI versus Normal, which the SCP codes genuinely support. Discovering that a labeling scheme is infeasible and proposing an honest alternative is the actual work of label engineering.

### 2. Building the MI code set from the codebook, not from memory

An early version hardcoded nine MI codes by hand. Pulling them from `scp_statements.csv` instead reveals **fourteen**, including five `INJ*` injury pattern codes (`INJAS`, `INJAL`, `INJIN`, `INJIL`, `INJLA`) that a hand written list silently misses. Any recording whose only MI finding was an injury code would have been mislabeled.

```python
mi_codes = set(scp[(scp['diagnostic'] == 1) & (scp['diagnostic_class'] == 'MI')].index)
```

Ask the codebook. Do not trust recall.

### 3. Comparison against published work is indicative, not head to head

The reference benchmark for PTB-XL is Strodthoff et al. 2021, which reports roughly 0.93 macro AUROC for the strongest single models (`resnet1d_wang` 0.930, `xresnet1d101` 0.928), all of which edge out LSTM based approaches (0.921 to 0.927).

But that benchmark is **multi label**: it keeps recordings carrying several concurrent superclasses and predicts each independently with per class sigmoids. This project is **single label**: multi label recordings were discarded so that softmax over mutually exclusive classes is valid, leaving 16,244 of 21,799 records.

Same dataset, same folds, same metric, different task difficulty. The comparison is worth making and worth caveating, and pretending the numbers line up exactly would be dishonest.

---

## Class distribution after single label filtering

| Class | Count |
|---|---|
| NORM | 9,069 |
| MI | 2,532 |
| STTC | 2,400 |
| CD | 1,708 |
| HYP | 535 |

Roughly 17:1 between the largest and smallest class. HYP is the one to watch: with only 535 records total, its test fold holds around 53 examples, which makes its AUROC the least stable number in the table. Class weighting via `compute_class_weight('balanced', ...)` gives HYP roughly six times the weight of a NORM example, which is what stops the model from ignoring it entirely.

---

## Architectures

**Signal encoder.** Three convolutional blocks (32 → 64 → 128 channels, kernels 7 → 5 → 3, padding `k//2` so only pooling changes length). Input is `(12, 1000)`, twelve leads at 100 Hz. Because the first convolution spans all twelve leads at once, a single filter can learn cross lead signatures from layer one, which matters because an infarction shows itself in whichever leads view the damaged wall.

**Sequence readers.** The CNN compresses the signal to a 250 step sequence of 128 dimensional features. Two readers were built on top of it:

- **BiLSTM**, hidden size 128, bidirectional, giving a 256 dimensional embedding from both directions' final hidden states. Bidirectional is defensible here because the full ten second recording is available offline, and ECG features depend on context from both sides.
- **Transformer encoder**, `d_model=128`, 8 attention heads, with a learnable positional embedding (attention is order blind, so position has to be injected) and mean pooling across time to a 128 dimensional embedding.

**Tabular encoder.** Age, sex, height, weight. Height is missing in roughly 68% of records and weight in 57%, handled with median imputation fit on the training split only. Kept deliberately small, because four features do not support a large network.

**Fusion.** Intermediate fusion: each encoder produces its own embedding, the two are concatenated, and a joint classifier head learns from both together. Trained two ways, from scratch and by transfer (chopping the heads off pretrained encoders with `nn.Identity()` and fine tuning at a lower learning rate).

---

## What the loss curves said

The Transformer fusion model overfits noticeably harder than the BiLSTM version. Training loss falls smoothly to about 0.15 while validation loss climbs past 2.5 by epoch 30. This is consistent with a Transformer having less built in inductive bias for sequential structure than an LSTM, so on a dataset of this size it memorizes rather than generalizes.

Best validation checkpointing is what saves the reported number. The evaluated model comes from around epoch 5 to 7, where validation loss was still near its minimum, not from the collapsed final epoch. This is the concrete payoff of holding out fold 9 and snapshotting on best validation loss rather than trusting the last epoch.

---

## Repository layout

```
week2.py    PTB-XL exploration: 12 lead visualization, superclass distribution, NORM versus MI
week3.py    Binary MI versus Normal: label engineering, 1D CNN, tabular MLP
week4.py    Five class multimodal: single label extraction, CNN+BiLSTM, four feature MLP, fusion
week5.py    Transformer encoder, benchmark comparison, Inception and residual backbone
```

Each file is a self contained `# %%` notebook that runs top to bottom. They hand off through files on disk rather than shared memory, so any one can be run alone once the ones before it have run once.

Signals are cached to `./cache/*.npy` after the first load. Reading 16,244 WFDB files takes several minutes; reading the cache takes a second.

---

## Methodology notes

**Patient safe splitting is not optional.** PTB-XL ships a `strat_fold` column assigning every recording to one of ten folds, with all of a given patient's recordings sharing a fold and each fold preserving the overall class balance. Folds 1 to 8 train, fold 9 validates, fold 10 tests. Splitting randomly instead would let one patient's recordings land in both train and test, which inflates metrics and is a recurring flaw in published ECG work.

**AUROC over accuracy.** At 17:1 imbalance, a model predicting NORM for everything scores well on accuracy and is useless. AUROC asks a question that survives imbalance: given one positive and one negative, does the model rank them correctly.

**Preprocessing statistics come from the training split only.** Age normalization, median imputation, and standardization are all fit on train and applied to validation and test. Fitting per split leaks distributional information and is a quiet form of the same problem the folds exist to prevent.

**Softmax and CrossEntropyLoss expect raw logits.** No softmax in `forward`. It is applied once, at evaluation, to turn logits into probabilities for the ROC computation.

---

## Environment

```bash
python -m venv ecg-env
source ecg-env/bin/activate
pip install torch numpy pandas matplotlib scikit-learn wfdb
```

Download PTB-XL from PhysioNet and place it at `./ptb-xl/` (the folder containing `ptbxl_database.csv`, `scp_statements.csv`, `records100/`, `records500/`). The data itself is gitignored.

Run order: `week3.py` for the binary task. For the five class task, `week4.py` first (it writes `ptbxl_5class.csv` and builds the signal cache), then `week5.py`.

---

## Where this is going

- Subgroup resolved analysis of when demographics actually improve MI detection, motivated by a contradiction between two studies out of the same Gothenburg group
- Deeper residual and Inception style backbones, since the benchmark shows those are what top this task
- Cross cohort generalization, with the Indus Hospital dataset as the target

---

## Related work

- Wagner et al. 2020, *PTB-XL, a large publicly available electrocardiography dataset* — the dataset, folds, and superclass taxonomy
- Strodthoff et al. 2021, *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL* — the reference benchmark
- Zhang et al. 2024, *Multi-label ECG classification based on multiscale features and transformer encoder* — 94.1% macro AUC on the same five superclasses
- DLTM-ECG 2022 — a Transformer that also fuses patient meta information, closest published relative to this project

---

## Acknowledgements

Supervised by **Dr. Zafar Iqbal**, whose guidance shaped both the direction of the work and the standard it was held to. Supported by the Habib University Office of Research.

The PTB-XL dataset is made openly available by the Physikalisch Technische Bundesanstalt through PhysioNet.
