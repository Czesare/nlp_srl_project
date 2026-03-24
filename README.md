# Challenging Semantic Role Labeling Models: A CheckList-Style Behavioral Evaluation

An end-to-end NLP project comparing a traditional feature-engineered approach against a fine-tuned transformer for Semantic Role Labeling (SRL), plus a behavioral evaluation of both models using the CheckList methodology ([Ribeiro et al., 2020](https://aclanthology.org/2020.acl-main.442.pdf)).

Dataset: [Universal Proposition Banks v1.0](https://github.com/UniversalPropositions/UP-1.0) English.  
Full write-up: [`challenging_srl_models_checklist_evaluation.pdf`](takehome/challenging_srl_models_checklist_evaluation.pdf)

---

## Project Structure

```
nlp_srl_project/
├── a1/          # Logistic Regression with hand-engineered features
├── a2/          # DistilBERT fine-tuned for token classification
└── takehome/    # CheckList-style behavioral evaluation + report
```

---

## Models & Results

### A1 — Logistic Regression

Three spaCy-based features: directed dependency path + predicate lemma, NER type + relative position, and POS tag + dependency relation. Vectorized with `DictVectorizer(sparse=True)`.

| Role        | F1   |
|-------------|------|
| ARG0        | 0.79 |
| ARG1        | 0.78 |
| ARG2        | 0.70 |
| ARGM-MOD    | 0.95 |
| ARGM-NEG    | 0.94 |
| ARGM-TMP    | 0.72 |
| Accuracy    | 0.96 |

### A2 — DistilBERT

`distilbert-base-uncased` fine-tuned with predicate markers (`[unused0]`, `[unused1]`) following [Shi & Lin (2019)](https://arxiv.org/abs/1904.05255). Trained for 6 epochs on Apple M3 Max (MPS).

| Role        | F1   |
|-------------|------|
| ARG0        | 0.89 |
| ARG1        | 0.90 |
| ARG2        | 0.84 |
| ARGM-MOD    | 0.97 |
| ARGM-NEG    | 0.97 |
| ARGM-TMP    | 0.85 |
| Weighted F1 | 0.98 |

ARGM-TMP is the sharpest differentiator — the dependency path alone is insufficient for temporal adjuncts, but DistilBERT's contextual representations handle them well.

---

## Behavioral Evaluation

The `takehome/` directory contains a CheckList-style challenge dataset testing 6 SRL capabilities:

| Capability | Test |
|---|---|
| CAP1 | Core argument identification |
| CAP2 | Voice alternation (active/passive) |
| CAP3 | PP attachment ambiguity |
| CAP4 | Rare goal arguments (ARGM-GOL) |
| CAP5 | Spray-load alternation |
| CAP6 | Entity substitution robustness |

Both models are evaluated by failure rate per capability. See the [report](takehome/challenging_srl_models_checklist_evaluation.pdf) for full analysis.

---

## Setup

**Pre-trained models** (too large for git): [Google Drive](https://drive.google.com/drive/folders/1cBWbba5ksZcSRVrCOjdPXHSC_GFuupNH?usp=sharing)

Place them at:
```
a1/model/model.joblib
a1/model/vectorizer.joblib
a2/srl-distilbert-model/
```

**Data:** [Universal PropBank v1.0](https://github.com/UniversalPropositions/UP-1.0) — place `.conllu` files in `data/`.

```bash
git clone https://github.com/Czesare/nlp_srl_project.git
cd nlp_srl_project
pip install -r a1/requirements.txt   # LR model
pip install -r a2/requirements.txt   # DistilBERT
pip install -r takehome/requirements.txt
```

---

## Tech Stack

spaCy · scikit-learn · HuggingFace Transformers · PyTorch