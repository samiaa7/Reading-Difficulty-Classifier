# Reading Difficulty Classifier

Classifies text by reading difficulty **and** predicts the emotional/cognitive load
it places on a struggling reader using two ML models trained on 15
linguistic features.

## Models

| Model | Task | Algorithm | Library |
|---|---|---|---|
| Difficulty classifier | Easy / Moderate / Hard / Very Hard | Random Forest + GridSearchCV | scikit-learn |
| Emotional load regressor | Score 0–100 (Calm → Frustrated) | Feedforward Neural Net | PyTorch |

## 15 Linguistic Features

| Group | Features |
|---|---|
| Readability | Flesch score, FK grade, avg word length, avg sentence length, syllable density |
| Vocabulary | Rare word %, type-token ratio, long word % |
| Syntax | Noun ratio, verb ratio, adj ratio, passive voice count |
| Cognitive | Negation density, punctuation density, clause density |

## File structure

```
emotion_classifier/
├── data_pipeline.py     # Feature extraction, label generation, train/val/test split
├── train_sklearn.py     # Random Forest training + evaluation plots
├── train_pytorch.py     # PyTorch training loop + loss curves
├── app.py               # FastAPI serving both models
├── index.html           # Frontend with radar chart + gauge
```

## Output plots (saved to plots/)

- `confusion_matrix.png`     — RF confusion matrix (counts + normalised)
- `feature_importance.png`   — Gini importance of all 15 features
- `cv_scores.png`            — 5-fold cross-validation accuracy
- `loss_curve.png`           — PyTorch train/val MSE over epochs
- `prediction_scatter.png`   — True vs predicted emotional load
- `residuals.png`            — Residual analysis

## Dataset
https://github.com/louismartin/dress-data/tree/master/data-simplification/wikilarge
