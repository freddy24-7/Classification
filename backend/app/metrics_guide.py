"""The metrics guide: accuracy, confusion matrix, precision, recall, F1 and ROC AUC on the diabetes data,
for every model, with a verdict on which metric to trust and which model suits the task.

Predictions are out-of-fold over all 768 patients (5-fold CV): every patient is predicted by a model that never
saw them, and with 268 diabetics the numbers are steadier than on the 54 of the test split.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from . import charts
from .analyses import _pct, _table
from .dataset import get_df, xy
from .models import MODEL_NAMES, make_pipeline, oof_proba, threshold_metrics

# What each model brings beyond its scores: the part of 'suitability' that numbers do not show
TRAITS = {
    "logistic": "Coefficients explain each risk factor and the probabilities are usually well calibrated: easy to "
                "defend in a clinical setting.",
    "svm": "No native probabilities (calibrated here) and hard to explain; its kernels bring little on 8 features.",
    "knn": "Needs scaling and is thrown off by the impossible zeros; every prediction searches the whole training set.",
    "naive_bayes": "Assumes the features are independent (glucose, BMI and insulin are not), so its probabilities "
                   "tend to be overconfident; extremely fast.",
    "tree": "Readable as if-else rules, but a single tree is unstable: a few changed patients can change its rules.",
    "forest": "Robust and needs no scaling; slower and harder to explain than one tree or a logistic regression.",
    "boosting": "Strong on tabular data but needs tuning to avoid overfitting; the hardest of these to explain.",
}
METRICS = ["accuracy", "precision", "recall", "f1", "ROC AUC"]


@lru_cache(maxsize=1)
def _oof_all() -> dict:
    """Out-of-fold probabilities for every model (fitted once, then reused for every threshold)."""
    X, y = xy(get_df())
    return {name: oof_proba(make_pipeline(name), X, y) for name in MODEL_NAMES}


def _best_precision_at_recall(y, proba, target=0.8) -> float:
    """The highest precision among all thresholds that still catch at least `target` of the diabetics."""
    precision, recall, _ = precision_recall_curve(y, proba)
    return float(precision[recall >= target].max())


def metric_concepts(m: dict, y: pd.Series, auc: float, name: str, threshold: float) -> pd.DataFrame:
    """Each metric explained with this model's counts, the trap it hides on this dataset, and a verdict."""
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    n, pos = len(y), int(y.sum())
    rows = [
        ("Confusion matrix", "Counts of the 4 outcomes", f"TP {tp}, FP {fp}, FN {fn}, TN {tn}",
         "—", f"Shows the two kinds of error in patients: {fn} diabetics missed, {fp} false alarms.",
         "Always look at it first: every metric below is built from these four numbers."),
        ("Accuracy", "(TP + TN) / all", f"({tp} + {tn}) / {n}", _pct(m["accuracy"]),
         f"Saying 'not diabetic' to everyone already scores {_pct(1 - pos / n)}, because 65% are healthy.",
         "Misleading on its own here. Report it, never choose on it."),
        ("Precision", "TP / (TP + FP)", f"{tp} / ({tp} + {fp})", _pct(m["precision"]),
         "Rises when the model flags fewer patients, so it can look good while many diabetics are missed.",
         "Secondary: it measures the cost of false alarms (extra tests)."),
        ("Recall (sensitivity)", "TP / (TP + FN)", f"{tp} / ({tp} + {fn})", _pct(m["recall"]),
         "Flagging everyone gives 100%, so it must be read together with precision.",
         "Primary for screening: a missed diabetic is the expensive mistake."),
        ("F1 score", "2·P·R / (P + R)", f"2·{m['precision']:.3f}·{m['recall']:.3f} / ({m['precision']:.3f} + "
         f"{m['recall']:.3f})", _pct(m["f1"]),
         "Weighs precision and recall equally; screening does not.",
         "A good single number when both errors cost about the same; secondary here."),
        ("ROC AUC", "P(score of a diabetic > score of a non-diabetic)", "area under the ROC curve", f"{auc:.3f}",
         f"Independent of the threshold, so it can be high while recall at {threshold:.2f} is poor.",
         "Best for comparing models; then choose the threshold for recall."),
    ]
    return pd.DataFrame(rows, columns=["metric", "formula", f"worked out ({MODEL_NAMES[name]})", "value",
                                       "trap on this dataset", "verdict for diabetes screening"]).set_index("metric")


def _suitability(table: pd.DataFrame, best_auc: float) -> pd.DataFrame:
    rows = []
    for name, r in table.iterrows():
        gap = best_auc - r["ROC AUC"]
        tier = ("Strong candidate" if gap <= 0.01 else "Usable" if gap <= 0.03 else "Weaker on this data")
        rows.append({
            "model": MODEL_NAMES[name], "verdict": tier,
            "evidence": f"AUC {r['ROC AUC']:.3f} ({gap:.3f} behind the best); catches {_pct(r['recall'])} of "
                        f"diabetics at this threshold; at 80% recall its precision is "
                        f"{_pct(r['precision @ 80% recall'])}; Brier {r['Brier']:.3f}.",
            "beyond the numbers": TRAITS[name],
        })
    return pd.DataFrame(rows).set_index("model")


def metrics_guide(df: pd.DataFrame, model: str = "logistic", threshold: float = 0.5) -> dict:
    """Every metric for every model on out-of-fold predictions, the concepts worked out, and a verdict per model."""
    X, y = xy(df)
    probas = _oof_all()
    rows, cms, curves = {}, {}, {}
    for name, p in probas.items():
        m = threshold_metrics(y, p, threshold)
        prec80 = _best_precision_at_recall(y, p)
        rows[name] = {"accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                      "ROC AUC": roc_auc_score(y, p), "precision @ 80% recall": prec80,
                      "Brier": brier_score_loss(y, p), "missed (FN)": m["fn"], "false alarms (FP)": m["fp"]}
        cms[MODEL_NAMES[name]] = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        fpr, tpr, _ = roc_curve(y, p)
        curves[MODEL_NAMES[name]] = (fpr, tpr, rows[name]["ROC AUC"])
    table = pd.DataFrame(rows).T.sort_values("ROC AUC", ascending=False)
    best_auc = table["ROC AUC"].max()

    mine = threshold_metrics(y, probas[model], threshold)
    auc = table.loc[model, "ROC AUC"]
    base = pd.Series({"accuracy": 1 - y.mean(), "precision": 0.0, "recall": 0.0, "f1": 0.0, "ROC AUC": 0.5})
    shown = table.loc[model, METRICS].astype(float)

    best_by = {k: MODEL_NAMES[table[k].idxmax()] for k in METRICS}
    heat = table[METRICS + ["precision @ 80% recall"]].astype(float).rename(index=MODEL_NAMES)
    full = table.astype({"missed (FN)": int, "false alarms (FP)": int}).rename(index=MODEL_NAMES)
    point = (mine["fp"] / (mine["fp"] + mine["tn"]), mine["recall"], f"{MODEL_NAMES[model]} at {threshold:.2f}")
    spread = table["ROC AUC"].max() - table["ROC AUC"].min()
    return {
        "message": f"{MODEL_NAMES[model]} at threshold {threshold:.2f}: accuracy {_pct(mine['accuracy'])}, precision "
        f"{_pct(mine['precision'])}, recall {_pct(mine['recall'])}, F1 {_pct(mine['f1'])}, ROC AUC {auc:.3f}. It misses "
        f"{mine['fn']} of {int(y.sum())} diabetics. The 'best' model changes with the metric: accuracy favours "
        f"{best_by['accuracy']}, recall {best_by['recall']}, ROC AUC {best_by['ROC AUC']}. The seven models span only "
        f"{spread:.3f} in ROC AUC, so the choice of metric and threshold matters more than the choice of model.",
        "metrics": [{"label": "Accuracy", "value": _pct(mine["accuracy"])},
                    {"label": "Precision", "value": _pct(mine["precision"])},
                    {"label": "Recall", "value": _pct(mine["recall"])},
                    {"label": "F1 score", "value": _pct(mine["f1"])},
                    {"label": "ROC AUC", "value": f"{auc:.3f}"},
                    {"label": "Missed diabetics", "value": f"{mine['fn']} of {int(y.sum())}"}],
        "images": [
            charts.to_png(charts.confusion(cms[MODEL_NAMES[model]], f"{MODEL_NAMES[model]}, threshold {threshold:.2f}"),
                          "Confusion matrix (out-of-fold, all 768 patients)"),
            charts.to_png(charts.metric_vs_baseline(shown, base, MODEL_NAMES[model]),
                          "Five metrics for the same predictions, against the always-'no' baseline"),
            charts.to_png(charts.grid_heatmap(heat.round(3), f"Every model on every metric (threshold {threshold:.2f})"),
                          "Models × metrics"),
            charts.to_png(charts.confusion_grid(cms, threshold), "Confusion matrices of all seven models"),
            charts.to_png(charts.roc_curves(curves, point), "ROC curves (out-of-fold)"),
        ],
        "tables": [
            {**_table(f"The concepts, worked out for {MODEL_NAMES[model]} at threshold {threshold:.2f}",
                      metric_concepts(mine, y, auc, model, threshold), "metric"), "top": True},
            _table("Suitability of each model for diabetes screening", _suitability(table, best_auc), "model"),
            _table("All models, all metrics (out-of-fold, 768 patients)", full.round(4), "model"),
        ],
    }
