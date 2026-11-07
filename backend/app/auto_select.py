"""Automated model selection with scikit-learn's own tools, set against the analyst walkthrough.

Two built-in tools do the mechanical part of the walkthrough in a few lines:
- GridSearchCV with the model itself as a search parameter compares every model family and its settings.
- TunedThresholdClassifierCV chooses the decision threshold by cross-validation.
"""
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import charts
from . import walkthrough
from .analyses import _pct, _table
from .dataset import get_df, notebook_split, zeros_to_nan
from .models import MODEL_NAMES, estimator, folds, threshold_metrics

# The same small grids as the walkthrough's tuning step, so the two pages search the same space
GRIDS = walkthrough.TUNING


def search_space() -> list[dict]:
    """One dict per model family: the model step is itself a parameter of the search."""
    space = []
    for name, grid in GRIDS.items():
        # The SVM is wrapped for probabilities, so its settings sit one level deeper
        prefix = "model__estimator__" if name == "svm" else "model__"
        space.append({"model": [estimator(name)], **{prefix + p: v for p, v in grid.items()}})
    return space


def precision_at_recall(recall_target: float):
    """Scorer for the threshold tuner: the best precision among thresholds that reach the recall target."""
    def score(y_true, y_pred):
        recall = recall_score(y_true, y_pred, zero_division=0)
        if recall < recall_target:
            return recall - 1.0  # below target: always worse than any threshold that meets it
        return precision_score(y_true, y_pred, zero_division=0)
    return make_scorer(score)


def automated_selection(df: pd.DataFrame, objective: str = "recall_target", recall_target: float = 0.8,
                        zeros: str = "median", seed: int = 42) -> dict:
    """GridSearchCV over all model families, then TunedThresholdClassifierCV, then one look at the test set."""
    return _cached(objective, recall_target, zeros, int(seed))


@lru_cache(maxsize=16)
def _cached(objective, recall_target, zeros, seed):
    """Cached per setting; the page and its PDF share one run."""
    df = get_df()
    X_train, X_test, y_train, y_test = notebook_split(df, seed=seed)
    if zeros == "median":
        X_train, X_test = zeros_to_nan(X_train), zeros_to_nan(X_test)

    # 1. One search over every model family and its settings, on the training split only
    steps = [("impute", SimpleImputer(strategy="median"))] if zeros == "median" else []
    pipe = Pipeline(steps + [("scale", StandardScaler()), ("model", estimator("logistic"))])
    start = time.perf_counter()
    search = GridSearchCV(pipe, search_space(), cv=folds(), scoring="roc_auc")
    search.fit(X_train, y_train)
    search_secs = time.perf_counter() - start

    # 2. Let the threshold be tuned by cross-validation as well
    scoring = precision_at_recall(recall_target) if objective == "recall_target" else objective
    tuned = TunedThresholdClassifierCV(search.best_estimator_, scoring=scoring, cv=folds(), store_cv_results=True)
    tuned.fit(X_train, y_train)

    # 3. Open the test set once
    proba = tuned.predict_proba(X_test)[:, 1]
    test = threshold_metrics(y_test, proba, tuned.best_threshold_)
    test_auc = roc_auc_score(y_test, proba)
    at05 = threshold_metrics(y_test, proba, 0.5)
    return _report(search, tuned, search_secs, test, test_auc, at05, objective, recall_target, zeros, seed)


def _family(est) -> str:
    key = {type(estimator(n)).__name__: n for n in MODEL_NAMES}[type(est).__name__]
    return MODEL_NAMES[key]


def _report(search, tuned, search_secs, test, test_auc, at05, objective, recall_target, zeros, seed) -> dict:
    res = pd.DataFrame(search.cv_results_)
    res["family"] = [_family(p["model"]) for p in res["params"]]
    res["settings"] = [", ".join(f"{k.split('__')[-1]}={v}" for k, v in p.items() if k != "model")
                       for p in res["params"]]
    best_rows = res.loc[res.groupby("family")["mean_test_score"].idxmax()].sort_values("mean_test_score",
                                                                                         ascending=False)
    best_family = best_rows.iloc[0]["family"]
    best_std = best_rows.iloc[0]["std_test_score"]
    leaderboard = pd.DataFrame({
        "best settings": best_rows["settings"].values,
        "CV ROC AUC": best_rows["mean_test_score"].values,
        "std": best_rows["std_test_score"].values,
        "behind the winner": best_rows.iloc[0]["mean_test_score"] - best_rows["mean_test_score"].values,
    }, index=best_rows["family"].values)
    leaderboard["within 1 SE"] = leaderboard["behind the winner"] <= best_std / np.sqrt(5)
    per_fold = [{"model": r["family"], "score": r[f"split{i}_test_score"]}
                for _, r in best_rows.iterrows() for i in range(5)]
    top = res.sort_values("rank_test_score").head(10)[["family", "settings", "mean_test_score", "std_test_score"]]
    top.columns = ["model", "settings", "CV ROC AUC", "std"]

    # The analyst walkthrough with the same split, zeros and recall target (cached)
    wt = walkthrough._cached(recall_target, zeros, "simplest_within_1se", seed)
    wt_steps = {s["title"]: s for s in wt["steps"]}
    wt_model = wt["metrics"][0]["value"]
    wt_threshold = float(wt["metrics"][1]["value"])
    wt_test = wt_steps["Evaluate once on the test set"]["tables"][0]
    wt_col = {row[0]: row[2] for row in wt_test["rows"]}
    wt_auc = float(wt["metrics"][4]["value"])

    goal = {"recall_target": f"best precision with recall ≥ {recall_target:.0%}", "f1": "best F1",
            "balanced_accuracy": "best balanced accuracy"}[objective]
    scores = tuned.cv_results_["scores"]
    shown_scores = np.where(np.asarray(scores) < 0, np.nan, scores) if objective == "recall_target" else scores
    compare = pd.DataFrame({
        "automated (scikit-learn)": [best_family, f"{tuned.best_threshold_:.2f}", f"{test_auc:.3f}",
                                     _pct(test["recall"]), _pct(test["precision"]), test["fn"], test["fp"]],
        "analyst walkthrough": [wt_model, f"{wt_threshold:.2f}", f"{wt_auc:.3f}", _pct(wt_col["recall"]),
                                _pct(wt_col["precision"]), int(wt_col["fn"]), int(wt_col["fp"])],
    }, index=["model", "threshold", "test ROC AUC", "test recall", "test precision", "missed diabetics",
              "false alarms"])
    same = best_family == wt_model
    agreement = (f"Both pick {best_family}." if same else
                 f"The search picks {best_family}; the walkthrough picks {wt_model}, because its rule prefers the "
                 f"simplest model within one standard error of the best. The search has no notion of 'simpler': it "
                 f"always takes the highest mean, even when the lead is within the noise.")
    fits = len(res) * 5 + 5 + 5
    return {
        "message": f"GridSearchCV tried {len(res)} combinations of {len(GRIDS)} model families ({len(res) * 5} fits, "
        f"{search_secs:.0f} s). Highest CV ROC AUC: {best_family} ({search.best_score_:.3f}). "
        f"TunedThresholdClassifierCV then set the threshold to {tuned.best_threshold_:.2f} for {goal}. On the test "
        f"set it catches {_pct(test['recall'])} of diabetics with precision {_pct(test['precision'])}. {agreement}",
        "metrics": [
            {"label": "Winner (CV ROC AUC)", "value": f"{best_family} ({search.best_score_:.3f})"},
            {"label": "Runner-up gap", "value": f"{leaderboard['behind the winner'].iloc[1]:.3f}"},
            {"label": "Families within 1 SE", "value": int(leaderboard["within 1 SE"].sum())},
            {"label": "Tuned threshold", "value": f"{tuned.best_threshold_:.2f}"},
            {"label": "Test recall (0.5 → tuned)", "value": f"{_pct(at05['recall'])} → {_pct(test['recall'])}"},
            {"label": "Test precision (0.5 → tuned)",
             "value": f"{_pct(at05['precision'])} → {_pct(test['precision'])}"},
            {"label": "Models fitted", "value": fits},
        ],
        "images": [
            charts.to_png(charts.score_boxes(pd.DataFrame(per_fold), "ROC AUC"),
                          "Best setting of each model family: ROC AUC per fold"),
            charts.to_png(charts.score_curve(tuned.cv_results_["thresholds"], {goal: (shown_scores, None)},
                                             "decision threshold", "cross-validated score",
                                             "TunedThresholdClassifierCV: score at each threshold",
                                             mark=tuned.best_threshold_),
                          "Threshold tuning (training split, 5 folds)"),
        ],
        "tables": [
            _table("Automated against the analyst walkthrough (same split and zeros; the walkthrough always "
                   f"targets recall ≥ {recall_target:.0%})", compare, "result"),
            _table("Leaderboard: best setting per model family (5-fold CV on the training split)",
                   leaderboard.round(4), "model"),
            _table("Top 10 combinations overall", top.round(4)),
        ],
    }
