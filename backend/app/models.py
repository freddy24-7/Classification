"""scikit-learn building blocks: the classifiers, pipelines, cross-validation and metrics.

All model code goes through this module, so every page builds and scores models the same way.
"""
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .dataset import SEED

MODEL_NAMES = {
    "logistic": "Logistic regression",
    "knn": "K-nearest neighbours",
    "tree": "Decision tree",
    "naive_bayes": "Gaussian naive Bayes",
    "svm": "Support vector machine",
    "forest": "Random forest",
    "boosting": "Gradient boosting",
}
# Distance- and margin-based models need scaled features; trees and naive Bayes do not
NEEDS_SCALING = {"logistic", "knn", "svm"}


def estimator(name: str, **params):
    """A classifier with the defaults used across the app; params override them."""
    if name == "logistic":
        return LogisticRegression(max_iter=1000, **params)
    if name == "knn":
        return KNeighborsClassifier(**{"n_neighbors": 15, **params})
    if name == "tree":
        return DecisionTreeClassifier(**{"max_depth": 4, "random_state": SEED, **params})
    if name == "naive_bayes":
        return GaussianNB(**params)
    if name == "svm":
        # SVC has no probabilities of its own (probability=True is deprecated in scikit-learn 1.9):
        # Platt scaling on 5 internal folds, then one SVC refitted on all the training rows
        probability = params.pop("probability", True)
        svc = SVC(**{"random_state": SEED, **params})
        return CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False) if probability else svc
    if name == "forest":
        return RandomForestClassifier(**{"n_estimators": 200, "min_samples_leaf": 3, "random_state": SEED,
                                         **params})
    if name == "boosting":
        return GradientBoostingClassifier(**{"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3,
                                             "random_state": SEED, **params})
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model '{name}'")


def make_pipeline(name: str, scale: bool | None = None, impute_zeros: bool = False, **params) -> Pipeline:
    """[optional median imputer] -> [optional StandardScaler] -> classifier.

    scale=None means: scale when the model needs it. impute_zeros expects the zeros already replaced by NaN.
    The scaler and imputer sit inside the pipeline, so cross-validation fits them on training folds only.
    """
    steps = []
    if impute_zeros:
        steps.append(("impute", SimpleImputer(strategy="median")))
    if scale if scale is not None else name in NEEDS_SCALING:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator(name, **params)))
    return Pipeline(steps)


def svc_of(pipe: Pipeline) -> SVC:
    """The fitted SVC inside a pipeline, whether or not it is wrapped for probabilities."""
    model = pipe.named_steps["model"]
    return model.calibrated_classifiers_[0].estimator if isinstance(model, CalibratedClassifierCV) else model


def folds(n: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED)


def cv_scores(pipe, X, y, scoring=("accuracy", "f1", "roc_auc"), n_folds: int = 5) -> dict:
    """Mean and per-fold cross-validated scores, plus fit time."""
    res = cross_validate(pipe, X, y, cv=folds(n_folds), scoring=list(scoring), return_train_score=True)
    out = {s: res[f"test_{s}"] for s in scoring}
    out.update({f"train_{s}": res[f"train_{s}"] for s in scoring})
    out["fit_time"] = res["fit_time"]
    out["score_time"] = res["score_time"]
    return out


def oof_proba(pipe, X, y, n_folds: int = 5) -> np.ndarray:
    """Out-of-fold probabilities: every row is predicted by a model that never saw it."""
    return cross_val_predict(pipe, X, y, cv=folds(n_folds), method="predict_proba")[:, 1]


def fit_timed(pipe, X, y):
    start = time.perf_counter()
    pipe.fit(X, y)
    return pipe, time.perf_counter() - start


def threshold_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Every confusion-matrix metric at one decision threshold."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def label_metrics(y_true, y_pred) -> dict:
    """The four metrics from the notebook's print_metrics()."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }


def auc(y_true, proba) -> float:
    return roc_auc_score(y_true, proba)
