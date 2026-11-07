"""One function per topic: these are the reference code shown on each page.

Every function takes the dataset (plus page settings) and returns
{message, metrics, tables, images} for the page and the PDF.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import charts
from .dataset import (
    DESCRIPTIONS,
    FEATURES,
    SEED,
    TARGET,
    ZERO_AS_MISSING,
    notebook_split,
    xy,
    zeros_to_nan,
)
from .models import (
    MODEL_NAMES,
    NEEDS_SCALING,
    cv_scores,
    estimator,
    fit_timed,
    folds,
    label_metrics,
    make_pipeline,
    oof_proba,
    svc_of,
    threshold_metrics,
)


# Helpers -----------------------------------------------------------------
def _py(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _table(title: str, frame: pd.DataFrame, index_name: str | None = None) -> dict:
    if index_name is not None:
        frame = frame.reset_index().rename(columns={"index": index_name})
    return {"title": title, "columns": [str(c) for c in frame.columns],
            "rows": [[_py(v) for v in row] for row in frame.itertuples(index=False)]}


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _two_features(df: pd.DataFrame, x: str, y: str):
    """Two features for a decision-boundary plot, without the impossible zeros (they squash the plot)."""
    if x == y:
        raise ValueError("Choose two different features for the decision boundary.")
    sub = df[[x, y, TARGET]]
    for c in (x, y):
        if c in ZERO_AS_MISSING:
            sub = sub[sub[c] != 0]
    return sub[[x, y]], sub[TARGET]


def _holdout(pipe, df: pd.DataFrame) -> dict:
    """Fit on the notebook's training split and score on its test split, plus 5-fold CV on all rows."""
    X_train, X_test, y_train, y_test = notebook_split(df)
    pipe.fit(X_train, y_train)
    X, y = xy(df)
    cv = cv_scores(pipe, X, y, scoring=("accuracy", "roc_auc"))
    return {
        "pipe": pipe,
        "train_acc": pipe.score(X_train, y_train),
        "test_acc": pipe.score(X_test, y_test),
        "test_auc": roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
        "cv_acc": cv["accuracy"].mean(),
        "cv_acc_sd": cv["accuracy"].std(),
        "cv_auc": cv["roc_auc"].mean(),
    }


def _score_metrics(s: dict) -> list[dict]:
    return [
        {"label": "Train accuracy", "value": _pct(s["train_acc"])},
        {"label": "Test accuracy", "value": _pct(s["test_acc"])},
        {"label": "Test ROC AUC", "value": f"{s['test_auc']:.3f}"},
        {"label": "5-fold CV accuracy", "value": f"{s['cv_acc']:.1%} ± {s['cv_acc_sd']:.1%}"},
        {"label": "5-fold CV ROC AUC", "value": f"{s['cv_auc']:.3f}"},
    ]


def _boundary(name: str, df: pd.DataFrame, x: str, y: str, title: str, **params):
    X2, y2 = _two_features(df, x, y)
    pipe = make_pipeline(name, **params).fit(X2, y2)
    return pipe, X2, y2, charts.to_png(charts.decision_boundary(pipe, X2, y2, title), f"Decision boundary on {x} and {y}")


# D.1 The dataset -------------------------------------------------------------
def dataset_overview(df: pd.DataFrame, column: str = "Glucose") -> dict:
    """Shape, class balance, impossible zeros and how one feature differs between the classes."""
    X, y = xy(df)
    zeros = (df[ZERO_AS_MISSING] == 0).sum().sort_values(ascending=False)
    rows_with_zero = (df[ZERO_AS_MISSING] == 0).any(axis=1).sum()
    by_class = df.groupby(TARGET)[FEATURES].mean().T.rename(columns={0: "mean if 0", 1: "mean if 1"})
    by_class["difference"] = by_class["mean if 1"] - by_class["mean if 0"]
    by_class["description"] = [DESCRIPTIONS[c] for c in by_class.index]
    return {
        "message": f"{len(df)} patients, {len(FEATURES)} features, {y.mean():.1%} diabetic. "
        f"df.isnull() finds no missing values, but {rows_with_zero} rows have an impossible 0 in "
        f"{', '.join(ZERO_AS_MISSING)}.",
        "metrics": [
            {"label": "Rows", "value": len(df)},
            {"label": "Features", "value": len(FEATURES)},
            {"label": "Diabetic (1)", "value": f"{int(y.sum())} ({y.mean():.1%})"},
            {"label": "Always-0 baseline accuracy", "value": _pct(1 - y.mean())},
            {"label": "Rows with an impossible 0", "value": int(rows_with_zero)},
            {"label": "Missing values (isnull)", "value": int(df.isnull().sum().sum())},
        ],
        "images": [
            charts.to_png(charts.class_balance(y), "Class balance"),
            charts.to_png(charts.zero_counts(zeros, len(df)), "Zeros that mean 'not measured'"),
            charts.to_png(charts.hist_by_class(df, column, TARGET), f"{column} by Outcome"),
        ],
        "tables": [
            _table("Feature means by class", by_class.round(3), "feature"),
            _table("df.describe()", X.describe().T.round(2), "feature"),
        ],
    }


# 10.2 Logistic regression ----------------------------------------------------
def logistic_regression(df: pd.DataFrame, C: float = 1.0, x: str = "Glucose", y: str = "BMI") -> dict:
    """Linear score -> sigmoid -> probability. Coefficients on standardised features are comparable."""
    s = _holdout(make_pipeline("logistic", C=C), df)
    model = s["pipe"].named_steps["model"]
    coefs = pd.Series(model.coef_[0], index=FEATURES)
    coef_table = pd.DataFrame({
        "coefficient (per 1 SD)": coefs,
        "odds ratio (per 1 SD)": np.exp(coefs),
        "effect": ["raises risk" if c > 0 else "lowers risk" for c in coefs],
    }).sort_values("coefficient (per 1 SD)", ascending=False)

    # One feature, unscaled, so the S-curve is in real units
    one = df[df[x] != 0] if x in ZERO_AS_MISSING else df
    lr1 = LogisticRegression(C=C, max_iter=1000).fit(one[[x]], one[TARGET])
    xs = np.linspace(one[x].min(), one[x].max(), 300)
    proba = lr1.predict_proba(pd.DataFrame({x: xs}))[:, 1]
    b0, b1 = lr1.intercept_[0], lr1.coef_[0, 0]
    x50 = -b0 / b1 if b1 != 0 and one[x].min() <= -b0 / b1 <= one[x].max() else None

    _, _, _, boundary = _boundary("logistic", df, x, y, f"Logistic regression (C={C})", C=C)
    top = coef_table.index[0]
    return {
        "message": f"Test accuracy {_pct(s['test_acc'])}, ROC AUC {s['test_auc']:.3f}. {top} has the largest "
        f"effect: one standard deviation more multiplies the odds of diabetes by "
        f"{coef_table.loc[top, 'odds ratio (per 1 SD)']:.2f}.",
        "metrics": _score_metrics(s) + [
            {"label": f"P = 0.5 at {x}", "value": "outside range" if x50 is None else f"{x50:.1f}"},
            {"label": "Intercept (log-odds)", "value": f"{model.intercept_[0]:.3f}"},
        ],
        "images": [
            charts.to_png(charts.sigmoid_curve(one[x], one[TARGET], xs, proba, x50), f"The sigmoid on {x}"),
            charts.to_png(charts.signed_bars(coefs, "Coefficients on standardised features", "log-odds per 1 SD"),
                          "Coefficients"),
            boundary,
        ],
        "tables": [_table("Coefficients and odds ratios", coef_table.round(3), "feature")],
    }


# 10.2 K-nearest neighbours -----------------------------------------------------
def k_nearest_neighbours(df: pd.DataFrame, k: int = 15, weights: str = "uniform", scale: bool = True,
                         x: str = "Glucose", y: str = "BMI") -> dict:
    """Predict the majority class of the k closest training rows. Distances need scaled features."""
    k = int(k)
    X, yy = xy(df)
    ks = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 75]
    curve = {"train": [], "cv": [], "sd": []}
    for kk in ks:
        r = cv_scores(make_pipeline("knn", scale=scale, n_neighbors=kk, weights=weights), X, yy, scoring=("accuracy",))
        curve["train"].append(r["train_accuracy"].mean())
        curve["cv"].append(r["accuracy"].mean())
        curve["sd"].append(r["accuracy"].std())
    best_k = ks[int(np.argmax(curve["cv"]))]
    s = _holdout(make_pipeline("knn", scale=scale, n_neighbors=k, weights=weights), df)
    other = cv_scores(make_pipeline("knn", scale=not scale, n_neighbors=k, weights=weights), X, yy,
                      scoring=("accuracy",))["accuracy"].mean()
    _, _, _, boundary = _boundary("knn", df, x, y, f"KNN (k={k}, {'scaled' if scale else 'unscaled'})",
                                  scale=scale, n_neighbors=k, weights=weights)
    table = pd.DataFrame({"k": ks, "train accuracy": curve["train"], "CV accuracy": curve["cv"],
                          "CV std": curve["sd"]})
    return {
        "message": f"With k={k}, CV accuracy is {_pct(s['cv_acc'])}; the best k on this grid is {best_k}. "
        f"Turning scaling {'off' if scale else 'on'} gives {_pct(other)}. Without scaling, the large-valued "
        f"columns (Insulin, Glucose) dominate the distance"
        + (", which happens to suit this data because Glucose is the strongest feature; that is luck, not a rule."
           if (other > s["cv_acc"]) == scale else "."),
        "metrics": _score_metrics(s) + [
            {"label": "Best k (CV)", "value": best_k},
            {"label": f"CV accuracy {'unscaled' if scale else 'scaled'}", "value": _pct(other)},
        ],
        "images": [
            charts.to_png(charts.score_curve(ks, {"train": (curve["train"], None),
                                                  "cross-validation": (curve["cv"], np.array(curve["sd"]))},
                                             "k (number of neighbours)", "accuracy",
                                             "Small k overfits, large k underfits", mark=k), "Accuracy against k"),
            boundary,
        ],
        "tables": [_table("Accuracy for each k", table.round(4))],
    }


# 10.3 Decision tree ------------------------------------------------------------
def decision_tree(df: pd.DataFrame, max_depth="4", criterion: str = "gini", min_samples_leaf: int = 1) -> dict:
    """Recursive yes/no splits that make each node as pure as possible."""
    depth = None if max_depth == "None" else int(max_depth)
    params = dict(max_depth=depth, criterion=criterion, min_samples_leaf=int(min_samples_leaf))
    s = _holdout(make_pipeline("tree", **params), df)
    tree = s["pipe"].named_steps["model"]
    X, y = xy(df)
    depths = list(range(1, 16))
    train, cv, sd = [], [], []
    for d in depths:
        r = cv_scores(make_pipeline("tree", **{**params, "max_depth": d}), X, y, scoring=("accuracy",))
        train.append(r["train_accuracy"].mean())
        cv.append(r["accuracy"].mean())
        sd.append(r["accuracy"].std())
    imp = pd.Series(tree.feature_importances_, index=FEATURES)
    t = tree.tree_
    root = FEATURES[t.feature[0]]
    p1 = y.mean()
    gini_root = 1 - p1 ** 2 - (1 - p1) ** 2
    return {
        "message": f"Depth {tree.get_depth()}, {tree.get_n_leaves()} leaves: train {_pct(s['train_acc'])}, "
        f"test {_pct(s['test_acc'])}. The first split is {root} ≤ {t.threshold[0]:.1f}. "
        f"Deeper trees fit the training data better but generalise worse.",
        "metrics": _score_metrics(s) + [
            {"label": "Depth / leaves", "value": f"{tree.get_depth()} / {tree.get_n_leaves()}"},
            {"label": "Root split", "value": f"{root} ≤ {t.threshold[0]:.1f}"},
            {"label": "Gini of the full data", "value": f"{gini_root:.3f}"},
            {"label": "Best depth (CV)", "value": depths[int(np.argmax(cv))]},
        ],
        "images": [
            charts.to_png(charts.tree_diagram(tree, FEATURES, max_depth=3), "The tree (first 3 levels)"),
            charts.to_png(charts.score_curve(depths, {"train": (train, None), "cross-validation": (cv, np.array(sd))},
                                             "max_depth", "accuracy", "Depth against accuracy",
                                             mark=depth), "Overfitting with depth"),
            charts.to_png(charts.importance_bars(imp, "Impurity-based feature importance"), "Feature importance"),
        ],
        "tables": [_table("Feature importance", imp.sort_values(ascending=False).round(4).to_frame("importance"),
                          "feature")],
    }


# 10.3 Naive Bayes --------------------------------------------------------------
def naive_bayes(df: pd.DataFrame, column: str = "Glucose", patient: int = 0) -> dict:
    """Bayes' rule with the 'naive' assumption that features are independent given the class."""
    s = _holdout(make_pipeline("naive_bayes"), df)
    nb = s["pipe"].named_steps["model"]
    X_train, X_test, y_train, _ = notebook_split(df)
    j = FEATURES.index(column)
    params = {c: (nb.theta_[c, j], nb.var_[c, j]) for c in (0, 1)}

    # Worked example: one test patient, feature by feature
    row = X_test.iloc[int(patient)]

    def log_gauss(v, mu, var):
        return -0.5 * np.log(2 * np.pi * var) - (v - mu) ** 2 / (2 * var)

    contrib = pd.DataFrame({
        "value": row.values,
        "log P(x|1) − log P(x|0)": [log_gauss(row[f], nb.theta_[1, i], nb.var_[1, i]) -
                                    log_gauss(row[f], nb.theta_[0, i], nb.var_[0, i])
                                    for i, f in enumerate(FEATURES)],
    }, index=FEATURES)
    prior = np.log(nb.class_prior_[1] / nb.class_prior_[0])
    log_odds = prior + contrib.iloc[:, 1].sum()
    posterior = 1 / (1 + np.exp(-log_odds))
    corr = X_train.corr().abs().where(~np.eye(len(FEATURES), dtype=bool)).stack().idxmax()
    return {
        "message": f"Test accuracy {_pct(s['test_acc'])}, ROC AUC {s['test_auc']:.3f}, with no tuning at all. "
        f"For test patient {int(patient)}: prior log-odds {prior:.2f} plus the evidence of each feature gives "
        f"P(diabetic) = {posterior:.1%}.",
        "metrics": _score_metrics(s) + [
            {"label": "Prior P(1)", "value": _pct(nb.class_prior_[1])},
            {"label": f"Patient {int(patient)}: P(diabetic)", "value": _pct(posterior)},
            {"label": "Most correlated pair", "value": f"{corr[0]} / {corr[1]}"},
        ],
        "images": [charts.to_png(charts.class_densities(df, column, TARGET, params), f"Fitted Gaussians for {column}"),
                   charts.to_png(charts.signed_bars(contrib.iloc[:, 1], f"Evidence for test patient {int(patient)}",
                                                    "log-likelihood ratio (positive = towards diabetic)"),
                                 "Evidence per feature")],
        "tables": [
            _table(f"Test patient {int(patient)}, feature by feature (prior log-odds {prior:.3f}, "
                   f"total {log_odds:.3f})", contrib.round(3), "feature"),
            _table("Per-class mean and standard deviation", pd.DataFrame(
                {"mean if 0": nb.theta_[0], "sd if 0": np.sqrt(nb.var_[0]),
                 "mean if 1": nb.theta_[1], "sd if 1": np.sqrt(nb.var_[1])}, index=FEATURES).round(2), "feature"),
        ],
    }


# 10.4 Support vector machine -----------------------------------------------------
def support_vector_machine(df: pd.DataFrame, kernel: str = "rbf", C: float = 1.0, gamma="scale",
                           x: str = "Glucose", y: str = "BMI") -> dict:
    """Find the boundary with the widest margin; kernels let that boundary curve."""
    params = dict(kernel=kernel, C=C, gamma=gamma)
    s = _holdout(make_pipeline("svm", **params), df)
    svc = svc_of(s["pipe"])
    X, yy = xy(df)
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    train, cv = [], []
    for c in Cs:
        r = cv_scores(make_pipeline("svm", **{**params, "C": c, "probability": False}), X, yy, scoring=("accuracy",))
        train.append(r["train_accuracy"].mean())
        cv.append(r["accuracy"].mean())
    X2, y2 = _two_features(df, x, y)
    pipe2 = make_pipeline("svm", **params).fit(X2, y2)
    sv = pipe2.named_steps["scale"].inverse_transform(svc_of(pipe2).support_vectors_)
    img = charts.to_png(charts.decision_boundary(pipe2, X2, y2, f"SVM ({kernel}, C={C}, gamma={gamma})", support=sv),
                        f"Decision boundary on {x} and {y}")
    return {
        "message": f"{svc.n_support_.sum()} of {len(notebook_split(df)[0])} training rows are support vectors "
        f"({svc.n_support_.sum() / len(notebook_split(df)[0]):.0%}): only they define the boundary. "
        f"Test accuracy {_pct(s['test_acc'])}. Small C allows more margin violations (smoother); large C fits harder.",
        "metrics": _score_metrics(s) + [
            {"label": "Support vectors", "value": f"{svc.n_support_.sum()} (class 0: {svc.n_support_[0]}, "
                                                  f"class 1: {svc.n_support_[1]})"},
            {"label": "Best C (CV)", "value": Cs[int(np.argmax(cv))]},
        ],
        "images": [img, charts.to_png(charts.score_curve(Cs, {"train": (train, None), "cross-validation": (cv, None)},
                                                         "C (log scale)", "accuracy", f"C against accuracy ({kernel})",
                                                         mark=C, log_x=True), "Regularisation C")],
        "tables": [_table("Accuracy for each C", pd.DataFrame({"C": Cs, "train accuracy": train,
                                                                "CV accuracy": cv}).round(4))],
    }


# 10.4 Random forest ------------------------------------------------------------
def random_forest(df: pd.DataFrame, max_features="sqrt", max_depth="None", min_samples_leaf: int = 3) -> dict:
    """Many decorrelated trees (bootstrap rows, random feature subsets) voting together."""
    params = dict(max_features=max_features if max_features in ("sqrt", "log2") else float(max_features),
                  max_depth=None if max_depth == "None" else int(max_depth), min_samples_leaf=int(min_samples_leaf))
    s = _holdout(make_pipeline("forest", **params), df)
    forest = s["pipe"].named_steps["model"]
    X_train, X_test, y_train, y_test = notebook_split(df)
    sizes = [1, 5, 10, 25, 50, 100, 200, 400]
    oob, test = [], []
    for n in sizes:
        rf = RandomForestClassifier(n_estimators=n, oob_score=n >= 25, bootstrap=True, random_state=SEED,
                                    **params).fit(X_train, y_train)
        oob.append(rf.oob_score_ if n >= 25 else np.nan)
        test.append(rf.score(X_test, y_test))
    single = make_pipeline("tree", max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"])
    single_acc = single.fit(X_train, y_train).score(X_test, y_test)
    imp = pd.Series(forest.feature_importances_, index=FEATURES)
    return {
        "message": f"200 trees reach {_pct(s['test_acc'])} test accuracy against {_pct(single_acc)} for one tree "
        f"with the same settings. Averaging many overfit, decorrelated trees cancels their errors.",
        "metrics": _score_metrics(s) + [
            {"label": "One tree, same settings (test)", "value": _pct(single_acc)},
            {"label": "OOB accuracy (400 trees)", "value": _pct(oob[-1])},
        ],
        "images": [
            charts.to_png(charts.score_curve(sizes, {"test": (test, None), "out-of-bag": (oob, None)},
                                             "number of trees (log scale)", "accuracy",
                                             "More trees: lower variance, then a plateau", log_x=True),
                          "Number of trees"),
            charts.to_png(charts.importance_bars(imp, "Mean decrease in impurity"), "Feature importance"),
        ],
        "tables": [
            _table("Accuracy by number of trees", pd.DataFrame({"trees": sizes, "test accuracy": test,
                                                                 "OOB accuracy": oob}).round(4)),
            _table("Feature importance", imp.sort_values(ascending=False).round(4).to_frame("importance"), "feature"),
        ],
    }


# 10.4 Gradient boosting ----------------------------------------------------------
def gradient_boosting(df: pd.DataFrame, learning_rate: float = 0.05, max_depth: int = 3) -> dict:
    """Shallow trees added one at a time, each fitting the errors of the ensemble so far."""
    params = dict(learning_rate=learning_rate, max_depth=int(max_depth), n_estimators=400)
    X_train, X_test, y_train, y_test = notebook_split(df)
    gb = estimator("boosting", **params).fit(X_train, y_train)
    stages = np.arange(1, 401)
    train = [np.mean(p == y_train) for p in gb.staged_predict(X_train)]
    test = [np.mean(p == y_test) for p in gb.staged_predict(X_test)]
    test_loss = [log_loss(y_test, p[:, 1]) for p in gb.staged_predict_proba(X_test)]
    best = int(np.argmin(test_loss)) + 1
    s = _holdout(make_pipeline("boosting", **{**params, "n_estimators": 100}), df)
    every = [1, 5, 10, 25, 50, 100, 150, 200, 300, 400]
    table = pd.DataFrame({"trees": every, "train accuracy": [train[i - 1] for i in every],
                          "test accuracy": [test[i - 1] for i in every],
                          "test log loss": [test_loss[i - 1] for i in every]})
    return {
        "message": f"Training accuracy keeps climbing to {_pct(train[-1])}, but test log loss is lowest after "
        f"{best} trees and rises after that. A smaller learning rate needs more trees but overfits more slowly.",
        "metrics": _score_metrics(s) + [
            {"label": "Best number of trees (test log loss)", "value": best},
            {"label": "Train accuracy at 400 trees", "value": _pct(train[-1])},
        ],
        "images": [
            charts.to_png(charts.score_curve(stages, {"train accuracy": (train, None), "test accuracy": (test, None)},
                                             "number of trees", "accuracy",
                                             f"Boosting stages (learning_rate={learning_rate}, depth={int(max_depth)})",
                                             mark=best), "Accuracy by stage"),
            charts.to_png(charts.score_curve(stages, {"test log loss": (test_loss, None)}, "number of trees",
                                             "log loss", "Test log loss: where boosting starts to overfit", mark=best),
                          "Log loss by stage"),
        ],
        "tables": [_table("Scores by number of trees (CV metrics above use 100 trees)", table.round(4))],
    }


# 10.5 Model comparison -----------------------------------------------------------
# The course's summary table (10.5). "Ensemble (XGBoost)" is gradient boosting here: scikit-learn's version.
CHEAT_SHEET = {
    "logistic": ("Linear boundary + probability", "No", "Fast", "High", "Baseline models, probability outputs"),
    "svm": ("Maximum-margin boundary", "Yes (kernels)", "Medium", "Medium", "Text, high-dimensional data"),
    "knn": ("Neighbours vote", "Yes", "Slow (large data)", "High", "Small datasets, pattern matching"),
    "naive_bayes": ("Probability rules", "Limited", "Very fast", "Medium", "Text classification"),
    "tree": ("If-else rules", "Yes", "Fast", "Very high", "Rule-based decisions"),
    "forest": ("Many trees voting", "Yes", "Medium", "Medium", "Strong tabular baseline"),
    "boosting": ("Combine many models (boosting)", "Yes", "Slow", "Low", "High-accuracy systems"),
}


def cheat_sheet(measured: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The course's model summary, with what each model actually did on this dataset next to it."""
    rows = {}
    for key, (idea, nonlinear, speed, interpret, use) in CHEAT_SHEET.items():
        r = measured.loc[MODEL_NAMES[key]]
        rows[MODEL_NAMES[key]] = {
            "core idea": idea, "non-linear data": nonlinear, "speed": speed, "interpretability": interpret,
            "best used for": use, f"here: CV {metric}": round(r[metric], 3),
            "here: fit (ms)": round(r["fit time (ms)"], 1), "here: predict + score (ms)": round(r["predict time (ms)"], 1),
        }
    return pd.DataFrame(rows).T


def model_comparison(df: pd.DataFrame, metric: str = "roc_auc", scaling: str = "auto",
                     x: str = "Glucose", y: str = "BMI") -> dict:
    """Every model, same folds, same metrics; plus a baseline that always predicts the majority class."""
    X, yy = xy(df)
    scoring = ("accuracy", "precision", "recall", "f1", "roc_auc")
    rows, per_fold, fitted = [], [], {}
    X2, y2 = _two_features(df, x, y)
    for name, label in {"dummy": "Baseline (always 0)", **MODEL_NAMES}.items():
        scale = None if scaling == "auto" else scaling == "always"
        pipe = make_pipeline(name, scale=scale)
        with warnings.catch_warnings():  # the baseline never predicts 1, so its precision is undefined (0)
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            r = cv_scores(pipe, X, yy, scoring=scoring)
        rows.append({"model": label, **{m: r[m].mean() for m in scoring}, "std": r[metric].std(),
                     "train": r[f"train_{metric}"].mean(), "validation": r[metric].mean(),
                     "fit time (ms)": r["fit_time"].mean() * 1000,
                     "predict time (ms)": r["score_time"].mean() * 1000})
        per_fold += [{"model": label, "score": v} for v in r[metric]]
        if name != "dummy":
            fitted[label] = make_pipeline(name, scale=scale).fit(X2, y2)
    table = pd.DataFrame(rows).set_index("model").sort_values(metric, ascending=False)
    best = table.index[0]
    return {
        "message": f"Best by {metric}: {best} ({table.loc[best, metric]:.3f}). The baseline gets "
        f"{_pct(table.loc['Baseline (always 0)', 'accuracy'])} accuracy with recall 0, which is why accuracy alone "
        f"is not enough. Differences smaller than the fold-to-fold spread are not meaningful.",
        "metrics": [
            {"label": f"Best ({metric})", "value": best},
            {"label": f"Its {metric}", "value": f"{table.loc[best, metric]:.3f} ± {table.loc[best, 'std']:.3f}"},
            {"label": "Baseline accuracy", "value": _pct(table.loc["Baseline (always 0)", "accuracy"])},
            {"label": "Scaling", "value": scaling},
        ],
        "images": [
            charts.to_png(charts.score_boxes(pd.DataFrame(per_fold), metric), f"{metric} per fold"),
            charts.to_png(charts.train_vs_test(table.drop("Baseline (always 0)")[["train", "validation"]], metric),
                          "Train vs validation"),
            charts.to_png(charts.boundary_grid(fitted, X2, y2), f"Decision boundaries on {x} and {y}"),
        ],
        "tables": [{**_table("Model cheat sheet: the course's summary (left) and what each model did on this "
                             "dataset (right, 5-fold CV)", cheat_sheet(table, metric), "model"), "top": True},
                   _table("Cross-validated scores (mean of 5 folds)",
                          table.drop(columns=["train", "validation"]).round(4), "model")],
    }


# 10.6 Evaluation metrics -------------------------------------------------------------
def _test_proba(df: pd.DataFrame, model: str):
    X_train, X_test, y_train, y_test = notebook_split(df)
    pipe = make_pipeline(model).fit(X_train, y_train)
    return y_test, pipe.predict_proba(X_test)[:, 1]


def confusion_and_metrics(df: pd.DataFrame, model: str = "logistic", threshold: float = 0.5) -> dict:
    """Confusion matrix on the test split and every metric derived from it."""
    y_test, proba = _test_proba(df, model)
    m = threshold_metrics(y_test, proba, threshold)
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    cm = np.array([[tn, fp], [fn, tp]])
    formulas = pd.DataFrame([
        ("Accuracy", "(TP + TN) / all", f"({tp} + {tn}) / {tp + tn + fp + fn}", m["accuracy"],
         "Share of all predictions that are right"),
        ("Precision", "TP / (TP + FP)", f"{tp} / ({tp} + {fp})", m["precision"],
         "Of those flagged diabetic, how many are"),
        ("Recall (sensitivity)", "TP / (TP + FN)", f"{tp} / ({tp} + {fn})", m["recall"],
         "Of the diabetics, how many we catch"),
        ("Specificity", "TN / (TN + FP)", f"{tn} / ({tn} + {fp})", m["specificity"],
         "Of the healthy, how many we clear"),
        ("F1-score", "2·P·R / (P + R)", "harmonic mean", m["f1"], "Balances precision and recall"),
        ("Balanced accuracy", "(recall + specificity) / 2", "", m["balanced_accuracy"], "Accuracy that ignores imbalance"),
        ("MCC", "correlation of truth and prediction", "", m["mcc"], "−1 to 1; 0 is random"),
    ], columns=["metric", "formula", "worked", "value", "meaning"]).set_index("metric")
    report = pd.DataFrame(classification_report(y_test, (proba >= threshold).astype(int), output_dict=True,
                                                zero_division=0)).T.drop(index="accuracy")
    base = threshold_metrics(y_test, np.zeros(len(y_test)), 0.5)
    return {
        "message": f"{MODEL_NAMES[model]} at threshold {threshold}: accuracy {_pct(m['accuracy'])}, precision "
        f"{_pct(m['precision'])}, recall {_pct(m['recall'])}. It misses {fn} of {tp + fn} diabetics (false "
        f"negatives). Predicting 'not diabetic' for everyone already scores {_pct(base['accuracy'])} accuracy.",
        "metrics": [{"label": k, "value": _pct(m[v])} for k, v in
                    [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                     ("Specificity", "specificity"), ("F1-score", "f1"), ("Balanced accuracy", "balanced_accuracy")]]
        + [{"label": "MCC", "value": f"{m['mcc']:.3f}"},
           {"label": "Always-0 accuracy", "value": _pct(base["accuracy"])}],
        "images": [charts.to_png(charts.confusion(cm, f"Confusion matrix (test, threshold {threshold})"),
                                 "Confusion matrix"),
                   charts.to_png(charts.proba_hist(y_test, proba, threshold), "Predicted probabilities")],
        "tables": [_table("Metrics from the confusion matrix", formulas.round(4), "metric"),
                   _table("classification_report()", report.round(3), "class")],
    }


def roc_analysis(df: pd.DataFrame, model: str = "logistic") -> dict:
    """ROC curve: recall against false positive rate over every threshold; AUC summarises it."""
    curves, aucs = {}, {}
    names = list(MODEL_NAMES) if model == "all" else [model]
    for name in names:
        y_test, proba = _test_proba(df, name)
        fpr, tpr, thr = roc_curve(y_test, proba)
        aucs[MODEL_NAMES[name]] = roc_auc_score(y_test, proba)
        curves[MODEL_NAMES[name]] = (fpr, tpr, aucs[MODEL_NAMES[name]])
    focus = "logistic" if model == "all" else model
    y_test, proba = _test_proba(df, focus)
    fpr, tpr, thr = roc_curve(y_test, proba)
    m = threshold_metrics(y_test, proba, 0.5)
    point = (1 - m["specificity"], m["recall"], "threshold 0.5")
    # AUC = chance a random diabetic gets a higher score than a random non-diabetic
    pos, neg = proba[y_test.values == 1], proba[y_test.values == 0]
    pairwise = (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
    youden = int(np.argmax(tpr - fpr))
    pick = np.unique(np.linspace(1, len(thr) - 1, 12).astype(int))
    thr_table = pd.DataFrame({"threshold": thr[pick], "false positive rate": fpr[pick], "true positive rate": tpr[pick]})
    return {
        "message": f"{MODEL_NAMES[focus]}: AUC {aucs[MODEL_NAMES[focus]]:.3f}. Check: a random diabetic gets a higher "
        f"score than a random non-diabetic in {pairwise:.1%} of pairs, which is the same number. The best "
        f"Youden threshold (TPR − FPR) is {thr[youden]:.2f}.",
        "metrics": [{"label": f"AUC {k}", "value": f"{v:.3f}"} for k, v in aucs.items()] + [
            {"label": "Pairwise ranking check", "value": f"{pairwise:.3f}"},
            {"label": "Youden threshold", "value": f"{thr[youden]:.2f} (TPR {tpr[youden]:.0%}, FPR {fpr[youden]:.0%})"},
        ],
        "images": [charts.to_png(charts.roc_curves(curves, point if model != "all" else None), "ROC curve")],
        "tables": [_table(f"Points on the {MODEL_NAMES[focus]} curve", thr_table.round(3)),
                   _table("AUC by model (test split)", pd.Series(aucs).sort_values(ascending=False).round(4)
                          .to_frame("ROC AUC"), "model")],
    }


def precision_recall_analysis(df: pd.DataFrame, model: str = "logistic") -> dict:
    """Precision against recall over every threshold; better than ROC when the positive class is rare."""
    curves, aps = {}, {}
    names = list(MODEL_NAMES) if model == "all" else [model]
    for name in names:
        y_test, proba = _test_proba(df, name)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        aps[MODEL_NAMES[name]] = average_precision_score(y_test, proba)
        curves[MODEL_NAMES[name]] = (rec, prec, aps[MODEL_NAMES[name]])
    focus = "logistic" if model == "all" else model
    y_test, proba = _test_proba(df, focus)
    prec, rec, thr = precision_recall_curve(y_test, proba)
    m = threshold_metrics(y_test, proba, 0.5)
    prevalence = y_test.mean()
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    targets = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    rows = []
    for t in targets:
        ok = np.where(rec[:-1] >= t)[0]
        i = ok[np.argmax(prec[ok])] if len(ok) else None
        rows.append({"recall at least": t, "best precision": None if i is None else prec[i],
                     "threshold": None if i is None else thr[i]})
    return {
        "message": f"{MODEL_NAMES[focus]}: average precision {aps[MODEL_NAMES[focus]]:.3f}, against {prevalence:.2f} "
        f"for random guessing (the share of diabetics). The highest F1 ({f1[best]:.3f}) is at threshold "
        f"{thr[best]:.2f}.",
        "metrics": [{"label": f"AP {k}", "value": f"{v:.3f}"} for k, v in aps.items()] + [
            {"label": "Random baseline", "value": f"{prevalence:.3f}"},
            {"label": "Best F1 threshold", "value": f"{thr[best]:.2f} (F1 {f1[best]:.3f})"},
        ],
        "images": [charts.to_png(charts.pr_curves(curves, prevalence,
                                                  (m["recall"], m["precision"], "threshold 0.5") if model != "all"
                                                  else None), "Precision–recall curve")],
        "tables": [_table(f"Precision you can keep at a required recall ({MODEL_NAMES[focus]})",
                          pd.DataFrame(rows).round(3)),
                   _table("Average precision by model (test split)", pd.Series(aps).sort_values(ascending=False)
                          .round(4).to_frame("average precision"), "model")],
    }


def threshold_selection(df: pd.DataFrame, model: str = "logistic", objective: str = "f1", fn_cost: float = 5.0,
                        target_recall: float = 0.8) -> dict:
    """Choose the threshold on out-of-fold training predictions, then check it once on the test split."""
    X_train, X_test, y_train, y_test = notebook_split(df)
    oof = oof_proba(make_pipeline(model), X_train, y_train)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    table = pd.DataFrame([threshold_metrics(y_train, oof, t) for t in grid], index=grid)
    table["cost"] = table["fn"] * fn_cost + table["fp"]
    table["youden"] = table["recall"] + table["specificity"] - 1
    if objective == "f1":
        chosen = table["f1"].idxmax()
    elif objective == "youden":
        chosen = table["youden"].idxmax()
    elif objective == "cost":
        chosen = table["cost"].idxmin()
    else:  # recall_target: the highest threshold that still reaches the target recall
        ok = table[table["recall"] >= target_recall]
        chosen = ok.index.max() if len(ok) else grid[0]
    proba = make_pipeline(model).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    at_05, at_t = threshold_metrics(y_test, proba, 0.5), threshold_metrics(y_test, proba, chosen)
    keys = ["accuracy", "precision", "recall", "specificity", "f1", "tp", "fp", "fn", "tn"]
    compare = pd.DataFrame({"threshold 0.5": [at_05[k] for k in keys], f"threshold {chosen:.2f}": [at_t[k] for k in keys]},
                           index=keys)
    compare.loc["cost (FN×{:g} + FP)".format(fn_cost)] = [at_05["fn"] * fn_cost + at_05["fp"],
                                                          at_t["fn"] * fn_cost + at_t["fp"]]
    desc = {"f1": "maximises F1", "youden": "maximises recall + specificity − 1",
            "cost": f"minimises {fn_cost:g}×FN + FP", "recall_target": f"keeps recall ≥ {target_recall:.0%}"}[objective]
    return {
        "message": f"Threshold {chosen:.2f} {desc} on the training folds. On the test split recall goes from "
        f"{_pct(at_05['recall'])} to {_pct(at_t['recall'])} and precision from {_pct(at_05['precision'])} to "
        f"{_pct(at_t['precision'])}. 0.5 is only a default, not a law.",
        "metrics": [
            {"label": "Chosen threshold", "value": f"{chosen:.2f}"},
            {"label": "Test recall (0.5 → chosen)", "value": f"{_pct(at_05['recall'])} → {_pct(at_t['recall'])}"},
            {"label": "Test precision (0.5 → chosen)", "value": f"{_pct(at_05['precision'])} → {_pct(at_t['precision'])}"},
            {"label": "Test F1 (0.5 → chosen)", "value": f"{_pct(at_05['f1'])} → {_pct(at_t['f1'])}"},
            {"label": "Missed diabetics (0.5 → chosen)", "value": f"{at_05['fn']} → {at_t['fn']}"},
        ],
        "images": [charts.to_png(charts.metrics_vs_threshold(table, chosen, "cost" if objective == "cost" else None),
                                 "Out-of-fold metrics on the training split"),
                   charts.to_png(charts.proba_hist(y_test, proba, chosen), "Test probabilities and the chosen threshold")],
        "tables": [_table("Test split: default against chosen threshold", compare.round(4), "metric")],
    }


def calibration_analysis(df: pd.DataFrame, model: str = "naive_bayes") -> dict:
    """Do predicted probabilities mean what they say? Compare raw, Platt (sigmoid) and isotonic calibration."""
    X, y = xy(df)
    variants = {
        f"{MODEL_NAMES[model]} (raw)": make_pipeline(model),
        "+ sigmoid (Platt)": CalibratedClassifierCV(make_pipeline(model), method="sigmoid", cv=5),
        "+ isotonic": CalibratedClassifierCV(make_pipeline(model), method="isotonic", cv=5),
    }
    curves, rows = {}, []
    for name, est in variants.items():
        p = oof_proba(est, X, y)
        frac, pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        curves[name] = (pred, frac)
        rows.append({"variant": name, "Brier score": brier_score_loss(y, p), "log loss": log_loss(y, p),
                     "ROC AUC": roc_auc_score(y, p), "mean predicted": p.mean(), "actual rate": y.mean()})
    table = pd.DataFrame(rows).set_index("variant")
    raw = table.iloc[0]
    best = table["Brier score"].idxmin()
    return {
        "message": f"Raw {MODEL_NAMES[model]}: Brier {raw['Brier score']:.3f}, log loss {raw['log loss']:.3f}. "
        f"Best calibrated: {best}. Calibration changes the probabilities, not the ranking, so ROC AUC barely moves.",
        "metrics": [{"label": f"Brier {k}", "value": f"{v:.4f}"} for k, v in table["Brier score"].items()]
        + [{"label": "Actual diabetic rate", "value": _pct(y.mean())}],
        "images": [charts.to_png(charts.calibration(curves), "Reliability diagram (out-of-fold, all rows)")],
        "tables": [_table("Probability quality (lower Brier and log loss are better)", table.round(4), "variant")],
    }


# 10.7 Hands-on ---------------------------------------------------------------------
def hands_on(df: pd.DataFrame, test_size: float = 0.2, stratify: bool = True, scale: bool = True,
             seed: int = 42) -> dict:
    """Notebook 10.7 end to end: split, scale, logistic regression, metrics, confusion matrix, ROC."""
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = notebook_split(df, test_size=test_size, stratify=stratify, seed=int(seed))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) if scale else X_train.values  # fit on train only
    X_test_scaled = scaler.transform(X_test) if scale else X_test.values

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    train_m, test_m = label_metrics(y_train, y_train_pred), label_metrics(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    auc_score = roc_auc_score(y_test, y_test_prob)

    metrics_table = pd.DataFrame({"train": train_m, "test": test_m})
    first = pd.DataFrame({"actual": y_test.values[:5], "predicted": y_test_pred[:5],
                          "P(diabetic)": y_test_prob[:5]}, index=y_test.index[:5])
    return {
        "message": f"Test accuracy {_pct(test_m['Accuracy'])}, precision {_pct(test_m['Precision'])}, recall "
        f"{_pct(test_m['Recall'])}, F1 {_pct(test_m['F1-score'])}, ROC AUC {auc_score:.3f}. Train accuracy is "
        f"{_pct(train_m['Accuracy'])}, so the test set is the honest number.",
        "metrics": [{"label": "X_train / X_test", "value": f"{X_train.shape} / {X_test.shape}"},
                    {"label": "Rows, features", "value": f"{X.shape}"}]
        + [{"label": f"Test {k}", "value": _pct(v)} for k, v in test_m.items()]
        + [{"label": "ROC AUC", "value": f"{auc_score:.3f}"}],
        "images": [charts.to_png(charts.confusion(cm, "Confusion Matrix (Test Data)"), "Confusion matrix"),
                   charts.to_png(charts.roc_curves({"Logistic regression": (fpr, tpr, auc_score)}), "ROC curve")],
        "tables": [_table("Train and test metrics (print_metrics)", metrics_table.round(4), "metric"),
                   _table("First test predictions", first.round(4), "row"),
                   _table("Coefficients", pd.Series(model.coef_[0], index=X.columns).round(4)
                          .to_frame("coefficient"), "feature")],
    }


# Beyond the course -----------------------------------------------------------------
ZERO_STRATEGIES = {
    "keep": "Keep the zeros (notebook)",
    "median": "Zeros → NaN → median",
    "indicator": "Median + missing indicator",
    "drop_rows": "Drop rows with any zero",
}


def zeros_as_missing(df: pd.DataFrame, model: str = "logistic") -> dict:
    """The notebook leaves the impossible zeros in. Compare treating them as missing, inside the CV pipeline."""
    rows = []
    for key, label in ZERO_STRATEGIES.items():
        data = df if key == "keep" else zeros_to_nan(df)
        if key == "drop_rows":
            data = data.dropna()
        X, y = xy(data)
        steps = []
        if key in ("median", "indicator"):
            steps.append(("impute", SimpleImputer(strategy="median", add_indicator=key == "indicator")))
        if model in NEEDS_SCALING:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", estimator(model)))
        r = cv_scores(Pipeline(steps), X, y, scoring=("accuracy", "recall", "f1", "roc_auc"))
        rows.append({"strategy": label, "rows": len(X), "accuracy": r["accuracy"].mean(), "recall": r["recall"].mean(),
                     "F1": r["f1"].mean(), "ROC AUC": r["roc_auc"].mean(), "AUC std": r["roc_auc"].std()})
    table = pd.DataFrame(rows).set_index("strategy")
    keep, med = table.iloc[0]["ROC AUC"], table.iloc[1]["ROC AUC"]
    return {
        "message": f"{MODEL_NAMES[model]}: ROC AUC {keep:.3f} keeping zeros, {med:.3f} with median imputation. "
        f"Dropping rows leaves only {int(table.iloc[3]['rows'])} patients, and its score is on a different, "
        f"easier subset, so it is not directly comparable.",
        "metrics": [{"label": k, "value": f"{v:.3f}"} for k, v in table["ROC AUC"].items()],
        "images": [charts.to_png(charts.importance_bars(table["ROC AUC"].rename(lambda s: s[:30]),
                                                        "Cross-validated ROC AUC by strategy", "ROC AUC",
                                                        errors=table["AUC std"].rename(lambda s: s[:30])),
                                 "ROC AUC by strategy")],
        "tables": [_table("5-fold CV scores; imputers are fitted inside each training fold", table.round(4), "strategy")],
    }


GRIDS = {
    "logistic": ("C", [0.001, 0.01, 0.1, 1, 10, 100], "l1_ratio", [0.0, 1.0]),  # 0 = L2 (ridge), 1 = L1 (lasso)
    "knn": ("n_neighbors", [3, 7, 11, 15, 21, 31, 45], "weights", ["uniform", "distance"]),
    "tree": ("max_depth", [2, 3, 4, 5, 6, 8, None], "min_samples_leaf", [1, 5, 10, 20]),
    "svm": ("C", [0.1, 0.3, 1, 3, 10], "gamma", [0.003, 0.01, 0.03, 0.1, 0.3]),
    "forest": ("max_depth", [3, 5, 8, None], "min_samples_leaf", [1, 3, 10]),
    "boosting": ("learning_rate", [0.01, 0.05, 0.1, 0.3], "n_estimators", [50, 100, 200]),
}


def hyperparameter_tuning(df: pd.DataFrame, model: str = "svm", scoring: str = "roc_auc") -> dict:
    """GridSearchCV on the training split only; the test split is touched once, at the end."""
    p1, v1, p2, v2 = GRIDS[model]
    X_train, X_test, y_train, y_test = notebook_split(df)
    extra = {"probability": False} if model == "svm" else {"n_estimators": 100} if model == "forest" else {}
    if model == "logistic":
        extra = {"solver": "liblinear"}
    pipe = make_pipeline(model, **extra)
    search = GridSearchCV(pipe, {f"model__{p1}": v1, f"model__{p2}": v2}, cv=folds(), scoring=scoring)
    search.fit(X_train, y_train)
    res = pd.DataFrame(search.cv_results_)
    heat = res.pivot_table(index=f"param_model__{p1}", columns=f"param_model__{p2}", values="mean_test_score",
                           dropna=False)
    heat.index = [str(i) for i in heat.index]
    heat.columns = [str(c) for c in heat.columns]
    heat.index.name, heat.columns.name = p1, p2

    def test_score(est):
        if scoring == "roc_auc":
            return roc_auc_score(y_test, est.predict_proba(X_test)[:, 1]) if hasattr(est, "predict_proba") \
                else roc_auc_score(y_test, est.decision_function(X_test))
        return est.score(X_test, y_test)

    default = make_pipeline(model, **extra).fit(X_train, y_train)
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    top = res.sort_values("rank_test_score").head(8)[[f"param_model__{p1}", f"param_model__{p2}", "mean_test_score",
                                                       "std_test_score"]]
    top.columns = [p1, p2, f"mean CV {scoring}", "std"]
    top = top.astype({p1: str, p2: str})
    return {
        "message": f"Best {MODEL_NAMES[model]} settings: {best}, CV {scoring} {search.best_score_:.3f}. On the "
        f"untouched test split: {test_score(search.best_estimator_):.3f} tuned against {test_score(default):.3f} "
        f"with defaults. The CV score of the winner is optimistic because it was picked as the maximum.",
        "metrics": [
            {"label": "Best settings", "value": ", ".join(f"{k}={v}" for k, v in best.items())},
            {"label": f"Best CV {scoring}", "value": f"{search.best_score_:.3f}"},
            {"label": f"Test {scoring}, tuned", "value": f"{test_score(search.best_estimator_):.3f}"},
            {"label": f"Test {scoring}, defaults", "value": f"{test_score(default):.3f}"},
            {"label": "Combinations × folds", "value": f"{len(res)} × 5 = {len(res) * 5} fits"},
        ],
        "images": [charts.to_png(charts.grid_heatmap(heat, f"Mean CV {scoring} for each combination"), "Grid search")],
        "tables": [_table("Top combinations", top.round(4))],
    }


COMPLEXITY = {
    "logistic": ("C", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100], True),
    "knn": ("n_neighbors", [1, 3, 5, 9, 15, 25, 41, 61, 101], False),
    "tree": ("max_depth", [1, 2, 3, 4, 5, 6, 8, 10, 14], False),
    "svm": ("C", [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100], True),
    "forest": ("min_samples_leaf", [1, 2, 3, 5, 10, 20, 40], False),
    "boosting": ("n_estimators", [10, 25, 50, 100, 200, 400], False),
    "naive_bayes": ("var_smoothing", [1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0], True),
}


def learning_and_validation_curves(df: pd.DataFrame, model: str = "tree") -> dict:
    """Learning curve: score against training-set size. Validation curve: score against model complexity."""
    X, y = xy(df)
    extra = {"probability": False} if model == "svm" else {"n_estimators": 100} if model == "forest" else {}
    pipe = make_pipeline(model, **extra)
    sizes, tr, va = learning_curve(pipe, X, y, cv=folds(), train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy")
    param, values, log_x = COMPLEXITY[model]
    vtr, vva = validation_curve(pipe, X, y, param_name=f"model__{param}", param_range=values, cv=folds(),
                                scoring="accuracy")
    gap = tr.mean(1)[-1] - va.mean(1)[-1]
    diagnosis = ("high variance (overfitting): more data or more regularisation should help" if gap > 0.08 else
                 "high bias (underfitting): a more flexible model or better features would help" if va.mean(1)[-1] < 0.74
                 else "a reasonable balance: train and validation scores are close")
    best = values[int(np.argmax(vva.mean(1)))]
    return {
        "message": f"{MODEL_NAMES[model]} with all data: train {_pct(tr.mean(1)[-1])}, validation "
        f"{_pct(va.mean(1)[-1])} (gap {gap:.1%}). That suggests {diagnosis}. Best {param} on the validation "
        f"curve: {best}.",
        "metrics": [
            {"label": "Train accuracy (all data)", "value": _pct(tr.mean(1)[-1])},
            {"label": "Validation accuracy", "value": _pct(va.mean(1)[-1])},
            {"label": "Gap", "value": f"{gap:.1%}"},
            {"label": f"Best {param}", "value": best},
        ],
        "images": [
            charts.to_png(charts.score_curve(sizes, {"train": (tr.mean(1), tr.std(1)),
                                                     "validation": (va.mean(1), va.std(1))},
                                             "training rows", "accuracy", "Learning curve"), "Learning curve"),
            charts.to_png(charts.score_curve([str(v) for v in values] if not log_x else values,
                                             {"train": (vtr.mean(1), vtr.std(1)), "validation": (vva.mean(1), vva.std(1))},
                                             param + (" (log scale)" if log_x else ""), "accuracy",
                                             f"Validation curve for {param}", log_x=log_x), "Validation curve"),
        ],
        "tables": [_table("Learning curve", pd.DataFrame({"training rows": sizes, "train": tr.mean(1),
                                                           "validation": va.mean(1), "validation std": va.std(1)}).round(4)),
                   _table(f"Validation curve ({param})", pd.DataFrame({param: [str(v) for v in values],
                                                                       "train": vtr.mean(1),
                                                                       "validation": vva.mean(1)}).round(4))],
    }


def feature_importance(df: pd.DataFrame, model: str = "forest") -> dict:
    """Permutation importance on the test split works for any model; compare it with the model's own view."""
    X_train, X_test, y_train, y_test = notebook_split(df)
    pipe, secs = fit_timed(make_pipeline(model), X_train, y_train)
    perm = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED)
    table = pd.DataFrame({"permutation (AUC drop)": perm.importances_mean, "std": perm.importances_std},
                         index=FEATURES)
    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        table["built-in (impurity)"] = est.feature_importances_
    elif hasattr(est, "coef_"):
        table["built-in (|coefficient|)"] = np.abs(est.coef_[0])
    table = table.sort_values("permutation (AUC drop)", ascending=False)
    top = table.index[0]
    weak = table[table["permutation (AUC drop)"] <= table["std"]].index.tolist()
    return {
        "message": f"Shuffling {top} lowers test ROC AUC by {table.loc[top, 'permutation (AUC drop)']:.3f}, the most of "
        f"any feature. " + (f"{', '.join(weak)} make no clear difference to this model." if weak else ""),
        "metrics": [{"label": "Most important", "value": top},
                    {"label": "Features with no clear effect", "value": len(weak)},
                    {"label": "Built-in importance", "value": "yes" if table.shape[1] > 2 else "none for this model"}],
        "images": [charts.to_png(charts.importance_bars(table["permutation (AUC drop)"], "Permutation importance (test split)",
                                                        "drop in ROC AUC when shuffled", errors=table["std"]),
                                 "Permutation importance")],
        "tables": [_table("Importance by feature", table.round(4), "feature")],
    }
