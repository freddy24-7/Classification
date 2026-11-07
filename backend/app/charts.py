"""All chart generation (matplotlib + seaborn) lives here.

Every function returns a figure; to_png() turns it into a base64 PNG that both the
web page and the PDF embed. Plotting code is kept close to what you would write in a notebook.
"""
import base64
import io
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
BLUE, RED, GREY, GREEN, ORANGE = "#2f5d8a", "#b83232", "#8a94a6", "#2e7d4f", "#c0762b"
CLASS_COLORS = {0: BLUE, 1: RED}
SERIES = [BLUE, RED, GREEN, ORANGE, "#6b4fa0", "#2a8c8c", "#8a5a44", GREY]


def to_png(fig, title: str) -> dict:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"title": title, "png": base64.b64encode(buf.getvalue()).decode()}


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# Dataset ---------------------------------------------------------------
def class_balance(y: pd.Series):
    fig, ax = plt.subplots(figsize=(4.2, 3))
    counts = y.value_counts().sort_index()
    ax.bar(["0 not diabetic", "1 diabetic"], counts.values, color=[BLUE, RED])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / len(y):.0%})", ha="center", va="bottom")
    ax.set(ylabel="rows", title="Outcome")
    _clean(ax)
    return fig


def zero_counts(zeros: pd.Series, n: int):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(zeros.index[::-1], zeros.values[::-1], color=ORANGE)
    for i, v in enumerate(zeros.values[::-1]):
        ax.text(v, i, f" {v} ({v / n:.0%})", va="center", fontsize=9)
    ax.set(xlabel="rows with value 0", title="Impossible zeros (hidden missing values)")
    _clean(ax)
    return fig


def hist_by_class(df: pd.DataFrame, column: str, target: str):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    sns.histplot(data=df, x=column, hue=target, kde=True, stat="density", common_norm=False,
                 palette=CLASS_COLORS, ax=ax)
    ax.set(title=f"{column} by Outcome")
    _clean(ax)
    return fig


# Decision boundaries ---------------------------------------------------
def decision_boundary(model, X2: pd.DataFrame, y: pd.Series, title: str, ax=None, support=None, steps: int = 200):
    """Colour the plane by predicted probability of class 1 for a model fitted on two features."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 4.6))
    a, b = X2.columns
    pad_a, pad_b = (X2[a].max() - X2[a].min()) * 0.05, (X2[b].max() - X2[b].min()) * 0.05
    xx, yy = np.meshgrid(np.linspace(X2[a].min() - pad_a, X2[a].max() + pad_a, steps),
                         np.linspace(X2[b].min() - pad_b, X2[b].max() + pad_b, steps))
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[a, b])
    zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.2)
    ax.scatter(X2[a], X2[b], c=y.map(CLASS_COLORS), s=12, edgecolor="white", linewidth=0.3)
    if support is not None:
        ax.scatter(support[:, 0], support[:, 1], s=40, facecolors="none", edgecolors="black", linewidth=0.6,
                   label="support vectors")
        ax.legend(loc="upper left", fontsize=8)
    ax.set(xlabel=a, ylabel=b, title=title)
    if own:
        fig.tight_layout()
        return fig
    return ax


def boundary_grid(fitted: dict, X2: pd.DataFrame, y: pd.Series):
    """One decision-boundary panel per model."""
    n = len(fitted)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), sharex=True, sharey=True)
    for ax, (name, model) in zip(axes.ravel(), fitted.items()):
        decision_boundary(model, X2, y, name, ax=ax, steps=120)
        ax.title.set_fontsize(10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# Model intuition -------------------------------------------------------
def sigmoid_curve(x: pd.Series, y: pd.Series, xs: np.ndarray, proba: np.ndarray, threshold_x: float | None):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    jitter = np.random.default_rng(0).uniform(-0.03, 0.03, len(y))
    ax.scatter(x, y + jitter, c=y.map(CLASS_COLORS), s=10, alpha=0.5)
    ax.plot(xs, proba, color="black", linewidth=2, label="P(diabetic | x)")
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=1)
    if threshold_x is not None:
        ax.axvline(threshold_x, color=GREY, linestyle=":", linewidth=1)
        ax.text(threshold_x, 0.52, f" p = 0.5 at {threshold_x:.0f}", fontsize=9)
    ax.set(xlabel=x.name, ylabel="probability / class", title=f"Logistic regression on {x.name} alone")
    ax.legend(loc="center right")
    _clean(ax)
    return fig


def signed_bars(values: pd.Series, title: str, xlabel: str, ref: float = 0.0):
    values = values.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(values) + 1.2))
    ax.barh(values.index, values.values - ref, left=ref, color=[RED if v > ref else BLUE for v in values.values])
    ax.axvline(ref, color="black", linewidth=0.8)
    ax.set(title=title, xlabel=xlabel)
    _clean(ax)
    return fig


def importance_bars(values: pd.Series, title: str, xlabel: str = "importance", errors: pd.Series | None = None):
    values = values.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(values) + 1.2))
    ax.barh(values.index, values.values, color=BLUE,
            xerr=None if errors is None else errors.reindex(values.index).values, ecolor=GREY)
    ax.set(title=title, xlabel=xlabel)
    _clean(ax)
    return fig


def score_curve(x, series: dict, xlabel: str, ylabel: str, title: str, mark=None, log_x: bool = False):
    """Line chart of one or more score series, e.g. train vs validation accuracy against k."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (label, (values, band)), color in zip(series.items(), SERIES):
        ax.plot(x, values, marker="o", markersize=3, label=label, color=color)
        if band is not None:
            ax.fill_between(x, np.asarray(values) - band, np.asarray(values) + band, color=color, alpha=0.15)
    if mark is not None:
        ax.axvline(mark, color=GREY, linestyle="--", linewidth=1)
    if log_x:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    _clean(ax)
    return fig


def tree_diagram(model, features: list[str], max_depth: int = 3):
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(model, feature_names=features, class_names=["0", "1"], filled=True, rounded=True,
              impurity=True, proportion=False, max_depth=max_depth, fontsize=8, ax=ax)
    return fig


def class_densities(df: pd.DataFrame, column: str, target: str, params: dict):
    """Histogram per class with the Gaussian that naive Bayes fits on top."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = np.linspace(df[column].min(), df[column].max(), 300)
    for cls, color in CLASS_COLORS.items():
        sub = df.loc[df[target] == cls, column]
        ax.hist(sub, bins=30, density=True, alpha=0.3, color=color)
        mu, var = params[cls]
        ax.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var), color=color, linewidth=2,
                label=f"class {cls}: N({mu:.1f}, {np.sqrt(var):.1f}²)")
    ax.set(xlabel=column, ylabel="density", title=f"P({column} | class) as naive Bayes sees it")
    ax.legend()
    _clean(ax)
    return fig


# Comparison ------------------------------------------------------------
def score_boxes(scores: pd.DataFrame, metric: str):
    order = scores.groupby("model")["score"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.4))
    with warnings.catch_warnings():  # seaborn 0.13 still passes matplotlib's deprecated vert= argument
        warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
        sns.boxplot(data=scores, y="model", x="score", order=order, color="#dbe5f0", ax=ax, fliersize=0)
    sns.stripplot(data=scores, y="model", x="score", order=order, color=BLUE, size=4, ax=ax)
    ax.set(xlabel=f"{metric} per fold", ylabel="", title=f"Cross-validated {metric} (5 folds)")
    _clean(ax)
    return fig


def train_vs_test(frame: pd.DataFrame, metric: str):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(frame))
    ax.bar(x - 0.2, frame["train"], 0.4, label="train", color=GREY)
    ax.bar(x + 0.2, frame["validation"], 0.4, label="validation", color=BLUE)
    ax.set_xticks(x, frame.index, rotation=25, ha="right")
    ax.set(ylabel=metric, title="Train vs validation: the gap is overfitting", ylim=(0, 1.05))
    ax.legend()
    _clean(ax)
    return fig


# Metrics ---------------------------------------------------------------
def confusion(cm: np.ndarray, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    labels = np.array([[f"TN\n{cm[0, 0]}", f"FP\n{cm[0, 1]}"], [f"FN\n{cm[1, 0]}", f"TP\n{cm[1, 1]}"]])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
    ax.set(xlabel="Predicted label", ylabel="Actual label", title=title)
    return fig


def roc_curves(curves: dict, point=None):
    """curves: name -> (fpr, tpr, auc). point: (fpr, tpr, label) to mark the chosen threshold."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for (name, (fpr, tpr, auc)), color in zip(curves.items(), SERIES):
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, label="random (AUC 0.5)")
    if point is not None:
        ax.scatter([point[0]], [point[1]], s=60, color="black", zorder=5, label=point[2])
    ax.set(xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (recall)", title="ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    _clean(ax)
    return fig


def pr_curves(curves: dict, prevalence: float, point=None):
    """curves: name -> (recall, precision, average precision)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for (name, (rec, prec, ap)), color in zip(curves.items(), SERIES):
        ax.plot(rec, prec, label=f"{name} (AP {ap:.3f})", color=color)
    ax.axhline(prevalence, linestyle="--", color=GREY, label=f"random (AP {prevalence:.2f})")
    if point is not None:
        ax.scatter([point[0]], [point[1]], s=60, color="black", zorder=5, label=point[2])
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve", ylim=(0, 1.05))
    ax.legend(loc="lower left", fontsize=8)
    _clean(ax)
    return fig


def metrics_vs_threshold(table: pd.DataFrame, chosen: float, extra: str | None = None):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for col, color in zip(["precision", "recall", "f1", "accuracy"], [GREEN, RED, BLUE, GREY]):
        ax.plot(table.index, table[col], label=col, color=color)
    if extra:
        ax2 = ax.twinx()
        ax2.plot(table.index, table[extra], color=ORANGE, linestyle="--", label=extra)
        ax2.set_ylabel(extra, color=ORANGE)
        ax2.grid(False)
    ax.axvline(chosen, color="black", linestyle=":", linewidth=1.2)
    ax.axvline(0.5, color=GREY, linestyle="--", linewidth=0.8)
    ax.set(xlabel="decision threshold", ylabel="score", title="Metrics across thresholds", ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    return fig


def proba_hist(y: pd.Series, proba: np.ndarray, threshold: float):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for cls, color in CLASS_COLORS.items():
        ax.hist(proba[np.asarray(y) == cls], bins=20, range=(0, 1), alpha=0.55, color=color, label=f"actual {cls}")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold {threshold:.2f}")
    ax.set(xlabel="predicted probability of diabetes", ylabel="patients", title="Where the threshold cuts")
    ax.legend()
    _clean(ax)
    return fig


def calibration(curves: dict):
    """curves: name -> (mean predicted, fraction positive, counts)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, label="perfectly calibrated")
    for (name, (pred, frac)), color in zip(curves.items(), SERIES):
        ax.plot(pred, frac, marker="o", label=name, color=color)
    ax.set(xlabel="mean predicted probability", ylabel="fraction actually diabetic", title="Reliability diagram",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=8)
    _clean(ax)
    return fig


def grid_heatmap(table: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(1.0 * table.shape[1] + 2.5, 0.5 * table.shape[0] + 1.6))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="Blues", ax=ax, cbar=False)
    ax.set(title=title)
    return fig


# Metrics overview ------------------------------------------------------
def confusion_grid(cms: dict, threshold: float):
    """One small confusion matrix per model, same colour scale, so the error types can be compared at a glance."""
    n = len(cms)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.8 * rows))
    vmax = max(cm.max() for cm in cms.values())
    for ax, (name, cm) in zip(axes.flat, cms.items()):
        labels = np.array([[f"TN\n{cm[0, 0]}", f"FP\n{cm[0, 1]}"], [f"FN\n{cm[1, 0]}", f"TP\n{cm[1, 1]}"]])
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax, vmin=0, vmax=vmax,
                    xticklabels=["pred 0", "pred 1"], yticklabels=["actual 0", "actual 1"])
        ax.set_title(name, fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"Out-of-fold confusion matrices, all 768 patients, threshold {threshold:.2f}", y=1.0)
    fig.tight_layout()
    return fig


def metric_vs_baseline(model: pd.Series, baseline: pd.Series, model_name: str):
    """Grouped bars: each metric for one model next to the always-'no' baseline."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(model))
    b1 = ax.bar(x - 0.2, model.values, 0.4, label=model_name, color=BLUE)
    b2 = ax.bar(x + 0.2, baseline.values, 0.4, label="baseline: always 'not diabetic'", color=GREY)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, model.index)
    ax.set(ylim=(0, 1.1), ylabel="score", title="The same predictions, five different numbers")
    ax.legend(loc="upper left", fontsize=8)
    _clean(ax)
    return fig
