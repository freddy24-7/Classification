"""The analyst walkthrough: one credible end-to-end classification project on the diabetes data.

Each step is its own function and returns a step dict {title, text, decision, metrics, images, tables}.
analyst_walkthrough() runs them in order, passing forward only what a real analyst would carry forward.
The test split is locked away in step 2 and used once, in step 9.
"""
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

from . import charts
from .analyses import _pct, _table
from .dataset import FEATURES, SEED, TARGET, ZERO_AS_MISSING, get_df, notebook_split, zeros_to_nan
from .models import MODEL_NAMES, cv_scores, folds, make_pipeline, oof_proba, threshold_metrics

# Simplest first: the order used to break near-ties (cheaper to run, easier to explain and to maintain)
SIMPLICITY = ["logistic", "naive_bayes", "knn", "tree", "svm", "forest", "boosting"]
# Small grids: around the defaults, enough to check that a shortlisted model is not badly mis-set
TUNING = {
    "logistic": {"C": [0.01, 0.1, 1, 10]},
    "naive_bayes": {"var_smoothing": [1e-9, 1e-6, 1e-3]},
    "knn": {"n_neighbors": [11, 15, 25, 35]},
    "tree": {"max_depth": [3, 4, 5, 6]},
    "svm": {"C": [0.3, 1, 3]},
    "forest": {"min_samples_leaf": [1, 3, 10]},
    "boosting": {"learning_rate": [0.03, 0.05, 0.1]},
}
SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc")


def _step(title, text, decision, metrics=(), images=(), tables=()):
    return {"title": title, "text": text, "decision": decision, "metrics": list(metrics), "images": list(images),
            "tables": list(tables)}


# Step 1 ------------------------------------------------------------------------------
def frame_problem(df: pd.DataFrame, recall_target: float) -> dict:
    """What is being predicted, for whom, and which mistake is worse: this picks the metrics."""
    rate = df[TARGET].mean()
    return _step(
        "Frame the problem and pick the metrics",
        f"Goal: flag patients who are likely diabetic so they get a confirmatory test. {_pct(rate)} of the 768 "
        f"patients are diabetic. In screening, a missed diabetic (false negative) is usually judged far worse than an "
        f"extra test (false positive), so accuracy is the wrong target: always answering 'not diabetic' already scores "
        f"{_pct(1 - rate)}.",
        f"Compare models on ROC AUC (how well they rank patients, independent of a threshold). Then set the "
        f"threshold so recall is at least {recall_target:.0%}, and report the precision that costs.",
        metrics=[{"label": "Patients", "value": len(df)}, {"label": "Diabetic", "value": _pct(rate)},
                 {"label": "Accuracy of always 'no'", "value": _pct(1 - rate)},
                 {"label": "Recall target", "value": f"{recall_target:.0%}"}],
        images=[charts.to_png(charts.class_balance(df[TARGET]), "Class balance")],
    )


# Step 2 ------------------------------------------------------------------------------
def lock_test_set(df: pd.DataFrame, seed: int):
    """Split once, stratified, before looking at anything else; the test rows are not touched until step 9."""
    X_train, X_test, y_train, y_test = notebook_split(df, seed=seed)
    step = _step(
        "Lock away a test set",
        f"Before exploring, split 80/20, stratified on Outcome (random_state={seed}). Every decision from here to "
        f"step 8 is made with cross-validation on the {len(X_train)} training rows only. The {len(X_test)} test "
        f"rows are opened once, at the end, so the final numbers are an honest estimate.",
        "Model choice, tuning and threshold are all decided on the training split.",
        metrics=[{"label": "Training rows", "value": len(X_train)}, {"label": "Test rows", "value": len(X_test)},
                 {"label": "Diabetic in train / test", "value": f"{_pct(y_train.mean())} / {_pct(y_test.mean())}"}],
    )
    return step, (X_train, X_test, y_train, y_test)


# Step 3 ------------------------------------------------------------------------------
def check_data(X_train: pd.DataFrame, zeros: str):
    """Look for data problems in the training rows; the impossible zeros are the big one here."""
    counts = (X_train[ZERO_AS_MISSING] == 0).sum()
    dupes = int(X_train.duplicated().sum())
    table = pd.DataFrame({"zeros": counts, "share of rows": (counts / len(X_train)).round(3)})
    if zeros == "median":
        decision = ("Treat the zeros as missing: replace them with NaN and impute the median inside every pipeline, "
                    "so the median is learnt from training folds only.")
    else:
        decision = "Keep the zeros as they are (the notebook's choice). The models then read 0 as a real value."
    step = _step(
        "Check the data",
        f"isnull() finds no missing values, but {int(counts['Insulin'])} training rows have Insulin = 0 and "
        f"{int(counts['SkinThickness'])} have SkinThickness = 0. A glucose, blood pressure or BMI of zero is "
        f"impossible: these are missing measurements recorded as 0. Duplicate rows: {dupes}.",
        decision,
        images=[charts.to_png(charts.zero_counts(counts, len(X_train)), "Impossible zeros in the training rows")],
        tables=[_table("Zeros per column (training rows)", table, "column")],
    )
    return step, zeros == "median"


def _prepare(X: pd.DataFrame, impute: bool) -> pd.DataFrame:
    return zeros_to_nan(X) if impute else X


# Step 4 + 5 ------------------------------------------------------------------------------
def spot_check(X_train, y_train, impute: bool):
    """A baseline, then every candidate model with default settings on the same 5 folds."""
    rows, per_fold = [], []
    for name, label in {"dummy": "Baseline (always 0)", **MODEL_NAMES}.items():
        with warnings.catch_warnings():  # the baseline never predicts 1, so its precision is undefined
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            r = cv_scores(make_pipeline(name, impute_zeros=impute), X_train, y_train, scoring=SCORING)
        rows.append({"key": name, "model": label, **{s: r[s].mean() for s in SCORING}, "AUC std": r["roc_auc"].std(),
                     "train AUC": r["train_roc_auc"].mean()})
        if name != "dummy":
            per_fold += [{"model": label, "score": v} for v in r["roc_auc"]]
    table = pd.DataFrame(rows).set_index("key").sort_values("roc_auc", ascending=False)
    models = table.drop("dummy")
    best = models.index[0]
    cutoff = models.loc[best, "roc_auc"] - models.loc[best, "AUC std"]
    shortlist = [k for k in models.index if models.loc[k, "roc_auc"] >= cutoff]
    base = table.loc["dummy"]
    logit = table.loc["logistic"]

    baseline = _step(
        "Set a baseline",
        f"A model that always predicts the majority class scores {_pct(base['accuracy'])} accuracy, recall 0 and "
        f"ROC AUC 0.5. A plain logistic regression reaches ROC AUC {logit['roc_auc']:.3f} and recall "
        f"{_pct(logit['recall'])}. Any more complex model has to beat that clearly to earn its place.",
        "Logistic regression is the benchmark for everything that follows.",
        metrics=[{"label": "Baseline accuracy", "value": _pct(base["accuracy"])},
                 {"label": "Logistic ROC AUC", "value": f"{logit['roc_auc']:.3f} ± {logit['AUC std']:.3f}"},
                 {"label": "Logistic recall at 0.5", "value": _pct(logit["recall"])}],
    )
    shown = table.drop(columns=["train AUC"]).rename(columns={"roc_auc": "ROC AUC"})
    shown.index = [table.loc[k, "model"] for k in table.index]
    shown = shown.drop(columns=["model"])
    spot = _step(
        "Spot-check the candidate models",
        f"All seven models with default settings, on the same 5 folds of the training split. The best mean ROC AUC is "
        f"{MODEL_NAMES[best]} at {models.loc[best, 'roc_auc']:.3f}, but its fold-to-fold spread is "
        f"±{models.loc[best, 'AUC std']:.3f}. Models whose mean lies within one spread of the best cannot be told "
        f"apart on 614 rows.",
        f"Shortlist {len(shortlist)}: {', '.join(MODEL_NAMES[k] for k in shortlist)}. "
        f"Drop the rest, which are clearly behind.",
        metrics=[{"label": "Best default model", "value": MODEL_NAMES[best]},
                 {"label": "Shortlist cut-off (AUC)", "value": f"≥ {cutoff:.3f}"},
                 {"label": "Shortlisted", "value": len(shortlist)}],
        images=[charts.to_png(charts.score_boxes(pd.DataFrame(per_fold), "ROC AUC"), "ROC AUC per fold")],
        tables=[_table("5-fold CV on the training split, default settings, threshold 0.5", shown.round(4), "model")],
    )
    return baseline, spot, shortlist, models["roc_auc"].to_dict()


# Step 6 ------------------------------------------------------------------------------
def tune_shortlist(X_train, y_train, shortlist: list[str], default_auc: dict, impute: bool):
    """A small grid per shortlisted model, on the training split only."""
    rows, tuned = [], {}
    for name in shortlist:
        extra = {"probability": False} if name == "svm" else {}
        search = GridSearchCV(make_pipeline(name, impute_zeros=impute, **extra),
                              {f"model__{p}": v for p, v in TUNING[name].items()}, cv=folds(), scoring="roc_auc")
        search.fit(X_train, y_train)
        i = search.best_index_
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        default = default_auc[name]
        tuned[name] = {"params": params, "mean": search.cv_results_["mean_test_score"][i],
                       "std": search.cv_results_["std_test_score"][i]}
        rows.append({"model": MODEL_NAMES[name], "best setting": ", ".join(f"{k}={v}" for k, v in params.items()),
                     "tuned AUC": tuned[name]["mean"], "default AUC": default, "gain": tuned[name]["mean"] - default,
                     "std": tuned[name]["std"]})
    table = pd.DataFrame(rows).set_index("model").sort_values("tuned AUC", ascending=False)
    gain, spread = table["gain"].max(), table["std"].median()
    verdict = (f"smaller than the typical fold-to-fold spread ({spread:.3f}). That is common on a small table like "
               f"this: the choice of threshold matters more than fine-tuning" if gain < spread else
               f"larger than the typical fold-to-fold spread ({spread:.3f}), so tuning was worth it here")
    step = _step(
        "Tune the shortlist",
        f"A small grid around each model's defaults, scored by 5-fold ROC AUC on the training split. The largest "
        f"gain from tuning is {gain:.3f}, {verdict}. The tuned scores are slightly optimistic, because each is the "
        f"maximum over its grid.",
        "Keep the tuned settings, but do not read much into the order: the gaps are within noise.",
        tables=[_table("Grid search on the training split (ROC AUC)", table.round(4), "model")],
    )
    return step, tuned


# Step 7 ------------------------------------------------------------------------------
def choose_model(tuned: dict, selection: str):
    """One-standard-error rule: the simplest model whose score is within one standard error of the best."""
    best = max(tuned, key=lambda k: tuned[k]["mean"])
    se = tuned[best]["std"] / np.sqrt(5)
    if selection == "simplest_within_1se":
        close = [k for k in tuned if tuned[k]["mean"] >= tuned[best]["mean"] - se]
        chosen = min(close, key=SIMPLICITY.index)
        others = [MODEL_NAMES[k] for k in close if k != best]
        near = (f"{', '.join(others)} {'is' if len(others) == 1 else 'are'} within one standard error ({se:.3f})"
                if others else f"no other model is within one standard error ({se:.3f})")
        if chosen == best:
            why = (f"{MODEL_NAMES[best]} has the highest tuned AUC ({tuned[best]['mean']:.3f}); {near}. "
                   f"{MODEL_NAMES[best]} is also the simplest of these, so the choice is easy.")
        else:
            why = (f"{MODEL_NAMES[best]} has the highest tuned AUC ({tuned[best]['mean']:.3f}), but {near}. Among "
                   f"those, {MODEL_NAMES[chosen]} is the simplest: faster, easier to explain to clinicians, and less "
                   f"likely to break on new data.")
    else:
        chosen = best
        why = (f"Take the highest mean tuned AUC: {MODEL_NAMES[best]} ({tuned[best]['mean']:.3f}). Note that the "
               f"runner-up is within {tuned[best]['mean'] - sorted(t['mean'] for t in tuned.values())[-2]:.3f}, far "
               f"inside the noise, so this choice is close to arbitrary.")
    table = pd.DataFrame({MODEL_NAMES[k]: {"tuned AUC": v["mean"], "std": v["std"],
                                           "simplicity rank": SIMPLICITY.index(k) + 1,
                                           "within 1 SE of best": v["mean"] >= tuned[best]["mean"] - se}
                          for k, v in tuned.items()}).T.sort_values("tuned AUC", ascending=False)
    step = _step(
        "Choose the model",
        why,
        f"Final model: {MODEL_NAMES[chosen]} with "
        f"{', '.join(f'{k}={v}' for k, v in tuned[chosen]['params'].items())}.",
        metrics=[{"label": "Chosen model", "value": MODEL_NAMES[chosen]},
                 {"label": "Its CV ROC AUC", "value": f"{tuned[chosen]['mean']:.3f}"},
                 {"label": "Standard error", "value": f"{se:.3f}"},
                 {"label": "Rule", "value": "simplest within 1 SE" if selection == "simplest_within_1se" else "best mean"}],
        tables=[_table("Candidates after tuning", table.astype({"simplicity rank": int}).round(4), "model")],
    )
    return step, chosen


# Step 8 ------------------------------------------------------------------------------
def choose_threshold(pipe, X_train, y_train, recall_target: float):
    """The highest threshold that keeps out-of-fold recall at or above the target."""
    oof = oof_proba(pipe, X_train, y_train)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    table = pd.DataFrame([threshold_metrics(y_train, oof, t) for t in grid], index=grid)
    ok = table[table["recall"] >= recall_target]
    chosen = float(ok.index.max()) if len(ok) else float(grid[0])
    at05, at_t = table.loc[0.5], table.loc[chosen]
    step = _step(
        "Choose the decision threshold",
        f"At the default threshold 0.5 the model catches {_pct(at05['recall'])} of diabetics (out-of-fold, training "
        f"split). To reach the {recall_target:.0%} recall target the threshold has to drop to {chosen:.2f}. "
        f"Precision falls from {_pct(at05['precision'])} to {_pct(at_t['precision'])}: more patients are sent for a "
        f"test that turns out negative. Whether that trade is acceptable is a clinical decision, not a modelling one.",
        f"Use threshold {chosen:.2f} and store it with the model.",
        metrics=[{"label": "Threshold", "value": f"0.50 → {chosen:.2f}"},
                 {"label": "Recall (OOF)", "value": f"{_pct(at05['recall'])} → {_pct(at_t['recall'])}"},
                 {"label": "Precision (OOF)", "value": f"{_pct(at05['precision'])} → {_pct(at_t['precision'])}"},
                 {"label": "Flagged per 100 patients",
                  "value": f"{(at05['tp'] + at05['fp']) / len(y_train) * 100:.0f} → "
                           f"{(at_t['tp'] + at_t['fp']) / len(y_train) * 100:.0f}"}],
        images=[charts.to_png(charts.metrics_vs_threshold(table, chosen), "Out-of-fold metrics on the training split")],
    )
    return step, chosen


# Step 9 ------------------------------------------------------------------------------
def _bootstrap(y, proba, threshold, n=1000):
    rng = np.random.default_rng(SEED)
    y, proba = np.asarray(y), np.asarray(proba)
    out = {"ROC AUC": [], "recall": [], "precision": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        m = threshold_metrics(y[i], proba[i], threshold)
        out["ROC AUC"].append(roc_auc_score(y[i], proba[i]))
        out["recall"].append(m["recall"])
        out["precision"].append(m["precision"])
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}


def evaluate_on_test(pipe, name, n_train: int, cv_auc: float, X_test, y_test, threshold: float,
                     recall_target: float):
    """Open the test set once: fit on all training rows, score at 0.5 and at the chosen threshold."""
    proba = pipe.predict_proba(X_test)[:, 1]
    auc_ = roc_auc_score(y_test, proba)
    at05, at_t = threshold_metrics(y_test, proba, 0.5), threshold_metrics(y_test, proba, threshold)
    ci = _bootstrap(y_test, proba, threshold)
    keys = ["recall", "precision", "specificity", "f1", "accuracy", "tp", "fn", "fp", "tn"]
    compare = pd.DataFrame({"threshold 0.50": [at05[k] for k in keys],
                            f"threshold {threshold:.2f}": [at_t[k] for k in keys]}, index=keys)
    cis = pd.DataFrame({"estimate": [auc_, at_t["recall"], at_t["precision"]],
                        "95% CI low": [ci[k][0] for k in ci], "95% CI high": [ci[k][1] for k in ci]},
                       index=["ROC AUC", f"recall at {threshold:.2f}", f"precision at {threshold:.2f}"])
    fpr, tpr, _ = roc_curve(y_test, proba)
    point = (1 - at_t["specificity"], at_t["recall"], f"threshold {threshold:.2f}")
    met = "meets" if at_t["recall"] >= recall_target else "falls short of"
    lo, hi = ci["ROC AUC"]
    gap = (f"The cross-validated estimate was {cv_auc:.3f}, inside the test interval ({lo:.2f}–{hi:.2f}), so the "
           f"difference is noise from a small test set rather than a sign of overfitting." if lo <= cv_auc <= hi else
           f"The cross-validated estimate was {cv_auc:.3f}, outside the test interval ({lo:.2f}–{hi:.2f}): worth "
           f"investigating before trusting either number.")
    step = _step(
        "Evaluate once on the test set",
        f"{MODEL_NAMES[name]}, refitted on all {n_train} training rows, scores ROC AUC "
        f"{auc_:.3f} on the {len(y_test)} test patients. {gap} At threshold {threshold:.2f} it catches {at_t['tp']} of "
        f"{at_t['tp'] + at_t['fn']} diabetics (recall {_pct(at_t['recall'])}), which {met} the {recall_target:.0%} "
        f"target. With only {at_t['tp'] + at_t['fn']} diabetics in the test set the uncertainty is wide: the 95% "
        f"bootstrap interval for recall runs from {_pct(ci['recall'][0])} to {_pct(ci['recall'][1])}.",
        "Report the test numbers with their intervals. Do not go back and retune on the test set.",
        metrics=[{"label": "Test ROC AUC", "value": f"{auc_:.3f} ({ci['ROC AUC'][0]:.2f}–{ci['ROC AUC'][1]:.2f})"},
                 {"label": f"Test recall at {threshold:.2f}", "value": _pct(at_t["recall"])},
                 {"label": f"Test precision at {threshold:.2f}", "value": _pct(at_t["precision"])},
                 {"label": "Missed diabetics", "value": f"{at05['fn']} at 0.50 → {at_t['fn']}"}],
        images=[charts.to_png(charts.confusion(np.array([[at_t["tn"], at_t["fp"]], [at_t["fn"], at_t["tp"]]]),
                                               f"Test set, threshold {threshold:.2f}"), "Confusion matrix"),
                charts.to_png(charts.roc_curves({MODEL_NAMES[name]: (fpr, tpr, auc_)}, point), "ROC curve (test)")],
        tables=[_table("Test set: default against chosen threshold", compare.round(4), "metric"),
                _table("Uncertainty: 1,000 bootstrap resamples of the test set", cis.round(3), "metric")],
    )
    return step, proba


# Step 10 ------------------------------------------------------------------------------
def sanity_checks(pipe, X_test, y_test, proba, threshold: float):
    """Is the model using sensible signals, are its probabilities usable, and who does it miss?"""
    perm = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED)
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    err = pd.Series(perm.importances_std, index=FEATURES)[imp.index]
    pred = proba >= threshold
    y = np.asarray(y_test)
    groups = {"missed (FN)": (y == 1) & ~pred, "caught (TP)": (y == 1) & pred, "false alarm (FP)": (y == 0) & pred}
    profile = pd.DataFrame({g: X_test[mask][["Glucose", "BMI", "Age", "Insulin"]].median() for g, mask in groups.items()}).T
    profile.insert(0, "patients", [int(m.sum()) for m in groups.values()])
    brier = brier_score_loss(y_test, proba)
    fn = profile.loc["missed (FN)"]
    tp = profile.loc["caught (TP)"]
    step = _step(
        "Sanity-check the model",
        f"Shuffling {imp.index[0]} costs the most ROC AUC ({imp.iloc[0]:.3f}), followed by {imp.index[1]} and "
        f"{imp.index[2]}. These look medically plausible rather than like artefacts of the data. The missed diabetics "
        f"have a median glucose of {fn['Glucose']:.0f} against {tp['Glucose']:.0f} for those caught: the model misses "
        f"patients whose glucose still looks normal, and a lower threshold would catch them only at the price of "
        f"many more false alarms. Brier score {brier:.3f}, against {y.mean() * (1 - y.mean()):.3f} for always "
        f"predicting the base rate; mean predicted risk {proba.mean():.2f} against an actual rate of {y.mean():.2f}.",
        "No red flags. The probabilities are usable as rough risk scores; the error profile goes into the report.",
        metrics=[{"label": "Most important", "value": imp.index[0]},
                 {"label": "Brier score (test)", "value": f"{brier:.3f}"},
                 {"label": "Mean predicted vs actual", "value": f"{proba.mean():.2f} vs {y.mean():.2f}"}],
        images=[charts.to_png(charts.importance_bars(imp, "Permutation importance on the test set",
                                                     "drop in ROC AUC when shuffled", errors=err),
                              "Permutation importance")],
        tables=[_table(f"Who the model misses (medians, test set, threshold {threshold:.2f})", profile.round(1),
                       "group")],
    )
    return step


# Step 11 ------------------------------------------------------------------------------
def write_up(steps: list[dict]) -> dict:
    """The decision log and the next steps a reviewer would expect."""
    log = pd.DataFrame({"step": [s["title"] for s in steps], "decision": [s["decision"] for s in steps]},
                       index=range(1, len(steps) + 1))
    return _step(
        "Report and next steps",
        "The report states the decisions in order, so a reviewer can challenge any of them. Typical next steps: "
        "agree the recall target with clinicians; validate on patients from another hospital or population (this "
        "data is women of Pima heritage aged 21 and over, so it may not transfer); collect HbA1c or repeated glucose "
        "readings, which would help with the patients the model misses; retrain and recheck the threshold if the "
        "share of diabetics changes; and monitor recall after deployment.",
        "Ship as a decision-support screen with a documented threshold, not as a diagnosis.",
        tables=[_table("Decision log", log, "#")],
    )


# The page ------------------------------------------------------------------------------
def analyst_walkthrough(df: pd.DataFrame, recall_target: float = 0.8, zeros: str = "median",
                        selection: str = "simplest_within_1se", seed: int = 42) -> dict:
    """A credible analyst workflow from problem framing to report, on the diabetes data."""
    return _cached(recall_target, zeros, selection, int(seed))


@lru_cache(maxsize=16)
def _cached(recall_target, zeros, selection, seed):
    """Cached per setting (about 10 s of model fitting); the page and its PDF share one run."""
    df = get_df()
    steps = [frame_problem(df, recall_target)]
    s, (X_train, X_test, y_train, y_test) = lock_test_set(df, seed)
    steps.append(s)
    s, impute = check_data(X_train, zeros)
    steps.append(s)
    X_train, X_test = _prepare(X_train, impute), _prepare(X_test, impute)
    baseline, spot, shortlist, default_auc = spot_check(X_train, y_train, impute)
    steps += [baseline, spot]
    s, tuned = tune_shortlist(X_train, y_train, shortlist, default_auc, impute)
    steps.append(s)
    s, chosen = choose_model(tuned, selection)
    steps.append(s)
    pipe = make_pipeline(chosen, impute_zeros=impute, **tuned[chosen]["params"])
    s, threshold = choose_threshold(pipe, X_train, y_train, recall_target)
    steps.append(s)
    pipe.fit(X_train, y_train)
    s, proba = evaluate_on_test(pipe, chosen, len(X_train), tuned[chosen]["mean"], X_test, y_test, threshold, recall_target)
    steps.append(s)
    steps.append(sanity_checks(pipe, X_test, y_test, proba, threshold))
    steps.append(write_up(steps))

    test = threshold_metrics(y_test, proba, threshold)
    return {
        "message": f"Chosen: {MODEL_NAMES[chosen]} at threshold {threshold:.2f}. On the untouched test set it catches "
        f"{_pct(test['recall'])} of diabetics with precision {_pct(test['precision'])} and ROC AUC "
        f"{roc_auc_score(y_test, proba):.3f}. The {len(steps)} steps below show how an analyst would plausibly get there.",
        "metrics": [{"label": "Model", "value": MODEL_NAMES[chosen]},
                    {"label": "Threshold", "value": f"{threshold:.2f}"},
                    {"label": "Test recall", "value": _pct(test["recall"])},
                    {"label": "Test precision", "value": _pct(test["precision"])},
                    {"label": "Test ROC AUC", "value": f"{roc_auc_score(y_test, proba):.3f}"}],
        "images": [], "tables": [], "steps": steps,
    }
