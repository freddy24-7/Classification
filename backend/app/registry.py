"""The catalogue of topics: explanations, page settings and the function that runs each one.

The frontend list, the detail pages and the PDFs are all generated from TOPICS.
"""
from dataclasses import dataclass, field
from typing import Callable

from . import analyses as a
from . import auto_select as auto
from . import metrics_guide as mg
from . import charts
from . import models as m
from . import walkthrough as w

GROUPS = {
    "data": "Start here: the dataset",
    "intuition": "10.2–10.4 Classification models: intuition",
    "compare": "10.5 Classification models: comparison",
    "metrics": "10.6 Evaluation metrics: classification",
    "handson": "10.7 Classification hands-on",
    "workflow": "Analyst walkthrough: a full project",
    "extras": "Beyond the course",
}


@dataclass(frozen=True)
class Topic:
    id: str
    number: str
    title: str
    func: Callable
    summary: str
    what: str
    why: str
    how: list[str]
    fix: list[str]
    pitfalls: list[str] = field(default_factory=list)
    group: str = "intuition"
    params: list[dict] = field(default_factory=list)
    helpers: list[Callable] = field(default_factory=list)
    fix_title: str = "When to use it"


def col(name, label, default, columns="features"):
    return {"name": name, "label": label, "type": "column", "columns": columns, "default": default}


def select(name, label, options, default):
    return {"name": name, "label": label, "type": "select", "options": options, "default": default}


def boolean(name, label, default):
    return {"name": name, "label": label, "type": "bool", "default": default}


BOUNDARY = [col("x", "Boundary plot: x feature", "Glucose"), col("y", "Boundary plot: y feature", "BMI")]
MODELS = list(m.MODEL_NAMES)
MODEL = select("model", "Model", MODELS, "logistic")

TOPICS: list[Topic] = [
    # Start here ------------------------------------------------------------
    Topic(
        id="dataset", number="D.1", group="data", title="The diabetes dataset",
        func=a.dataset_overview, helpers=[charts.zero_counts],
        summary="768 patients, 8 features, 35% diabetic, and zeros that really mean 'not measured'.",
        what="The Pima Indians Diabetes dataset from notebook 10.7: 768 women, 8 medical measurements, and the target "
        "Outcome (1 = diabetic). This page shows the class balance, how each feature differs between the classes, "
        "and a hidden data-quality problem.",
        why="Two facts shape every later page. First, 65% of patients are not diabetic, so a model that always says "
        "'no' already scores 65% accuracy: that is the baseline to beat. Second, isnull() reports no missing values, "
        "but Glucose, BloodPressure, SkinThickness, Insulin and BMI contain zeros that are physically impossible. "
        "They are missing values in disguise.",
        how=[
            "df['Outcome'].value_counts(normalize=True) gives the class balance.",
            "(df[cols] == 0).sum() counts the impossible zeros per column.",
            "df.groupby('Outcome').mean() shows which features separate the classes; Glucose differs the most.",
            "Pick a feature in the settings to see its distribution per class.",
        ],
        fix_title="What to do with it",
        fix=[
            "Always compare models against the majority-class baseline, not against 0%.",
            "Use stratify=y when splitting so train and test keep the 65/35 balance.",
            "Treat the zeros as missing and impute them inside the pipeline (see 'Zeros as missing values').",
        ],
        pitfalls=["Insulin is zero for 49% of rows: imputing it is guesswork, so check whether it helps at all."],
        params=[col("column", "Feature to plot", "Glucose")],
    ),
    # 10.2 ------------------------------------------------------------------
    Topic(
        id="logistic", number="10.2", title="Logistic regression",
        func=a.logistic_regression, helpers=[m.make_pipeline, charts.decision_boundary],
        summary="A weighted sum of the features, squashed by the sigmoid into a probability.",
        what="Logistic regression computes a linear score z = b0 + b1·x1 + … + bn·xn and turns it into a probability "
        "with the sigmoid σ(z) = 1 / (1 + e^−z). It predicts class 1 when the probability is at least 0.5, which "
        "is where z = 0. The boundary between the classes is therefore a straight line (a hyperplane).",
        why="It is the standard first model for classification: fast, hard to overfit, and its coefficients are "
        "readable. Each coefficient is a change in log-odds, so e^b is an odds ratio: on standardised features, how "
        "much the odds of diabetes multiply for one standard deviation more of that feature.",
        how=[
            "Fitting finds the coefficients that maximise the likelihood of the observed labels (minimise log loss).",
            "C is the inverse regularisation strength: small C shrinks the coefficients towards 0 (simpler model).",
            "Scale the features first, so coefficients are comparable and the solver converges.",
            "The S-curve shows the model on one feature in its original units; the boundary plot uses two features.",
        ],
        fix=[
            "A strong baseline for any binary problem, especially with roughly linear effects.",
            "When you need to explain the model: coefficients and odds ratios are easy to report.",
            "When you need well-calibrated probabilities out of the box.",
        ],
        pitfalls=[
            "It cannot learn interactions or curves on its own; add those as features.",
            "Correlated features split their weight, so individual coefficients become unstable.",
            "Coefficients on unscaled features cannot be compared with each other.",
        ],
        params=[select("C", "C (inverse regularisation)", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 1.0)] + BOUNDARY,
    ),
    Topic(
        id="knn", number="10.2", title="K-nearest neighbours",
        func=a.k_nearest_neighbours, helpers=[m.make_pipeline],
        summary="Look at the k most similar patients and take a vote.",
        what="KNN stores the training data. To classify a new patient it finds the k nearest training rows "
        "(Euclidean distance by default) and predicts the majority class; the share of votes is the probability. "
        "There is no training step beyond storing the data, which is why it is called a lazy learner.",
        why="It shows the bias–variance trade-off very clearly. k = 1 memorises the training data (100% train accuracy, "
        "jagged boundary, overfitting); a very large k averages over almost everyone and predicts the majority class "
        "(underfitting). It also shows why scaling matters: unscaled, Insulin (0–846) drowns out "
        "DiabetesPedigreeFunction (0.08–2.4) in the distance.",
        how=[
            "Distance between patients a and b: sqrt(Σ (a_i − b_i)²) over the features.",
            "weights='distance' gives closer neighbours a bigger vote.",
            "The accuracy-against-k chart uses 5-fold CV; pick k where the validation curve peaks.",
            "Use an odd k for two classes to avoid ties.",
        ],
        fix=[
            "Small datasets with few features and a meaningful distance.",
            "As an intuitive baseline, and when the boundary is irregular.",
        ],
        pitfalls=[
            "Always scale the features (StandardScaler inside the pipeline).",
            "Prediction is slow on large data: every prediction compares against all training rows.",
            "With many features, distances become similar to each other (the curse of dimensionality).",
        ],
        params=[select("k", "k (neighbours)", [1, 3, 5, 7, 11, 15, 21, 31, 51], 15),
                select("weights", "Vote weights", ["uniform", "distance"], "uniform"),
                boolean("scale", "Scale features", True)] + BOUNDARY,
    ),
    # 10.3 ------------------------------------------------------------------
    Topic(
        id="tree", number="10.3", title="Decision tree",
        func=a.decision_tree, helpers=[charts.tree_diagram],
        summary="A flowchart of yes/no questions learned from the data.",
        what="A decision tree splits the data again and again on one feature at a time (for example Glucose ≤ 127.5), "
        "choosing each split to make the two groups as pure as possible. A leaf predicts the share of diabetics in it. "
        "The boundary is made of axis-aligned rectangles.",
        why="Trees need no scaling, handle non-linear effects and interactions, and can be read by a human. They are "
        "also the building block of random forests and gradient boosting. On their own they overfit easily: a deep "
        "tree can memorise every training row.",
        how=[
            "Gini impurity of a node: 1 − p0² − p1². 0 means pure; 0.5 is the worst for two classes.",
            "Entropy (criterion='entropy'): −Σ p·log2(p). Both usually choose similar splits.",
            "Each split maximises the drop in impurity, weighted by the size of the child nodes.",
            "max_depth and min_samples_leaf stop the tree growing; the depth chart shows the overfitting.",
        ],
        fix=[
            "When the model must be explained as rules.",
            "As a quick check of which features and thresholds matter.",
            "In practice, use a forest or boosting for accuracy.",
        ],
        pitfalls=[
            "Unstable: a small change in the data can give a completely different tree.",
            "Impurity importance favours features with many distinct values.",
        ],
        params=[select("max_depth", "max_depth", ["1", "2", "3", "4", "5", "6", "8", "10", "None"], "4"),
                select("criterion", "Split criterion", ["gini", "entropy"], "gini"),
                select("min_samples_leaf", "min_samples_leaf", [1, 5, 10, 20, 50], 1)],
    ),
    Topic(
        id="naive_bayes", number="10.3", title="Naive Bayes",
        func=a.naive_bayes,
        summary="Bayes' rule, assuming the features are independent within each class.",
        what="Naive Bayes applies Bayes' rule: P(class | x) ∝ P(class) · P(x | class). The 'naive' part assumes the "
        "features are independent given the class, so P(x | class) is a product of one-feature distributions. "
        "Gaussian naive Bayes models each feature in each class as a normal distribution with its own mean and "
        "standard deviation.",
        why="It trains instantly, needs very little data and works surprisingly well, especially on text. It is also "
        "the clearest example of a probabilistic (generative) classifier. The worked example shows the whole "
        "prediction: start from the prior log-odds, then add each feature's evidence.",
        how=[
            "Log-odds of diabetes = log(P(1)/P(0)) + Σ log(P(x_i|1)/P(x_i|0)).",
            "Each term is the evidence from one feature: positive pushes towards diabetic.",
            "The posterior is the sigmoid of the total log-odds.",
            "Pick a feature to see the two Gaussians the model fitted, and a test patient for the worked example.",
        ],
        fix=[
            "Very small data, many features, or text (MultinomialNB on word counts).",
            "A fast baseline before trying slower models.",
        ],
        pitfalls=[
            "Correlated features are counted twice, which makes probabilities too extreme (see the calibration page).",
            "Gaussian NB assumes bell-shaped features; skewed ones like Insulin fit badly.",
        ],
        params=[col("column", "Feature to plot", "Glucose"),
                select("patient", "Test patient for the worked example", list(range(10)), 0)],
    ),
    # 10.4 ------------------------------------------------------------------
    Topic(
        id="svm", number="10.4", title="Support vector machine",
        func=a.support_vector_machine, helpers=[charts.decision_boundary],
        summary="The boundary with the widest margin; kernels let it curve.",
        what="A support vector machine finds the boundary that separates the classes with the widest possible margin. "
        "Only the training points on or inside the margin, the support vectors, determine it. C sets how much "
        "margin violation is tolerated. The kernel trick (RBF, polynomial) fits a linear boundary in a "
        "higher-dimensional space, which becomes a curved boundary in the original features.",
        why="SVMs are strong on small and medium datasets with clear margins, and the margin idea explains why they "
        "generalise well. They show how C and gamma control flexibility: large C or large gamma gives a wiggly "
        "boundary that overfits.",
        how=[
            "Linear kernel: a straight boundary, similar to logistic regression.",
            "RBF kernel: similarity exp(−gamma·distance²); large gamma means each point only influences its neighbours.",
            "Circled points in the boundary plot are the support vectors.",
            "SVC outputs a distance, not a probability; here it is wrapped in CalibratedClassifierCV (Platt scaling).",
        ],
        fix=[
            "Small to medium tabular datasets, after scaling.",
            "High-dimensional data such as text, with a linear kernel.",
        ],
        pitfalls=[
            "Training time grows faster than the number of rows; impractical beyond ~50k rows.",
            "Must scale the features.",
            "Tune C and gamma together (see the tuning page).",
        ],
        params=[select("kernel", "Kernel", ["linear", "rbf", "poly"], "rbf"),
                select("C", "C", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0),
                select("gamma", "gamma", ["scale", 0.01, 0.1, 1.0, 10.0], "scale")] + BOUNDARY,
    ),
    Topic(
        id="forest", number="10.4", title="Random forest",
        func=a.random_forest,
        summary="Hundreds of different trees voting together.",
        what="A random forest trains many decision trees, each on a bootstrap sample of the rows (sampling with "
        "replacement) and considering only a random subset of features at each split. The forest predicts the "
        "average of the trees' probabilities. This is bagging plus feature randomness.",
        why="Single trees have high variance. Because the forest's trees are different from each other, their errors "
        "partly cancel, so the average is far more stable and accurate. Forests work well with almost no tuning, "
        "which makes them a common default for tabular data.",
        how=[
            "Each tree leaves out about 37% of the rows; scoring them gives the free out-of-bag (OOB) estimate.",
            "max_features='sqrt' tries √8 ≈ 3 features per split; 1.0 uses all (plain bagging, more correlated trees).",
            "More trees never overfit more; the score levels off.",
            "min_samples_leaf and max_depth limit each tree.",
        ],
        fix=[
            "A strong, low-effort default for tabular data.",
            "When you want OOB validation and feature importance for free.",
        ],
        pitfalls=[
            "Larger and slower to predict than one tree, and not readable as rules.",
            "Probabilities are often squeezed away from 0 and 1.",
            "Impurity importance is biased; prefer permutation importance.",
        ],
        params=[select("max_features", "max_features", ["sqrt", "log2", "0.5", "1.0"], "sqrt"),
                select("max_depth", "max_depth", ["3", "5", "8", "None"], "None"),
                select("min_samples_leaf", "min_samples_leaf", [1, 3, 5, 10], 3)],
    ),
    Topic(
        id="boosting", number="10.4", title="Gradient boosting",
        func=a.gradient_boosting,
        summary="Small trees added one by one, each correcting the last.",
        what="Gradient boosting builds an ensemble in sequence. It starts from a constant prediction, then each new "
        "shallow tree is fitted to the errors (the gradient of the log loss) of the ensemble so far, and added with "
        "a small weight, the learning rate. Libraries such as XGBoost, LightGBM and CatBoost use the same idea.",
        why="Boosting is usually the most accurate method on tabular data. It reduces bias step by step, whereas "
        "a random forest reduces variance. But it can overfit: the stage charts show training accuracy rising "
        "while the test loss starts to get worse.",
        how=[
            "Each stage: prediction += learning_rate × new_tree(errors).",
            "A smaller learning rate needs more trees, but usually generalises better.",
            "max_depth 2–5 is typical; each tree only has to capture a little.",
            "Use early stopping (n_iter_no_change) or CV to choose the number of trees.",
        ],
        fix=[
            "When you want the best accuracy on tabular data and can tune.",
            "HistGradientBoostingClassifier for larger data; it also handles NaN natively.",
        ],
        pitfalls=[
            "More tuning-sensitive than a forest.",
            "Trees are built one after another, so training does not parallelise across trees.",
        ],
        params=[select("learning_rate", "learning_rate", [0.01, 0.05, 0.1, 0.3, 1.0], 0.05),
                select("max_depth", "max_depth", [1, 2, 3, 5], 3)],
    ),
    # 10.5 ------------------------------------------------------------------
    Topic(
        id="comparison", number="10.5", group="compare", title="Comparing the models",
        func=a.model_comparison, helpers=[m.cv_scores, m.make_pipeline, charts.boundary_grid],
        summary="Seven models and a baseline on the same folds, with fold-to-fold spread.",
        what="All seven models are scored with the same 5-fold stratified cross-validation on five metrics, next to a "
        "baseline that always predicts 'not diabetic'. The page also shows train against validation scores "
        "(overfitting) and every model's decision boundary on the same two features. It opens with the course's "
        "model cheat sheet, with each model's measured score and speed on this dataset next to it.",
        why="A fair comparison needs the same data splits, the same metric, a baseline, and a sense of the noise. "
        "With 768 rows, fold scores vary by a few points, so a 1% difference between models is usually not real. "
        "The boundary grid shows what each model can and cannot learn: lines, rectangles, smooth curves or blobs.",
        how=[
            "cross_validate() with StratifiedKFold(5, shuffle=True, random_state=42) for every model.",
            "Scaling sits inside each pipeline, so it is fitted on training folds only.",
            "'auto' scaling scales only the models that need it (logistic, KNN, SVM).",
            "Pick the metric that matches the problem; the table sorts by it.",
            "The cheat sheet's 'Ensemble (XGBoost)' row is gradient boosting here, scikit-learn's version.",
        ],
        fix_title="How to choose",
        fix=[
            "Start with the simplest model within noise of the best (often logistic regression here).",
            "Consider speed, interpretability and probability quality, not just the score.",
            "Confirm the winner once on a held-out test set.",
        ],
        pitfalls=[
            "Comparing on a single train/test split can reverse the ranking by chance.",
            "Tuning one model and not the others makes the comparison unfair.",
            "The cheat sheet is a rule of thumb. KNN is slow at prediction, not at fitting, and its 'high' "
            "interpretability means 'look at the neighbours', not a formula. Logistic regression can fit curves if "
            "you add features such as squares or interactions.",
        ],
        params=[select("metric", "Rank by", ["roc_auc", "accuracy", "f1", "recall", "precision"], "roc_auc"),
                select("scaling", "Scaling", ["auto", "always", "never"], "auto")] + BOUNDARY,
    ),
    # 10.6 ------------------------------------------------------------------
    Topic(
        id="metrics_guide", number="10.6", group="metrics", title="The metrics on this dataset, and which model fits",
        func=mg.metrics_guide, helpers=[mg.metric_concepts, charts.confusion_grid],
        summary="Accuracy, confusion matrix, precision, recall, F1 and ROC AUC for all seven models, and a verdict.",
        what="One page that goes through the six evaluation tools of 10.6 on the diabetes data. A table works out each "
        "metric with the chosen model's actual counts, names the trap it hides on this dataset and gives a verdict "
        "for diabetes screening. Charts show the same predictions scored five ways against the always-'no' baseline, "
        "every model on every metric, all seven confusion matrices and the ROC curves. A last table judges how "
        "suitable each model is, from the numbers and from what the numbers do not show.",
        why="Each metric answers a different question, and on imbalanced data they disagree: the model that looks "
        "best on accuracy is not the one that catches the most diabetics. Choosing the metric is choosing what "
        "'good' means for the problem, so it comes before choosing the model.",
        how=[
            "Predictions are out-of-fold over all 768 patients (5-fold CV), so every patient is predicted by a model "
            "that never saw them. 268 diabetics give steadier numbers than the 54 in the test split.",
            "Accuracy, precision, recall and F1 depend on the threshold; change it in the settings and watch them move.",
            "ROC AUC and 'precision at 80% recall' do not depend on the threshold.",
            "Brier score measures how trustworthy the probabilities are (lower is better).",
            "Verdict: 'strong candidate' within 0.01 ROC AUC of the best, 'usable' within 0.03.",
        ],
        fix_title="Which metric to use here",
        fix=[
            "Compare models on ROC AUC (or precision at a fixed recall).",
            "Choose the threshold for recall: in screening, a missed diabetic is the costly error.",
            "Report precision next to it, so the cost in false alarms is visible.",
            "Show the confusion matrix in patients, and never report accuracy alone.",
        ],
        pitfalls=[
            "The 0.5 threshold is a default: at 0.5 every model misses a large share of the diabetics.",
            "Out-of-fold numbers here differ slightly from the test-split numbers on the other pages; both are honest, "
            "on different patients.",
            "Precision–recall AUC (see the precision–recall page) is often more informative than ROC AUC when the "
            "positive class is rare; at 35% diabetics here, both work.",
        ],
        params=[MODEL, select("threshold", "Decision threshold", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 0.5)],
    ),
    Topic(
        id="confusion", number="10.6", group="metrics", title="Confusion matrix and its metrics",
        func=a.confusion_and_metrics, helpers=[m.threshold_metrics],
        summary="TP, FP, FN, TN and the accuracy, precision, recall, F1 and specificity built from them.",
        what="The confusion matrix counts, on the test split, the true negatives, false positives, false negatives and "
        "true positives. Every threshold metric comes from these four numbers. The table shows each formula worked "
        "out with the real counts.",
        why="Accuracy hides which mistakes the model makes. In diabetes screening a false negative (a missed diabetic) "
        "is much worse than a false positive (an extra test). With 65% negatives, a model that never predicts "
        "diabetic gets 65% accuracy and 0% recall. Precision and recall show the trade-off that matters.",
        how=[
            "Precision = TP / (TP + FP): when the model says diabetic, how often it is right.",
            "Recall = TP / (TP + FN): of the real diabetics, how many it finds.",
            "F1 = harmonic mean of precision and recall; it is low if either is low.",
            "Change the threshold to watch the counts move between the cells.",
        ],
        fix_title="Which metric to use",
        fix=[
            "Missing a positive is costly (screening, fraud): optimise recall, and report precision too.",
            "False alarms are costly (spam filter): optimise precision.",
            "Imbalanced classes: report balanced accuracy, F1 or MCC instead of accuracy.",
        ],
        pitfalls=["Precision and recall depend on the threshold; ROC AUC and average precision do not."],
        params=[MODEL, select("threshold", "Decision threshold",
                              [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9], 0.5)],
    ),
    Topic(
        id="roc", number="10.6", group="metrics", title="ROC curve and AUC",
        func=a.roc_analysis,
        summary="Recall against false positive rate at every threshold.",
        what="The ROC curve plots the true positive rate (recall) against the false positive rate (1 − specificity) as "
        "the threshold moves from 1 to 0. The area under it (AUC) summarises the curve in one number: 1.0 is "
        "perfect, 0.5 is random.",
        why="AUC measures how well the model ranks patients, independent of any threshold. It equals the probability "
        "that a randomly chosen diabetic gets a higher score than a randomly chosen non-diabetic; the page checks this "
        "directly. That makes it a good metric for comparing models before you pick a threshold.",
        how=[
            "roc_curve(y_test, y_prob) returns FPR, TPR and the thresholds.",
            "roc_auc_score(y_test, y_prob) needs probabilities (or scores), not predicted labels.",
            "Each point on the curve is one threshold; the dot marks 0.5.",
            "Youden's J = TPR − FPR picks the point furthest above the diagonal.",
        ],
        fix_title="When to use it",
        fix=[
            "Comparing models on ranking quality.",
            "When both classes matter and the classes are not extremely imbalanced.",
        ],
        pitfalls=[
            "With a rare positive class, ROC AUC can look good while precision is poor; check the PR curve.",
            "AUC says nothing about calibration or about the threshold you will use.",
        ],
        params=[select("model", "Model", ["all"] + MODELS, "logistic")],
    ),
    Topic(
        id="pr_curve", number="10.6", group="metrics", title="Precision–recall curve",
        func=a.precision_recall_analysis,
        summary="Precision against recall at every threshold; the honest view for imbalanced data.",
        what="The precision–recall curve plots precision against recall over all thresholds. Average precision (AP) "
        "summarises it. A random model's precision equals the share of positives (0.35 here), so that is the baseline.",
        why="It focuses on the positive class. When positives are rare, ROC can be dominated by the many easy "
        "negatives, while the PR curve shows how many false alarms you pay for each extra case found. The table "
        "answers a practical question: if we must catch 80% of diabetics, how precise can we be?",
        how=[
            "precision_recall_curve(y_test, y_prob) and average_precision_score(y_test, y_prob).",
            "Moving right along the curve lowers the threshold: more recall, usually less precision.",
            "The best F1 is the point that balances the two.",
        ],
        fix=["Imbalanced problems (fraud, rare disease, defects).", "When the positive class is what you care about."],
        pitfalls=["The curve is jagged on small test sets; 54 positives here."],
        params=[select("model", "Model", ["all"] + MODELS, "logistic")],
    ),
    Topic(
        id="threshold", number="10.6", group="metrics", title="Choosing the decision threshold",
        func=a.threshold_selection, helpers=[m.oof_proba],
        summary="Pick the threshold for your goal on training folds, then confirm on the test set.",
        what="predict() uses 0.5, but you can apply any threshold to predict_proba(). This page chooses the threshold "
        "by an objective (best F1, Youden's J, lowest cost, or a minimum recall) using out-of-fold predictions on the "
        "training split, and then reports the result once on the test split.",
        why="The best threshold depends on the cost of each mistake, which is a business or medical decision, not a "
        "modelling one. Choosing it on the test set would leak: the test score would be optimistic. Out-of-fold "
        "training predictions give an honest place to choose.",
        how=[
            "oof = cross_val_predict(pipe, X_train, y_train, method='predict_proba')[:, 1].",
            "Score every threshold from 0.05 to 0.95 on oof and pick by the objective.",
            "Cost objective: minimise fn_cost × FN + 1 × FP.",
            "Apply the chosen threshold to the test probabilities.",
        ],
        fix_title="How to choose",
        fix=[
            "Agree the cost of a false negative against a false positive with the domain experts.",
            "For screening, set a recall target and take the highest threshold that meets it.",
            "Store the threshold with the model; it is part of the model.",
        ],
        pitfalls=["Thresholds need well-ranked probabilities; recheck them if the class balance changes."],
        params=[MODEL,
                select("objective", "Objective", ["f1", "youden", "cost", "recall_target"], "f1"),
                select("fn_cost", "Cost of a missed diabetic (× false alarm)", [1.0, 2.0, 3.0, 5.0, 10.0], 5.0),
                select("target_recall", "Recall target", [0.7, 0.8, 0.9, 0.95], 0.8)],
    ),
    Topic(
        id="calibration", number="10.6", group="metrics", title="Probability calibration",
        func=a.calibration_analysis,
        summary="When the model says 70%, are 70% of those patients diabetic?",
        what="A model is calibrated when its predicted probabilities match observed frequencies. The reliability "
        "diagram groups predictions into bins and plots the mean prediction against the actual share of diabetics. "
        "The Brier score (mean squared error of the probabilities) and log loss measure probability quality.",
        why="Thresholds, risk scores and expected-cost decisions all assume the probabilities are meaningful. Naive "
        "Bayes is usually overconfident and forests underconfident. CalibratedClassifierCV fixes this with Platt "
        "(sigmoid) or isotonic scaling, without changing how the model ranks patients.",
        how=[
            "calibration_curve(y, p, n_bins=8, strategy='quantile') gives the reliability points.",
            "Points below the diagonal are overconfident; above are underconfident.",
            "Sigmoid fits a logistic curve to the scores; isotonic fits a step function (needs more data).",
            "All probabilities here are out-of-fold over the 768 rows.",
        ],
        fix=["Whenever probabilities are used as probabilities: risk scores, expected costs, ranking with thresholds."],
        pitfalls=["Isotonic calibration can overfit on small data like this.",
                  "Calibrate on data the model was not trained on (CalibratedClassifierCV does this with cv=5)."],
        params=[select("model", "Model", MODELS, "naive_bayes")],
    ),
    # 10.7 ------------------------------------------------------------------
    Topic(
        id="handson", number="10.7", group="handson", title="Classification hands-on (notebook)",
        func=a.hands_on, helpers=[m.label_metrics],
        summary="Notebook 10.7 end to end: split, scale, logistic regression, metrics, ROC.",
        what="The workflow from notebook 10_7_classification_hands_on.ipynb: separate X and y, an 80/20 stratified "
        "split with random_state=42, StandardScaler fitted on the training data, logistic regression, then train and "
        "test metrics, the confusion matrix, and the ROC curve with AUC. With the default settings the numbers match "
        "the notebook: test accuracy 71.43%, precision 60.87%, recall 51.85%, F1 56.00%.",
        why="This is the template for every classification project. Change the settings to see what each step does: "
        "a different seed moves the test accuracy by several points (one split is noisy), and removing stratification "
        "or scaling changes the result.",
        how=[
            "train_test_split(X, y, test_size=0.2, stratify=y, random_state=42).",
            "scaler.fit_transform(X_train), then scaler.transform(X_test): never fit on test data.",
            "model.fit, then predict for labels and predict_proba[:, 1] for probabilities.",
            "Metrics on train and test; the gap shows how much the model overfits.",
        ],
        fix_title="Next steps after the notebook",
        fix=[
            "Treat the zeros as missing values.",
            "Use cross-validation rather than one split.",
            "Tune the threshold for recall; a 52% recall misses half the diabetics.",
            "Compare other models (10.5).",
        ],
        pitfalls=["Seed 42 is not special: try other seeds to see how much one split varies."],
        params=[select("test_size", "test_size", [0.1, 0.2, 0.3, 0.4], 0.2),
                boolean("stratify", "Stratify on y", True),
                boolean("scale", "StandardScaler", True),
                select("seed", "random_state", [0, 1, 7, 42, 123], 42)],
    ),
    # Analyst walkthrough ---------------------------------------------------
    Topic(
        id="walkthrough", number="W.1", group="workflow", title="How an analyst would run this project",
        func=w.analyst_walkthrough,
        helpers=[w.frame_problem, w.lock_test_set, w.check_data, w.spot_check, w.tune_shortlist, w.choose_model,
                 w.choose_threshold, w.evaluate_on_test, w.sanity_checks, w.write_up],
        summary="Eleven steps from framing the problem to the report: compare, shortlist, tune, choose, test once.",
        what="A plausible end-to-end workflow on the diabetes data, the way an experienced analyst would typically "
        "run it after the notebook. It frames the problem, locks away a test set, fixes the data, sets a baseline, "
        "spot-checks every model, tunes a shortlist, picks a model and a threshold, evaluates once on the test set, "
        "sanity-checks the result and writes it up. Every number on the page is computed live; the decisions follow "
        "common practice, not the one right answer.",
        why="Each earlier page teaches one tool. A real project chains them, and the order matters: decide the metric "
        "before looking at scores, choose on cross-validation, touch the test set once. The page also shows the "
        "judgement calls that the numbers alone do not make, such as preferring a simpler model when the scores are "
        "tied, or trading precision for recall.",
        how=[
            "Steps 1–3 set the target metric, hold out 20% of the patients, and treat the impossible zeros as missing.",
            "Steps 4–5 score a baseline and all seven models on the same 5 folds of the training split.",
            "Models within one fold-to-fold spread of the best are shortlisted and tuned with a small grid (step 6).",
            "Step 7 uses the one-standard-error rule: the simplest model within one standard error of the best.",
            "Step 8 picks the highest threshold that meets the recall target on out-of-fold predictions.",
            "Steps 9–11 test once with bootstrap intervals, check importance, calibration and errors, and report.",
        ],
        fix_title="Choices you can change",
        fix=[
            "Recall target: how many diabetics the screen must catch. Higher means more false alarms.",
            "Zeros: impute them (recommended) or keep them as the notebook does.",
            "Selection rule: the simplest model within 1 SE, or simply the best mean score.",
            "Seed: a different train/test split. Watch how much the test numbers move.",
        ],
        pitfalls=[
            "This is one credible path, not the truth: another analyst might prefer F1, a cost matrix or recall at "
            "a fixed false-alarm rate, and would get a different threshold.",
            "With 154 test patients and 54 diabetics, the test numbers have wide intervals; read them with the ranges.",
            "The first run of each setting takes about 10 seconds: it fits several hundred models.",
        ],
        params=[select("recall_target", "Recall target", [0.7, 0.75, 0.8, 0.85, 0.9], 0.8),
                select("zeros", "Impossible zeros", ["median", "keep"], "median"),
                select("selection", "Model selection rule", ["simplest_within_1se", "best_mean"],
                       "simplest_within_1se"),
                select("seed", "Train/test split seed", [42, 0, 7, 123], 42)],
    ),
    Topic(
        id="auto_select", number="W.2", group="workflow", title="Automated model selection with scikit-learn",
        func=auto.automated_selection, helpers=[auto.search_space, auto.precision_at_recall],
        summary="One GridSearchCV over every model family, then TunedThresholdClassifierCV; compared with W.1.",
        what="scikit-learn can automate the mechanical part of the walkthrough. GridSearchCV accepts the model itself "
        "as a search parameter, so one search compares all seven model families and their settings on the same "
        "folds. TunedThresholdClassifierCV (scikit-learn 1.5 and later) then chooses the decision threshold by "
        "cross-validation for an objective you give it. The result is put next to the analyst walkthrough on the "
        "same split.",
        why="These tools replace loops you would otherwise write by hand, and they make the comparison fair: every "
        "candidate sees the same folds and the same preprocessing. What they do not do is judgement. The search "
        "always takes the highest mean score, even when the lead is far inside the noise, and the threshold "
        "objective still has to come from the problem.",
        how=[
            "Pipeline: [median imputer] → StandardScaler → model; scaling does not change tree-based models.",
            "param_grid is a list of dicts, one per family: {'model': [LogisticRegression()], 'model__C': [...]}.",
            "GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc').fit(X_train, y_train).",
            "TunedThresholdClassifierCV(search.best_estimator_, scoring=..., cv=5) tries 100 thresholds per fold.",
            "Recall target: a custom scorer gives the precision when recall reaches the target, and a penalty if not.",
            "The test set is used once, at the end, exactly as in the walkthrough.",
        ],
        fix_title="Other tools that do this",
        fix=[
            "RandomizedSearchCV or HalvingGridSearchCV when the grid gets large.",
            "LazyPredict: a one-line table of about 30 models with defaults (no cross-validation).",
            "PyCaret compare_models(): a cross-validated leaderboard, then tune_model() and finalize_model().",
            "FLAML, AutoGluon, TPOT, H2O AutoML: full AutoML with time budgets and ensembles.",
            "Optuna: smarter tuning of one model's settings.",
        ],
        pitfalls=[
            "The winner's CV score is optimistic: it is the maximum over 24 candidates. Nested CV or the untouched "
            "test set gives the honest number.",
            "A leaderboard order inside one standard error is noise; change the seed and the winner can change.",
            "With balanced accuracy or F1 the tuned threshold can land far from what screening needs.",
            "The first run of each setting takes about 15 seconds (it also runs the walkthrough for the comparison).",
        ],
        params=[select("objective", "Threshold objective", ["recall_target", "f1", "balanced_accuracy"],
                       "recall_target"),
                select("recall_target", "Recall target", [0.7, 0.75, 0.8, 0.85, 0.9], 0.8),
                select("zeros", "Impossible zeros", ["median", "keep"], "median"),
                select("seed", "Train/test split seed", [42, 0, 7, 123], 42)],
    ),
    # Beyond the course -----------------------------------------------------
    Topic(
        id="zeros", number="E.1", group="extras", title="Zeros as missing values",
        func=a.zeros_as_missing, helpers=[a.zeros_to_nan],
        summary="The notebook keeps the impossible zeros; does imputing them help?",
        what="Notebook 10.7 says handling zero-as-missing values can be explored later. This page does it: replace the "
        "zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI with NaN, then compare four strategies with "
        "cross-validation.",
        why="A zero blood pressure makes a linear model believe that low pressure has a strong effect. Imputing with "
        "the median removes those fake values, and a missing indicator lets the model use the fact that a value was "
        "not measured, which is often informative in medical data.",
        how=[
            "df[cols].replace(0, np.nan) marks the zeros as missing.",
            "SimpleImputer(strategy='median') inside the pipeline, so the median comes from the training folds only.",
            "add_indicator=True adds a 0/1 column per feature that had missing values.",
            "Dropping rows keeps 392 of 768 patients, a different population.",
        ],
        fix_title="What to do",
        fix=["Always check for impossible values, not only for NaN.",
             "Impute inside the pipeline and compare with cross-validation.",
             "Tree models are less affected by fake zeros than linear and distance models."],
        params=[select("model", "Model", MODELS, "logistic")],
    ),
    Topic(
        id="tuning", number="E.2", group="extras", title="Hyperparameter tuning",
        func=a.hyperparameter_tuning,
        summary="GridSearchCV over two settings, with an untouched test set.",
        what="GridSearchCV tries every combination of two hyperparameters with 5-fold CV on the training split, keeps "
        "the best, refits it on the whole training split, and is then scored once on the test split.",
        why="The defaults are rarely best. Searching on the training split only keeps the test score honest. The "
        "heatmap shows how sensitive the model is: a flat area means the choice hardly matters.",
        how=[
            "Parameter names use the pipeline step: model__C, model__gamma.",
            "scoring sets what 'best' means; refit=True retrains the winner.",
            "n_jobs=-1 runs the fits in parallel on bigger problems.",
            "For an unbiased estimate of the tuning itself, use nested CV.",
        ],
        fix_title="Tips",
        fix=["Start with a coarse logarithmic grid, then refine.",
             "RandomizedSearchCV or Optuna for many parameters.",
             "Tune every model you compare, or the comparison is unfair."],
        pitfalls=["The best CV score is optimistic: it was selected as the maximum of many noisy scores."],
        params=[select("model", "Model", list(a.GRIDS), "svm"),
                select("scoring", "Scoring", ["roc_auc", "accuracy", "f1", "recall"], "roc_auc")],
    ),
    Topic(
        id="curves", number="E.3", group="extras", title="Learning and validation curves",
        func=a.learning_and_validation_curves,
        summary="Diagnose under- and overfitting: more data, or a different model?",
        what="A learning curve shows train and validation scores as the training set grows. A validation curve shows "
        "them as one complexity setting changes (depth, k, C).",
        why="Together they tell you what to do next. A large gap between train and validation means high variance: "
        "more data or more regularisation helps. Both scores low and close together means high bias: you need a "
        "more flexible model or better features, and more data will not help.",
        how=[
            "learning_curve(pipe, X, y, train_sizes=…, cv=5).",
            "validation_curve(pipe, X, y, param_name='model__max_depth', param_range=…).",
            "Shaded bands are ± one standard deviation over the folds.",
        ],
        fix_title="Reading the curves",
        fix=["Validation still rising at full size: collect more data.",
             "Curves converged and low: add features or use a more flexible model.",
             "Validation peaks then falls with complexity: choose the peak."],
        params=[select("model", "Model", list(a.COMPLEXITY), "tree")],
    ),
    Topic(
        id="importance", number="E.4", group="extras", title="Feature importance",
        func=a.feature_importance,
        summary="Which features does the model rely on? Permutation importance for any model.",
        what="Permutation importance shuffles one feature at a time on the test split and measures how much the ROC "
        "AUC drops. It works for any model. It is compared with the model's built-in importance: impurity for trees "
        "and |coefficient| for logistic regression.",
        why="Importance checks that the model uses sensible signals (Glucose, BMI, age) rather than artefacts, helps "
        "remove useless features, and explains the model to others.",
        how=[
            "permutation_importance(pipe, X_test, y_test, scoring='roc_auc', n_repeats=20).",
            "Error bars are the spread over the 20 shuffles.",
            "A drop within its spread means the feature makes no clear difference.",
        ],
        fix_title="Tips",
        fix=["Compute it on held-out data, not on the training data.",
             "Use SHAP for per-patient explanations."],
        pitfalls=["Correlated features share importance: shuffling one leaves the other to carry the signal.",
                  "Importance is about this model, not about cause and effect."],
        params=[select("model", "Model", MODELS, "forest")],
    ),
]

BY_ID = {t.id: t for t in TOPICS}
