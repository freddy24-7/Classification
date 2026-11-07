# Classification

Classification models and how to evaluate them: course sections 10.2–10.7, an analyst walkthrough and four extra topics, run on the Pima Indians Diabetes dataset (`diabetes.csv`). Each topic has a page with an explanation, results you can explore (choose models, settings and features), the reference code and an explanatory PDF. Same structure as the `data quality` project.

| Group | Topics |
|---|---|
| Start here | the dataset: class balance, features by class, zeros that are really missing |
| 10.2–10.4 Model intuition | logistic regression, KNN (10.2); decision tree, naive Bayes (10.3); SVM, random forest, gradient boosting (10.4) |
| 10.5 Comparison | all models and a baseline on the same CV folds, train vs validation, decision boundaries side by side |
| 10.6 Evaluation metrics | confusion matrix and metrics, ROC/AUC, precision–recall, threshold choice, calibration |
| 10.5 cheat sheet | the course's model summary table, at the top of the 10.5 comparison page, with measured score, fit and predict time per model on this dataset |
| 10.6 metrics guide | accuracy, confusion matrix, precision, recall, F1 and ROC AUC worked out on this dataset for all seven models, with a suitability verdict per model (`backend/app/metrics_guide.py`) |
| 10.7 Hands-on | notebook `10_7_classification_hands_on.ipynb`, live; the defaults reproduce its numbers |
| Analyst walkthrough | one credible end-to-end project in 11 steps (`backend/app/walkthrough.py`): frame, lock a test set, baseline, spot-check, shortlist, tune, choose (1-SE rule), threshold, test once with bootstrap intervals, sanity checks, report. First run per setting takes ~10 s, then it is cached |
| Automated selection | `backend/app/auto_select.py`: one `GridSearchCV` over all model families plus `TunedThresholdClassifierCV`, compared with the walkthrough on the same split |
| Beyond the course | zeros as missing values, hyperparameter tuning, learning/validation curves, feature importance |

Which models belong to which part of 10.2–10.4 is a guess from the video titles; adjust `number=` in `registry.py` when you know.

```
backend/              FastAPI + scikit-learn
  data/diabetes.csv
  app/dataset.py      loads the CSV, the notebook's train/test split, the zero-as-missing columns
  app/models.py       scikit-learn building blocks: the classifiers, pipelines, CV and metric helpers
  app/analyses.py     one function per topic: the reference code
  app/charts.py       all matplotlib/seaborn plots
  app/registry.py     the topic catalogue: explanations and page settings
  app/pdf.py          builds the explanatory PDFs (reportlab)
  app/code_ref.py     source, file and line numbers for a function
  app/main.py         API routes and parameter validation
frontend/             React + Vite
notebooks/            the original 10.7 notebook
```

## Run

Ports 8001 and 5174, so it can run next to the data quality app (8000 / 5173).

Backend:

```bash
cd backend
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8001
```

Frontend (proxies `/api` to the backend):

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5174.

## API

| Route | Returns |
|---|---|
| `GET /api/topics` | all topics, grouped |
| `GET /api/topics/{id}?param=value…` | explanation, settings, result (metrics, tables, images), code references |
| `GET /api/topics/{id}/pdf?param=value…` | the explanatory PDF for those settings (`&download=true` to download it) |
| `GET /api/dataset` | column descriptions and first 10 rows |
| `GET /health` | `{"status": "ok"}` |

## How the pages evaluate

- Intuition pages report train and test scores on the notebook's split (80/20, stratified, `random_state=42`) and 5-fold stratified CV on all rows.
- Scaling and imputation sit inside the pipelines, so they are fitted on training data only.
- Thresholds and hyperparameters are chosen on the training split (out-of-fold or GridSearchCV); the test split is used once.
- Decision-boundary plots use two features and leave out rows where those features have an impossible 0.

## Adding a topic

1. Write a function in `backend/app/analyses.py` that returns `{message, metrics, tables, images}`.
2. Add a `Topic(...)` entry in `backend/app/registry.py`, with `params` for any page settings.
