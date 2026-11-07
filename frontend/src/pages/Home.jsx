import { Link } from "react-router-dom";
import { useApi, pdfUrl } from "../api.js";
import { Loading, DataTable } from "../components/ui.jsx";

const INTROS = {
  data: "What the models will learn from: class balance, how the features differ between the classes, and zeros that are really missing values.",
  intuition: "How each classifier makes a decision, with its boundary drawn on two features. Change the settings to watch it underfit and overfit.",
  compare: "All models on the same cross-validation folds and metrics, next to a baseline that always predicts the majority class.",
  metrics: "Measuring a classifier: the confusion matrix, ROC and precision–recall curves, choosing a threshold, and whether the probabilities can be trusted.",
  handson: "Notebook 10_7_classification_hands_on.ipynb, run live. The default settings reproduce the notebook's numbers.",
  workflow: "Everything above chained into one project, the way an analyst would typically run it: frame, compare, shortlist, tune, choose, test once, report. Then the same with scikit-learn's automated tools.",
  extras: "Steps a real project adds after the notebook: fixing the hidden missing values, tuning, diagnosing over- and underfitting, and explaining the model.",
};

function TopicCard({ t }) {
  return (
    <li className="card check-card">
      <div className="check-head">
        <span className="num-tag">{t.number}</span>
      </div>
      <h3>
        <Link to={`/topics/${t.id}`}>{t.title}</Link>
      </h3>
      <p className="muted">{t.summary}</p>
      <div className="links">
        <Link to={`/topics/${t.id}`}>Explanation and code →</Link>
        <a href={pdfUrl(t.id)} target="_blank" rel="noreferrer">PDF</a>
      </div>
    </li>
  );
}

export default function Home() {
  const groups = useApi("/api/topics");
  const dataset = useApi("/api/dataset");

  return (
    <>
      <section className="hero">
        <h1>Classification models and how to evaluate them</h1>
        <p>
          Course sections 10.2–10.7, run live on the Pima Indians Diabetes dataset. Every topic has a page with an
          explanation, results you can explore, the reference code, and an explanatory PDF.
        </p>
        {groups.data && (
          <nav className="jump">
            {groups.data.map((g) => <a key={g.id} href={`#${g.id}`}>{g.title}</a>)}
          </nav>
        )}
      </section>

      <Loading state={groups} />
      {groups.data?.map((g) => (
        <section key={g.id} id={g.id} className="group">
          <h2 className="group-title">{g.title}</h2>
          <p className="muted group-intro">{INTROS[g.id]}</p>
          <ol className="check-grid">
            {g.topics.map((t) => <TopicCard key={t.id} t={t} />)}
          </ol>
        </section>
      ))}

      {dataset.data && (
        <section className="card">
          <h2>The dataset</h2>
          <p className="muted">
            {dataset.data.name}: {dataset.data.rows} rows, {dataset.data.columns.length} columns. Outcome is the target.
          </p>
          <DataTable
            table={{
              title: "Columns",
              columns: ["column", "description"],
              rows: Object.entries(dataset.data.descriptions),
            }}
          />
          <DataTable
            table={{
              title: "First 10 rows",
              columns: dataset.data.columns,
              rows: dataset.data.head.map((r) => dataset.data.columns.map((c) => r[c])),
            }}
          />
        </section>
      )}
    </>
  );
}
