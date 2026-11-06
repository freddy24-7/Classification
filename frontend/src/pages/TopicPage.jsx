import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useApi, pdfUrl } from "../api.js";
import { BarChart, DataTable, Loading, StatusBadge } from "../components/ui.jsx";
import Controls from "../components/Controls.jsx";

function CodeBlock({ code }) {
  const [copied, setCopied] = useState(false);
  const vscode = `vscode://file${encodeURI(code.absolute_path)}:${code.start_line}`;
  const copy = () =>
    navigator.clipboard?.writeText(code.source).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  return (
    <>
      <p className="code-ref">
        <code>{code.file}</code>, lines {code.start_line}–{code.end_line}, function <code>{code.function}()</code>
      </p>
      <div className="links">
        <a href={vscode}>Open in VS Code</a>
        <button className="link-button" onClick={copy}>{copied ? "Copied" : "Copy code"}</button>
      </div>
      <pre className="code">
        {code.source.split("\n").slice(0, -1).map((line, i) => (
          <span className="code-line" key={i}>
            <span className="ln">{code.start_line + i}</span>
            {line || " "}
          </span>
        ))}
      </pre>
    </>
  );
}

function CodeReference({ refs }) {
  const [active, setActive] = useState(0);
  const current = refs[Math.min(active, refs.length - 1)];
  return (
    <section className="card">
      <h2>Reference code</h2>
      {refs.length > 1 && (
        <div className="tabs" role="tablist">
          {refs.map((r, i) => (
            <button key={r.function} role="tab" aria-selected={i === active}
              className={`tab ${i === active ? "active" : ""}`} onClick={() => setActive(i)}>
              {r.function}()
            </button>
          ))}
        </div>
      )}
      <CodeBlock code={current} />
    </section>
  );
}

function Plot({ im }) {
  return (
    <figure className="plot">
      <figcaption>{im.title}</figcaption>
      <img src={`data:image/png;base64,${im.png}`} alt={im.title} />
    </figure>
  );
}

function Metrics({ items }) {
  if (!items?.length) return null;
  return (
    <div className="metrics">
      {items.map((m) => (
        <div className="metric" key={m.label}>
          <div className="metric-value">{String(m.value)}</div>
          <div className="metric-label">{m.label}</div>
        </div>
      ))}
    </div>
  );
}

function Steps({ steps }) {
  return (
    <ol className="steps">
      {steps.map((s, i) => (
        <li className="card step" key={s.title} id={`step-${i + 1}`}>
          <div className="step-head">
            <span className="step-num">{i + 1}</span>
            <h3>{s.title}</h3>
          </div>
          <p>{s.text}</p>
          <p className="decision"><strong>Decision:</strong> {s.decision}</p>
          <Metrics items={s.metrics} />
          {s.images.map((im) => <Plot key={im.title} im={im} />)}
          {s.tables.map((t) => <DataTable key={t.title} table={t} />)}
        </li>
      ))}
    </ol>
  );
}

export default function TopicPage() {
  const { id } = useParams();
  const [search, setSearch] = useSearchParams();
  const query = search.toString();
  const state = useApi(`/api/topics/${id}${query ? `?${query}` : ""}`);
  const all = useApi("/api/topics");

  const c = state.data?.id === id ? state.data : null;
  if (!c) return <Loading state={state} />;
  const r = c.result;

  const setParam = (name, value) => {
    const next = new URLSearchParams(search);
    if (value === "" || value == null) next.delete(name);
    else next.set(name, value);
    setSearch(next, { replace: true });
  };

  const flat = all.data?.flatMap((g) => g.topics) ?? [];
  const idx = flat.findIndex((x) => x.id === id);
  const prev = idx > 0 ? flat[idx - 1] : null;
  const next = idx >= 0 && idx < flat.length - 1 ? flat[idx + 1] : null;

  return (
    <article className="check-page">
      <p className="crumbs">
        <Link to="/">All topics</Link> / {c.number}
      </p>
      <header className="check-title">
        <div>
          <span className="num-tag">{c.number}</span>
          <h1>{c.title}</h1>
          <p className="muted">{c.summary}</p>
        </div>
        <div className="pdf-actions">
          <a className="button" href={pdfUrl(c.id, query)} target="_blank" rel="noreferrer">Open explanatory PDF</a>
          <a className="link-small" href={pdfUrl(c.id, query, true)}>Download PDF</a>
        </div>
      </header>

      <div className="two-col">
        <section className="card">
          <h2>What it is</h2>
          <p>{c.what}</p>
          <h2>Why it matters</h2>
          <p>{c.why}</p>
          <h2>How it works</h2>
          <ul>{c.how.map((h) => <li key={h}>{h}</li>)}</ul>
        </section>
        <section className="card">
          <h2>{c.fix_title}</h2>
          <ul>{c.fix.map((h) => <li key={h}>{h}</li>)}</ul>
          {c.pitfalls.length > 0 && (
            <>
              <h2>Watch out for</h2>
              <ul>{c.pitfalls.map((h) => <li key={h}>{h}</li>)}</ul>
            </>
          )}
        </section>
      </div>

      <Controls params={c.params} values={c.values} busy={state.loading}
        onChange={setParam} onReset={() => setSearch({}, { replace: true })} />
      {state.error && <div className="card error">{state.error.message}</div>}

      <section className={`card ${state.loading ? "stale" : ""}`}>
        <div className="result-head">
          <h2>Result on the diabetes dataset</h2>
          {r.status && <StatusBadge status={r.status} />}
        </div>
        <p className="result-line">{r.message}</p>
        <Metrics items={r.metrics} />
        {r.steps && (
          <nav className="step-jump">
            {r.steps.map((s, i) => <a key={s.title} href={`#step-${i + 1}`}>{i + 1}. {s.title}</a>)}
          </nav>
        )}
        {r.flagged?.length > 0 && (
          <p>Flagged columns: {r.flagged.map((f) => <code className="flag" key={f}>{f}</code>)}</p>
        )}
        <BarChart chart={r.chart} />
        {r.images?.map((im) => <Plot key={im.title} im={im} />)}
        {r.tables.map((t) => <DataTable key={t.title} table={t} />)}
      </section>

      {r.steps && <div className={state.loading ? "stale" : ""}><Steps steps={r.steps} /></div>}

      <CodeReference refs={c.code} key={c.id} />

      <nav className="pager">
        {prev ? <Link to={`/topics/${prev.id}`}>← {prev.number} {prev.title}</Link> : <span />}
        {next ? <Link to={`/topics/${next.id}`}>{next.number} {next.title} →</Link> : <span />}
      </nav>
    </article>
  );
}
