export function StatusBadge({ status }) {
  const label = { pass: "Pass", warn: "Review", fail: "Issue" }[status] ?? status;
  return <span className={`badge badge-${status}`}>{label}</span>;
}

export function Loading({ state }) {
  if (state.loading) return <p className="muted">Loading…</p>;
  if (state.error)
    return (
      <div className="card error">
        Could not reach the backend ({state.error.message}). Is it running on port 8001?
      </div>
    );
  return null;
}

export function DataTable({ table, maxRows = 30 }) {
  const rows = table.rows.slice(0, maxRows);
  return (
    <div className="table-block">
      <h4>{table.title}</h4>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>{table.columns.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td key={j} className={typeof v === "number" ? "num" : String(v).length > 40 ? "text" : ""}>
                    {v === null ? <span className="muted">NaN</span> : String(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.rows.length > maxRows && (
        <p className="muted small">{table.rows.length - maxRows} more rows not shown</p>
      )}
    </div>
  );
}

/** Horizontal bar chart; bars at or above the threshold are highlighted. */
export function BarChart({ chart }) {
  if (!chart || !chart.labels.length) return null;
  const max = Math.max(100, ...chart.values);
  const { threshold, unit = "" } = chart;
  return (
    <figure className="chart">
      <figcaption>{chart.title}</figcaption>
      <div className="bars">
        {threshold != null && (
          <div className="threshold" style={{ left: `calc(var(--label-w) + (100% - var(--label-w) - var(--value-w)) * ${threshold / max})` }}>
            <span>threshold {threshold}{unit}</span>
          </div>
        )}
        {chart.labels.map((label, i) => {
          const v = chart.values[i];
          const hot = threshold != null && v >= threshold;
          return (
            <div className="bar-row" key={label}>
              <span className="bar-label" title={label}>{label}</span>
              <span className="bar-track">
                <span className={`bar ${hot ? "hot" : ""}`} style={{ width: `${(v / max) * 100}%` }} />
              </span>
              <span className="bar-value">{v}{unit}</span>
            </div>
          );
        })}
      </div>
    </figure>
  );
}
