/** Page parameters (column pickers, selects, toggles, numbers), stored in the URL query string. */
export default function Controls({ params, values, onChange, onReset, busy }) {
  if (!params.length) return null;
  return (
    <section className="card controls">
      <div className="controls-head">
        <h2>Settings</h2>
        {busy && <span className="muted small">Updating…</span>}
        <button className="link-button small" onClick={onReset}>Reset to defaults</button>
      </div>
      <div className="control-grid">
        {params.map((p) => {
          const value = values[p.name];
          const id = `p-${p.name}`;
          if (p.type === "bool")
            return (
              <label key={p.name} className="control control-bool" htmlFor={id}>
                <input id={id} type="checkbox" checked={!!value}
                  onChange={(e) => onChange(p.name, e.target.checked ? "true" : "false")} />
                {p.label}
              </label>
            );
          if (p.type === "number")
            return (
              <label key={p.name} className="control" htmlFor={id}>
                <span>{p.label}</span>
                <input id={id} type="number" step={p.step ?? "any"} min="0" placeholder="default"
                  defaultValue={value ?? ""} key={String(value)}
                  onBlur={(e) => onChange(p.name, e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onChange(p.name, e.target.value)} />
                {p.help && <small className="muted">{p.help}</small>}
              </label>
            );
          return (
            <label key={p.name} className="control" htmlFor={id}>
              <span>{p.label}</span>
              <select id={id} value={String(value)} onChange={(e) => onChange(p.name, e.target.value)}>
                {p.options.map((o) => <option key={String(o)} value={String(o)}>{String(o)}</option>)}
              </select>
            </label>
          );
        })}
      </div>
    </section>
  );
}
