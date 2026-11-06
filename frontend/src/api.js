import { useEffect, useState } from "react";

const withQuery = (path, query) => (query ? `${path}?${query}` : path);

export const pdfUrl = (id, query = "", download = false) =>
  withQuery(`/api/topics/${id}/pdf`, [query, download ? "download=true" : ""].filter(Boolean).join("&"));

/** Fetch JSON from the backend: returns { data, error, loading }. Keeps the previous data while reloading. */
export function useApi(path) {
  const [state, setState] = useState({ data: null, error: null, loading: true });

  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, error: null, loading: true }));
    fetch(path)
      .then(async (r) => {
        if (r.ok) return r.json();
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || `${r.status} ${r.statusText}`);
      })
      .then((data) => !cancelled && setState({ data, error: null, loading: false }))
      .catch((error) => !cancelled && setState((s) => ({ ...s, error, loading: false })));
    return () => {
      cancelled = true;
    };
  }, [path]);

  return state;
}
