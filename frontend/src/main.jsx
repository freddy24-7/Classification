import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home.jsx";
import TopicPage from "./pages/TopicPage.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import "./styles.css";

function App() {
  return (
    <BrowserRouter>
      <header className="topbar">
        <Link to="/" className="brand">
          <img src="/favicon.svg" alt="" width="22" height="22" /> Classification Lab
        </Link>
        <span className="muted">Sections 10.2–10.7 · diabetes.csv</span>
      </header>
      <main className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/topics/:id" element={<TopicPage />} />
          <Route path="*" element={<p>Page not found. <Link to="/">Back to all topics</Link></p>} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
