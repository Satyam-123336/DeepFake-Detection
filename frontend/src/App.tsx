import { useState, useEffect } from "react";
import VideoUploader from "./components/VideoUploader";
import AnalysisResults from "./components/AnalysisResults";
import JobTracker from "./components/JobTracker";
import SystemStats from "./components/SystemStats";
import "./App.css";

type AppMode = "analyze" | "results" | "operations" | "analytics";
type BackendStatus = "checking" | "online" | "offline";
type Theme = "dark" | "light";

interface AnalysisData {
  job_id?: string;
  video_file?: string;
  completed_at?: string;
  analysis: any;
  risk_color?: string;
}

export default function App() {
  const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
  const [mode, setMode] = useState<AppMode>("analyze");
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");

  // ── Theme ──────────────────────────────────────────────────
  const getInitialTheme = (): Theme => {
    const saved = localStorage.getItem("rg-theme") as Theme | null;
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  };
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("rg-theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === "dark" ? "light" : "dark");

  // ── Backend Health ─────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${apiBase}/health`, { signal: AbortSignal.timeout(3000) });
        setBackendStatus(res.ok ? "online" : "offline");
      } catch {
        setBackendStatus("offline");
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, [apiBase]);

  const handleAnalysisComplete = (data: AnalysisData) => {
    setAnalysisData(data);
    setMode("results");
  };

  const handleJobCreated = (id: string) => {
    setJobId(id);
    setMode("operations");
  };

  const handleReset = () => {
    setAnalysisData(null);
    setJobId(null);
    setMode("analyze");
  };

  const statusLabel: Record<BackendStatus, string> = {
    checking: "CHECKING",
    online:   "ENGINE ONLINE",
    offline:  "ENGINE OFFLINE",
  };

  return (
    <div className="app-wrapper hex-bg">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-inner">
          {/* Logo */}
          <div className="logo-block">
            <div className="logo-icon">🔬</div>
            <div className="logo-text">
              <div className="logo-name">
                <span>Reality</span>Guard AI
              </div>
              <div className="logo-tagline">Multi-Signal Deepfake Detection Engine</div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="nav-menu">
            <button
              id="nav-analyze"
              className={`nav-tab ${mode === "analyze" || mode === "results" ? "active" : ""}`}
              onClick={() => setMode("analyze")}
            >
              ⚡ Analyze
            </button>
            <button
              id="nav-operations"
              className={`nav-tab ${mode === "operations" ? "active" : ""}`}
              onClick={() => setMode("operations")}
            >
              📋 Operations
            </button>
            <button
              id="nav-analytics"
              className={`nav-tab ${mode === "analytics" ? "active" : ""}`}
              onClick={() => setMode("analytics")}
            >
              📊 Analytics
            </button>
          </nav>

          {/* Right controls */}
          <div className="header-right">
            {/* Backend status */}
            <div className={`status-badge ${backendStatus}`}>
              <div className={`status-dot ${backendStatus === "online" ? "pulse" : ""}`} />
              <span>{statusLabel[backendStatus]}</span>
            </div>

            {/* Theme toggle */}
            <button
              id="btn-theme-toggle"
              className="theme-toggle"
              onClick={toggleTheme}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? "☀️" : "🌙"}
            </button>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="app-main">
        {mode === "analyze" && (
          <VideoUploader
            onAnalysisComplete={handleAnalysisComplete}
            onJobCreated={handleJobCreated}
            backendStatus={backendStatus}
          />
        )}

        {mode === "results" && analysisData && (
          <AnalysisResults data={analysisData} onReset={handleReset} />
        )}

        {mode === "operations" && (
          jobId ? (
            <JobTracker jobId={jobId} onReset={handleReset} />
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">📋</div>
              <h2>No Active Operations</h2>
              <p>Submit a video for analysis first. Async jobs will appear here for real-time tracking.</p>
              <button className="btn btn-primary" onClick={() => setMode("analyze")}>
                ⚡ Go to Analyze
              </button>
            </div>
          )
        )}

        {mode === "analytics" && <SystemStats />}
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <span>RealityGuard AI</span> · Multi-Signal Explainable Deepfake Detection Engine · Powered by PyTorch + MediaPipe · v1.0.0
      </footer>
    </div>
  );
}
