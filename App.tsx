import { useState } from "react";
import VideoUploader from "./components/VideoUploader";
import AnalysisResults from "./components/AnalysisResults";
import JobTracker from "./components/JobTracker";
import SystemStats from "./components/SystemStats";
import "./App.css";

type AppMode = "upload" | "results" | "tracker" | "stats";

interface AnalysisData {
  job_id: string;
  video_file: string;
  completed_at: string;
  analysis: any;
  risk_color: string;
}

export default function App() {
  const [mode, setMode] = useState<AppMode>("upload");
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const handleAnalysisComplete = (data: AnalysisData) => {
    setAnalysisData(data);
    setMode("results");
  };

  const handleJobCreated = (id: string) => {
    setJobId(id);
    setMode("tracker");
  };

  const handleReset = () => {
    setAnalysisData(null);
    setJobId(null);
    setMode("upload");
  };

  return (
    <div className="app-wrapper">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo">🔍</div>
            <div className="logo-text">
              <h1>DeepFake Detection</h1>
              <p>Advanced AI Analysis Platform</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="nav-menu">
            <button
              className={`nav-btn ${mode === "upload" ? "active" : ""}`}
              onClick={() => setMode("upload")}
            >
              📹 Upload
            </button>
            <button
              className={`nav-btn ${mode === "tracker" ? "active" : ""}`}
              onClick={() => setMode("tracker")}
            >
              📊 Jobs
            </button>
            <button
              className={`nav-btn ${mode === "stats" ? "active" : ""}`}
              onClick={() => setMode("stats")}
            >
              ⚙️ System
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {mode === "upload" && (
          <VideoUploader
            onAnalysisComplete={handleAnalysisComplete}
            onJobCreated={handleJobCreated}
          />
        )}

        {mode === "results" && analysisData && (
          <AnalysisResults data={analysisData} onReset={handleReset} />
        )}

        {mode === "tracker" && (
          jobId ? (
            <JobTracker jobId={jobId} onReset={handleReset} />
          ) : (
            <div className="empty-state">
              <h2>No Active Job</h2>
              <p>Upload a video first to start async analysis and track its report.</p>
              <button className="button primary" onClick={() => setMode("upload")}>
                📹 Go to Upload
              </button>
            </div>
          )
        )}

        {mode === "stats" && <SystemStats />}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          DeepFake Detection AI • Powered by PyTorch & React •{" "}
          <span className="version">v1.0.0</span>
        </p>
      </footer>
    </div>
  );
}
