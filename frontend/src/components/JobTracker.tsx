import { useState, useEffect, useRef } from "react";
import axios from "axios";
import AnalysisResults from "./AnalysisResults";
import "./JobTracker.css";

interface JobTrackerProps {
  jobId: string;
  onReset: () => void;
}

const PIPELINE_STAGES = [
  { id: "preprocess", label: "Preprocessing",    icon: "🎬", threshold: 20 },
  { id: "behavioral", label: "Behavioral\nAnalysis", icon: "👁️", threshold: 45 },
  { id: "visual",     label: "CNN Visual\nDetection", icon: "🔬", threshold: 70 },
  { id: "nlp",        label: "NLP / Watermark", icon: "🎤", threshold: 88 },
  { id: "scoring",    label: "Scoring",          icon: "📊", threshold: 100 },
];

function getStageState(stageIdx: number, progress: number, jobStatus: string) {
  const thresh = PIPELINE_STAGES[stageIdx].threshold;
  const prevThresh = stageIdx > 0 ? PIPELINE_STAGES[stageIdx - 1].threshold : 0;
  if (jobStatus === "completed") return "done";
  if (jobStatus === "failed")    return stageIdx === 0 ? "failed" : "queued";
  if (progress >= thresh)        return "done";
  if (progress >= prevThresh)    return "active";
  return "queued";
}

function formatElapsed(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export default function JobTracker({ jobId, onReset }: JobTrackerProps) {
  const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
  const [job, setJob] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(Date.now());

  useEffect(() => {
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await axios.get(`${apiBase}/api/jobs/${jobId}`);
        setJob(res.data);
        setLoading(false);
        if (res.data.error) setError(res.data.error);
      } catch (err: any) {
        setError(err.response?.data?.detail || "Failed to fetch job status.");
        setLoading(false);
      }
    };
    fetch();
    const interval = setInterval(() => {
      if (job?.status !== "completed" && job?.status !== "failed") fetch();
    }, 1500);
    return () => clearInterval(interval);
  }, [jobId, job?.status]);

  if (loading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16, padding: 80 }}>
        <div className="spinner" />
        <p style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>Connecting to engine...</p>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="tracker-error">
        <span style={{ fontSize: "2.5rem" }}>❌</span>
        <h2>Job Not Found</h2>
        <p style={{ color: "var(--text-secondary)" }}>This job may have expired or doesn't exist.</p>
        <button className="btn btn-primary" onClick={onReset}>← Back to Upload</button>
      </div>
    );
  }

  if (showResults && job.status === "completed" && job.result) {
    return <AnalysisResults data={job.result} onReset={onReset} />;
  }

  const progress = job.progress ?? 0;
  const riskLevel = job.result?.analysis?.scoring?.risk_level || "unknown";
  const confidence = job.result?.analysis?.scoring?.confidence_score || 0;
  const confPct = Math.round(Math.max(0, Math.min(1, confidence)) * 100);

  const statusClasses: Record<string, string> = {
    queued: "queued", processing: "processing", completed: "completed", failed: "failed",
  };

  return (
    <div className="tracker-root">
      {/* ── Main Card ── */}
      <div className="tracker-main-card">
        {/* Header */}
        <div className="tracker-header">
          <div className="tracker-header-left">
            <div className="tracker-header-label">Active Operation</div>
            <div className="tracker-job-id">#{jobId}</div>
            <div className="tracker-filename">📁 {job.filename || "Unknown file"}</div>
          </div>
          <div className={`job-status-badge ${statusClasses[job.status] ?? "queued"}`}>
            {job.status === "processing" && <div className="status-spinner" />}
            {job.status === "completed" && "✓"}
            {job.status === "failed"    && "✗"}
            {job.status === "queued"    && "⏳"}
            {job.status.toUpperCase()}
          </div>
        </div>

        {/* Pipeline Stages */}
        <div className="pipeline-diagram">
          <div className="pipeline-diagram-title">Analysis Pipeline</div>
          <div className="pipeline-stages">
            {PIPELINE_STAGES.map((stage, idx) => {
              const state = getStageState(idx, progress, job.status);
              return (
                <div key={stage.id} className={`pipeline-stage stage-${state}`}>
                  <div className={`stage-node ${state}`}>{stage.icon}</div>
                  <div className="stage-label">{stage.label}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Progress */}
        {job.status !== "completed" && job.status !== "failed" && (
          <div className="tracker-progress-section">
            <div className="progress-row">
              <span className="progress-label">Overall Progress</span>
              <span className="progress-pct">{progress}%</span>
            </div>
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        {/* Elapsed */}
        <div className="elapsed-timer">
          ⏱ Elapsed: <span className="elapsed-time">{formatElapsed(elapsed)}</span>
        </div>

        {/* Error */}
        {error && (
          <div className="alert-banner danger" style={{ margin: "0 24px 16px" }}>
            <span>⛔</span>
            <span>{error}</span>
          </div>
        )}

        {/* Completed Summary */}
        {job.status === "completed" && job.result && (
          <div className="completed-summary">
            <div className="summary-metrics">
              <div className="summary-metric">
                <div className="summary-metric-label">Risk Level</div>
                <div className="summary-metric-value" style={{
                  color: riskLevel === "high" ? "var(--red)" : riskLevel === "medium" ? "var(--amber)" : "var(--green)"
                }}>
                  {riskLevel.toUpperCase()}
                </div>
              </div>
              <div className="summary-metric">
                <div className="summary-metric-label">Confidence</div>
                <div className="summary-metric-value" style={{ color: "var(--teal)" }}>{confPct}%</div>
              </div>
              <div className="summary-metric">
                <div className="summary-metric-label">Duration</div>
                <div className="summary-metric-value" style={{ color: "var(--text-secondary)" }}>{formatElapsed(elapsed)}</div>
              </div>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="tracker-actions">
          {job.status === "completed" && job.result && (
            <button id="btn-view-report" className="btn btn-primary" onClick={() => setShowResults(true)} style={{ flex: 1 }}>
              📊 View Full Forensic Report
            </button>
          )}
          {job.status === "failed" && (
            <button className="btn btn-primary" onClick={onReset}>🔄 Try Again</button>
          )}
          <button id="btn-back-upload" className="btn btn-secondary" onClick={onReset}>
            ← New Analysis
          </button>
        </div>
      </div>

      {/* ── Right Sidebar ── */}
      <div className="tracker-sidebar">
        {/* Job Info */}
        <div className="info-card">
          <div className="info-card-title">Job Details</div>
          {[
            ["Job ID",   jobId],
            ["Created",  job.created_at ? new Date(job.created_at).toLocaleTimeString() : "—"],
            ["Status",   job.status],
            ["Progress", `${progress}%`],
          ].map(([k, v]) => (
            <div key={k} className="info-kv-row">
              <span className="info-kv-key">{k}</span>
              <span className="info-kv-val">{v}</span>
            </div>
          ))}
        </div>

        {/* Module Queue */}
        <div className="info-card">
          <div className="info-card-title">Module Execution Queue</div>
          {[
            { label: "Video Decode",     pct: Math.min(progress, 20),  max: 20  },
            { label: "Face Tracking",    pct: Math.min(progress, 45),  max: 45  },
            { label: "CNN Inference",    pct: Math.min(progress, 70),  max: 70  },
            { label: "NLP Transcript",   pct: Math.min(progress, 88),  max: 88  },
            { label: "Score Aggregate",  pct: Math.min(progress, 100), max: 100 },
          ].map((m) => {
            const local = Math.round((m.pct / m.max) * 100);
            return (
              <div key={m.label} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>{m.label}</span>
                  <span style={{ fontSize: "0.72rem", color: "var(--teal)", fontFamily: "var(--font-mono)", fontWeight: 700 }}>{local}%</span>
                </div>
                <div className="progress-bar-track" style={{ height: 3 }}>
                  <div className="progress-bar-fill" style={{ width: `${local}%` }} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Tips */}
        <div className="info-card">
          <div className="info-card-title">Field Notes</div>
          <ul className="tips-list">
            <li>Analysis runs entirely on local hardware — no data leaves your machine</li>
            <li>Videos &gt;100 MB may take 2–5 minutes</li>
            <li>Concurrent submissions are supported</li>
            <li>Jobs are retained for 7 days automatically</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
