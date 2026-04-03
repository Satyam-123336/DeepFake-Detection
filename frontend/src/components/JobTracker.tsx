import { useState, useEffect } from "react";
import { CheckCircle, AlertCircle, Clock, XCircle } from "lucide-react";
import axios from "axios";
import AnalysisResults from "./AnalysisResults";
import "./JobTracker.css";

interface JobTrackerProps {
  jobId: string;
  onReset: () => void;
}

export default function JobTracker(props: JobTrackerProps) {
  const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
  const [job, setJob] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    fetchJobStatus();
    if (job?.status === "completed" || job?.status === "failed") {
      return;
    }
    const interval = setInterval(fetchJobStatus, 1000);
    return () => clearInterval(interval);
  }, [props.jobId, job?.status]);

  const fetchJobStatus = async () => {
    try {
      const response = await axios.get(`${apiBase}/api/jobs/${props.jobId}`);
      setJob(response.data);
      setLoading(false);

      if (response.data.error) {
        setError(response.data.error);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to fetch job status");
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle size={24} className="status-completed" />;
      case "failed":
        return <XCircle size={24} className="status-failed" />;
      case "processing":
        return <Clock size={24} className="status-processing" />;
      default:
        return <Clock size={24} className="status-queued" />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case "completed":
        return "✓ Completed";
      case "failed":
        return "✗ Failed";
      case "processing":
        return "⏳ Processing";
      case "queued":
        return "📋 Queued";
      default:
        return "Unknown";
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading job status...</p>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="error-container">
        <AlertCircle size={48} />
        <h2>Job Not Found</h2>
        <p>The job you're looking for doesn't exist or has expired.</p>
        <button className="button primary" onClick={props.onReset}>
          ← Back to Upload
        </button>
      </div>
    );
  }

  if (showResults && job.status === "completed" && job.result) {
    return <AnalysisResults data={job.result} onReset={props.onReset} />;
  }

  return (
    <div className="tracker-container">
      <div className="tracker-card">
        <div className="job-header">
          <div className="job-status">
            {getStatusIcon(job.status)}
            <div>
              <h1>{getStatusText(job.status)}</h1>
              <p>Job ID: {job.id}</p>
            </div>
          </div>
        </div>

        {error && (
          <div className="alert danger">
            <AlertCircle size={20} />
            <div>
              <strong>Error</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        <div className="job-details">
          <div className="detail">
            <span>File</span>
            <strong>{job.filename}</strong>
          </div>
          <div className="detail">
            <span>Created</span>
            <strong>{new Date(job.created_at).toLocaleTimeString()}</strong>
          </div>
          <div className="detail">
            <span>Progress</span>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${job.progress}%` }}></div>
              <span className="progress-text">{job.progress}%</span>
            </div>
          </div>
        </div>

        {job.status === "processing" && (
          <div className="processing-info">
            <h3>What's happening:</h3>
            <ul>
              <li>{job.progress < 25 ? "🔄" : "✓"} Preprocessing video</li>
              <li>{job.progress < 50 ? "🔄" : "✓"} Analyzing behavior</li>
              <li>{job.progress < 75 ? "🔄" : "✓"} Visual detection</li>
              <li>{job.progress < 90 ? "🔄" : "✓"} Computing results</li>
            </ul>
          </div>
        )}

        {job.status === "completed" && job.result && (
          <div className="result-preview">
            <h3>📊 Quick Summary</h3>
            <div className="summary-grid">
              <div className="summary-item">
                <span>Risk Level</span>
                <strong style={{ color: job.result.risk_color }}>
                  {job.result.analysis?.scoring?.risk_level?.toUpperCase() || "UNKNOWN"}
                </strong>
              </div>
              <div className="summary-item">
                <span>Confidence</span>
                <strong>
                  {Math.round((job.result.analysis?.scoring?.confidence_score || 0.5) * 100)}%
                </strong>
              </div>
              <div className="summary-item">
                <span>Face Detected</span>
                <strong>
                  {job.result.analysis?.visual?.face_path ? "✓ Yes" : "✗ No"}
                </strong>
              </div>
            </div>

            <button
              className="button primary"
              onClick={() => setShowResults(!showResults)}
            >
              View Detailed Report
            </button>
          </div>
        )}

        <div className="button-group">
          {job.status === "failed" && (
            <button className="button primary" onClick={props.onReset}>
              🔄 Try Again
            </button>
          )}
          <button className="button secondary" onClick={props.onReset}>
            ← Back to Upload
          </button>
        </div>
      </div>

      {/* Tips */}
      <div className="tips-box">
        <h3>💡 Pro Tips</h3>
        <ul>
          <li>You can check job status anytime by bookmarking this URL</li>
          <li>Jobs are stored for 7 days before automatic cleanup</li>
          <li>Large videos (&gt;100MB) may take 2-5 minutes to process</li>
          <li>Multiple concurrent uploads are supported</li>
        </ul>
      </div>
    </div>
  );
}
