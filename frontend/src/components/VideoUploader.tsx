import React, { useState, useRef } from "react";
import axios from "axios";
import "./VideoUploader.css";

interface VideoUploaderProps {
  onAnalysisComplete: (data: any) => void;
  onJobCreated: (jobId: string) => void;
  backendStatus: "checking" | "online" | "offline";
}

const MODULES = [
  { id: "blink",     icon: "👁️",  label: "Blink EAR",     desc: "Eye Aspect Ratio rhythm", cls: "icon-blink" },
  { id: "lipsync",   icon: "🎬",  label: "Lip-Sync",      desc: "Audio-visual alignment", cls: "icon-lipsync" },
  { id: "visual",    icon: "🔬",  label: "CNN Visual",    desc: "Artifact CNN inference",  cls: "icon-visual" },
  { id: "watermark", icon: "🔍",  label: "Watermark",     desc: "Synthetic trace scan",    cls: "icon-watermark" },
  { id: "nlp",       icon: "🎤",  label: "NLP Speech",    desc: "Speech pattern analysis", cls: "icon-nlp" },
];

export default function VideoUploader({ onAnalysisComplete, onJobCreated, backendStatus }: VideoUploaderProps) {
  const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [syncMode, setSyncMode] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validate = (f: File): string => {
    if (!f.type.startsWith("video/")) return "Please select a valid video file (MP4, AVI, MOV, MKV, WebM).";
    if (f.size > 500 * 1024 * 1024) return "File size must be less than 500 MB.";
    return "";
  };

  const applyFile = (f: File) => {
    const err = validate(f);
    if (err) { setError(err); return; }
    setError("");
    setFile(f);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result as string);
    reader.readAsDataURL(f);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) applyFile(f);
  };

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = () => setIsDragOver(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) applyFile(f);
  };

  const handleUpload = async () => {
    if (!file) { setError("Please select a video file first."); return; }
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const endpoint = syncMode ? `${apiBase}/api/analyze-sync` : `${apiBase}/api/analyze`;
      const response = await axios.post(endpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 240000,
      });
      if (syncMode) {
        onAnalysisComplete(response.data);
      } else {
        onJobCreated(response.data.job_id);
      }
    } catch (err: any) {
      if (axios.isAxiosError(err) && !err.response) {
        setError(`Cannot reach backend at ${apiBase}. Ensure the API server is running.`);
      } else {
        setError(err.response?.data?.detail || "Upload failed. Please retry.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview("");
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="uploader-root">
      {/* ── Left Sidebar ── */}
      <aside className="module-sidebar">
        <div className="sidebar-section-label">Detection Modules</div>
        {MODULES.map((m) => (
          <div key={m.id} className="module-check-item">
            <div className={`module-check-icon ${m.cls}`}>{m.icon}</div>
            <div className="module-check-label">
              <span className="module-check-name">{m.label}</span>
              <span className="module-check-desc">{m.desc}</span>
            </div>
            <span className="module-check-status status-ready">READY</span>
          </div>
        ))}

        <div className="sidebar-section-label" style={{ marginTop: 8 }}>Model Info</div>
        <div className="model-info-card">
          {[
            ["Accuracy",   "97.42%"],
            ["Threshold",  "0.285"],
            ["Train Samples", "34.8K"],
            ["Architecture", "CNN-4"],
            ["Input",      "128×128"],
          ].map(([k, v]) => (
            <div key={k} className="model-info-row">
              <span className="model-info-key">{k}</span>
              <span className="model-info-val">{v}</span>
            </div>
          ))}
        </div>
      </aside>

      {/* ── Right Upload Panel ── */}
      <div className="upload-panel">
        <div className="upload-panel-header">
          <h1 className="upload-panel-title">Submit Evidence</h1>
          <p className="upload-panel-subtitle">
            Upload a video for multi-signal forensic deepfake analysis. All processing runs locally.
          </p>
        </div>

        {/* Backend offline warning */}
        {backendStatus === "offline" && (
          <div className="alert-banner danger upload-alert">
            <span>⚠️</span>
            <span>Backend engine is offline. Start the API server: <code style={{ fontFamily: "var(--font-mono)", fontSize: "0.8em" }}>python api_server.py</code></span>
          </div>
        )}

        {/* Drop zone or preview */}
        {!preview ? (
          <div
            id="drop-zone"
            className={`scan-drop-zone ${isDragOver ? "drag-over" : "scan-active"}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            aria-label="Click or drag to upload video"
            onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
          >
            <div className="scan-corner scan-corner-tl" />
            <div className="scan-corner scan-corner-tr" />
            <div className="scan-corner scan-corner-bl" />
            <div className="scan-corner scan-corner-br" />
            <input
              ref={fileInputRef}
              type="file"
              id="video-file-input"
              accept="video/*"
              onChange={handleFileChange}
              hidden
            />
            <div className="drop-zone-content">
              <div className="drop-zone-icon">📡</div>
              <h2 className="drop-zone-title">Drag & Drop Evidence</h2>
              <p className="drop-zone-sub">or click to browse your filesystem</p>
              <p className="drop-zone-formats">MP4 · AVI · MOV · MKV · WEBM · MAX 500 MB</p>
            </div>
          </div>
        ) : (
          <div className="preview-container">
            <video src={preview} controls className="preview-video" />
            <div className="preview-meta">
              <span className="preview-filename">{file?.name}</span>
              <span className="preview-size">{((file?.size ?? 0) / 1024 / 1024).toFixed(2)} MB</span>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="alert-banner danger upload-alert">
            <span>⛔</span>
            <span>{error}</span>
          </div>
        )}

        {/* Mode selection */}
        {preview && (
          <div className="mode-toggle-row">
            <button
              id="mode-async"
              className={`mode-toggle-btn ${!syncMode ? "selected" : ""}`}
              onClick={() => setSyncMode(false)}
            >
              <span className="mode-label">⚡ Async Queue</span>
              <span className="mode-desc">Submit & track job progress in real time</span>
            </button>
            <button
              id="mode-sync"
              className={`mode-toggle-btn ${syncMode ? "selected" : ""}`}
              onClick={() => setSyncMode(true)}
            >
              <span className="mode-label">⏳ Sync Direct</span>
              <span className="mode-desc">Wait for full result immediately</span>
            </button>
          </div>
        )}

        {/* Loading indicator */}
        {loading && <div className="upload-progress-bar"><div className="upload-progress-fill" /></div>}

        {/* Action buttons */}
        <div className="action-row">
          {preview ? (
            <>
              <button
                id="btn-submit"
                className="btn btn-primary"
                onClick={handleUpload}
                disabled={loading || backendStatus === "offline"}
                style={{ flex: 1 }}
              >
                {loading ? (
                  <><span className="spinner spinner-sm" /> Submitting...</>
                ) : (
                  `🔍 Submit for Analysis`
                )}
              </button>
              <button id="btn-reset" className="btn btn-secondary" onClick={handleReset}>
                ↺ Clear
              </button>
            </>
          ) : (
            <button
              id="btn-browse"
              className="btn btn-primary"
              onClick={() => fileInputRef.current?.click()}
            >
              📁 Browse Files
            </button>
          )}
        </div>

        {/* Best practices */}
        <div className="best-practices">
          <div className="best-practices-title">📋 Evidence Quality Guidelines</div>
          <ul className="best-practices-list">
            <li>Clear frontal face positioning</li>
            <li>Adequate lighting conditions</li>
            <li>Audible speech for NLP/lip-sync</li>
            <li>High resolution preferred</li>
            <li>15+ seconds duration optimal</li>
            <li>Avoid extreme camera angles</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
