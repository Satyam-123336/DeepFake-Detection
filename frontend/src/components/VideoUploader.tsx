import React, { useState, useRef } from "react";
import { Upload, AlertCircle, CheckCircle } from "lucide-react";
import axios from "axios";
import "./VideoUploader.css";

interface VideoUploaderProps {
  onAnalysisComplete: (data: any) => void;
  onJobCreated: (jobId: string) => void;
}

export default function VideoUploader(props: VideoUploaderProps) {
  const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [syncMode, setSyncMode] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    // Validate file
    if (!selectedFile.type.startsWith("video/")) {
      setError("Please select a valid video file");
      return;
    }

    if (selectedFile.size > 500 * 1024 * 1024) {
      setError("File size must be less than 500MB");
      return;
    }

    setError("");
    setFile(selectedFile);

    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(droppedFile);
      const event = {
        target: { files: dataTransfer.files },
      } as any;
      handleFileSelect(event);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const endpoint = syncMode
        ? `${apiBase}/api/analyze-sync`
        : `${apiBase}/api/analyze`;
      const response = await axios.post(endpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 180000,
      });

      if (syncMode) {
        // Sync mode - directly show results
        props.onAnalysisComplete(response.data);
      } else {
        // Async mode - track job
        const jobId = response.data.job_id;
        props.onJobCreated(jobId);
      }
    } catch (err: any) {
      if (axios.isAxiosError(err) && !err.response) {
        setError(
          `Cannot reach backend API at ${apiBase}. Ensure backend is running (uvicorn api_server:app --reload --port 8000).`
        );
      } else {
        setError(err.response?.data?.detail || "Upload failed. Please try again.");
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
    <div className="uploader-container">
      <div className="uploader-card">
        <h1>📹 Upload & Analyze Video</h1>
        <p className="subtitle">
          Upload a video to get instant deepfake risk assessment
        </p>

        {/* File Input Area */}
        {!preview ? (
          <div
            className="drop-zone"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileSelect}
              hidden
            />
            <Upload size={48} className="upload-icon" />
            <h2>Drag & drop your video here</h2>
            <p>or click to browse</p>
            <p className="supported">
              Supported: MP4, AVI, MOV, MKV, WebM (max 500MB)
            </p>
          </div>
        ) : (
          <div className="preview-section">
            <video src={preview} controls className="video-preview" />
            <div className="file-info">
              <CheckCircle size={24} className="check-icon" />
              <div>
                <strong>{file?.name}</strong>
                <p>{(file!.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            </div>
          </div>
        )}

        {/* Mode Selection */}
        {preview && (
          <div className="mode-selector">
            <label className="mode-option">
              <input
                type="radio"
                checked={!syncMode}
                onChange={() => setSyncMode(false)}
              />
              <span>
                <strong>Async (Recommended)</strong>
                <em>File queued for analysis, check progress</em>
              </span>
            </label>
            <label className="mode-option">
              <input
                type="radio"
                checked={syncMode}
                onChange={() => setSyncMode(true)}
              />
              <span>
                <strong>Sync (Direct)</strong>
                <em>Wait for results directly (slower for large files)</em>
              </span>
            </label>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="alert alert-error">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {/* Best Practices */}
        <div className="best-practices">
          <h3>💡 Best Practices</h3>
          <ul>
            <li>✓ Clear face position (avoid extreme angles)</li>
            <li>✓ Good lighting (avoid harsh shadows)</li>
            <li>✓ Audible speech (15+ second clips work best)</li>
            <li>✓ High quality video when possible</li>
          </ul>
        </div>

        {/* Action Buttons */}
        <div className="button-group">
          {preview ? (
            <>
              <button
                className="button primary"
                onClick={handleUpload}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner-inline"></span> Uploading...
                  </>
                ) : (
                  `▶️ Start ${syncMode ? "Sync" : "Async"} Analysis`
                )}
              </button>
              <button className="button secondary" onClick={handleReset}>
                ↺ Choose Different File
              </button>
            </>
          ) : (
            <button
              className="button primary"
              onClick={() => fileInputRef.current?.click()}
            >
              📁 Choose Video File
            </button>
          )}
        </div>
      </div>

      {/* Features Highlight */}
      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon">👁️</div>
          <h3>Blink Detection</h3>
          <p>Analyze eye blinking patterns for authenticity</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🎬</div>
          <h3>Lip-Sync Check</h3>
          <p>Verify audio-visual synchronization</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>Visual Analysis</h3>
          <p>Detect artifacts and synthesis traces</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🔍</div>
          <h3>Watermark Detection</h3>
          <p>Identify generation watermarks</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🎤</div>
          <h3>NLP Analysis</h3>
          <p>Analyze speech pattern authenticity</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📈</div>
          <h3>Risk Scoring</h3>
          <p>Get confidence-calibrated risk levels</p>
        </div>
      </div>
    </div>
  );
}
