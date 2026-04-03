import { BarChart, Bar, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, CartesianGrid, XAxis, YAxis } from "recharts";
import { AlertCircle, CheckCircle, AlertTriangle } from "lucide-react";
import "./AnalysisResults.css";

interface AnalysisResultsProps {
  data: any;
  onReset: () => void;
}

const toPercent = (val: number) => Math.round(Math.max(0, Math.min(1, val)) * 100);

const getRiskColor = (risk: string) => {
  switch (risk.toLowerCase()) {
    case "high":
      return "#ef4444";
    case "medium":
      return "#f59e0b";
    default:
      return "#22c55e";
  }
};

const getVerdictText = (risk: string) => {
  switch (risk.toLowerCase()) {
    case "high":
      return "🚨 Likely Manipulated";
    case "medium":
      return "⚠️ Needs Manual Review";
    default:
      return "✓ Likely Authentic";
  }
};

export default function AnalysisResults(props: AnalysisResultsProps) {
  const { data } = props;
  const analysis = data?.analysis ?? data ?? {};
  const scoring = analysis.scoring || {};
  const moduleScores = scoring.module_scores || {};
  const confidence = scoring.confidence_score || 0;
  const riskLevel = scoring.risk_level || "unknown";
  const blinkUnavailable = (analysis.behavioral?.blink_count ?? 0) === 0 && (analysis.behavioral?.blink_irregularity ?? 1) >= 0.95;
  const lipsyncUnavailable = (analysis.behavioral?.lipsync_error ?? 1) >= 0.99;

  // Prepare chart data
  const moduleNames: { [key: string]: string } = {
    blink: "👁️ Blink",
    lipsync: "🎬 Lip-Sync",
    visual: "📊 Visual",
    watermark: "🔍 Watermark",
    nlp: "🎤 Speech",
  };

  const chartData = Object.entries(moduleScores).map(([key, value]) => ({
    name: moduleNames[key] || key,
    value: toPercent(value as number),
  }));

  const COLORS = ["#6366f1", "#8b5cf6", "#ec4899", "#22c55e", "#f59e0b"];

  const pieLegendData = chartData.map((item, index) => ({
    ...item,
    color: COLORS[index % COLORS.length],
  }));

  const radarData = chartData.map((d) => ({
    subject: d.name.split(" ")[1] || d.name,
    A: d.value,
    fullMark: 100,
  }));

  return (
    <div className="results-container">
      {/* Header */}
      <div className="results-header">
        <h1>Analysis Complete ✅</h1>
        <p className="video-name">Video: {data.video_file}</p>
      </div>

      {/* Key Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">RISK ASSESSMENT</div>
          <div className="metric-badge" style={{ background: getRiskColor(riskLevel) }}>
            {riskLevel.toUpperCase()}
          </div>
          <div className="verdict-text">{getVerdictText(riskLevel)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">MODEL CONFIDENCE</div>
          <div className="metric-value" style={{ color: getRiskColor(riskLevel) }}>
            {toPercent(confidence)}%
          </div>
          <div className="confidence-bar">
            <div
              className="bar-fill"
              style={{
                width: `${toPercent(confidence)}%`,
                background: getRiskColor(riskLevel),
              }}
            ></div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">EVIDENCE QUALITY</div>
          <div className="metric-sublabel">
            {toPercent(confidence) >= 75 ? "🟢 High" : toPercent(confidence) >= 50 ? "🟡 Medium" : "🔴 Low"}
          </div>
          <p className="small-text">Based on available signals</p>
        </div>

        <div className="metric-card">
          <div className="metric-label">FACE DETECTION</div>
          <div className="metric-sublabel">
            {analysis.visual?.face_path ? "✓ Detected" : "✗ Not Found"}
          </div>
          <p className="small-text">Required for analysis</p>
        </div>
      </div>

      {/* Visualizations */}
      <div className="charts-section">
        <h2>📊 Module Analysis</h2>

        <div className="charts-grid">
          {/* Bar Chart */}
          <div className="chart-card">
            <h3>Suspicion Scores</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={-15} textAnchor="end" height={60} />
                <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={{ fontSize: 12 }} />
                <Bar dataKey="value" fill="#6366f1" radius={8} />
                <Tooltip formatter={(v) => `${v}%`} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Pie Chart */}
          <div className="chart-card">
            <h3>Risk Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={chartData} cx="50%" cy="50%" labelLine={false} label={false} outerRadius="72%" fill="#8884d8" dataKey="value">
                  {chartData.map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(v) => `${v}%`} />
              </PieChart>
            </ResponsiveContainer>
            <ul className="pie-legend" aria-label="Risk distribution legend">
              {pieLegendData.map((item) => (
                <li key={item.name} className="pie-legend-item">
                  <span className="pie-legend-swatch" style={{ backgroundColor: item.color }}></span>
                  <span className="pie-legend-label">{item.name}</span>
                  <strong className="pie-legend-value">{item.value}%</strong>
                </li>
              ))}
            </ul>
          </div>

          {/* Radar Chart */}
          <div className="chart-card full-width">
            <h3>Multi-Module Profile</h3>
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar name="Suspicion" dataKey="A" stroke="#6366f1" fill="#6366f1" fillOpacity={0.6} />
                <Tooltip formatter={(v) => `${v}%`} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Reasons */}
      <div className="reasons-section">
        <h2>🔍 Analysis Reasons</h2>
        {scoring.reasons && scoring.reasons.length > 0 ? (
          <div className="reasons-list">
            {scoring.reasons.map((reason: string, idx: number) => (
              <div key={idx} className="reason-item warning">
                <AlertTriangle size={18} />
                <span>{reason}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="reason-item success">
            <CheckCircle size={18} />
            <span>No major red flags detected</span>
          </div>
        )}
      </div>

      {/* Technical Details */}
      <div className="details-section">
        <h2>📋 Technical Details</h2>

        <div className="details-grid">
          <div className="detail-group">
            <h3>Behavioral Metrics</h3>
            <div className="detail-item">
              <span>Blink Events</span>
              <strong>{blinkUnavailable ? "N/A" : (analysis.behavioral?.blink_count || 0)}</strong>
            </div>
            <div className="detail-item">
              <span>Lip-Sync Score</span>
              <strong>{lipsyncUnavailable ? "N/A" : `${((analysis.behavioral?.lipsync_correlation || 0) * 100).toFixed(1)}%`}</strong>
            </div>
          </div>

          <div className="detail-group">
            <h3>Visual Metrics</h3>
            <div className="detail-item">
              <span>Lighting Asymmetry</span>
              <strong>{(analysis.visual?.lighting_asymmetry || 0).toFixed(2)}</strong>
            </div>
            <div className="detail-item">
              <span>Sharpness</span>
              <strong>{(analysis.visual?.sharpness_score || 0).toFixed(2)}</strong>
            </div>
          </div>

          <div className="detail-group">
            <h3>NLP Analysis</h3>
            <div className="detail-item">
              <span>Speech Segments</span>
              <strong>{analysis.transcript?.speech_segments || 0}</strong>
            </div>
            <div className="detail-item">
              <span>Transcript Method</span>
              <strong>{analysis.transcript?.method || "Unknown"}</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="recommendations-section">
        <h2>💡 Recommendations</h2>

        {riskLevel.toLowerCase() === "high" && (
          <div className="alert danger">
            <AlertCircle size={20} />
            <div>
              <strong>High Risk - Take Action</strong>
              <p>Do NOT trust this content without independent verification. Consider reporting to content platform.</p>
            </div>
          </div>
        )}

        {riskLevel.toLowerCase() === "medium" && (
          <div className="alert warning">
            <AlertTriangle size={20} />
            <div>
              <strong>Medium Risk - Verify Carefully</strong>
              <p>Cross-check with multiple sources and look for corroborating evidence before sharing.</p>
            </div>
          </div>
        )}

        {riskLevel.toLowerCase() === "low" && (
          <div className="alert success">
            <CheckCircle size={20} />
            <div>
              <strong>Low Risk - Likely Authentic</strong>
              <p>No strong manipulation signals detected, but always maintain healthy skepticism.</p>
            </div>
          </div>
        )}

        <ul className="recommendations-list">
          <li>✓ Check original source for any platform disclaimers</li>
          <li>✓ Test multiple clips from same source for consistency</li>
          <li>✓ Look for forensic metadata in file properties</li>
          <li>✓ Verify speaker against known samples</li>
          <li>✓ Consider broader context: timing, motive, corroborating evidence</li>
        </ul>
      </div>

      {/* Action Buttons */}
      <div className="action-buttons">
        <button className="button primary" onClick={props.onReset}>
          📹 Analyze Another Video
        </button>
        <button className="button secondary" onClick={() => alert("Download feature coming soon!")}>
          📥 Download Report
        </button>
      </div>
    </div>
  );
}
