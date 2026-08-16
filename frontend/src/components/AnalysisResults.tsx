import {
  BarChart, Bar, ResponsiveContainer, PieChart, Pie, Cell, Tooltip,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  CartesianGrid, XAxis, YAxis,
} from "recharts";
import "./AnalysisResults.css";

interface AnalysisResultsProps {
  data: any;
  onReset: () => void;
}

const toPercent = (val: number) => Math.round(Math.max(0, Math.min(1, val)) * 100);

const getRiskClass = (risk: string) => {
  switch (risk.toLowerCase()) {
    case "high":   return "high";
    case "medium": return "medium";
    default:       return "low";
  }
};

const CHART_COLORS = ["#7c3aed", "#00d4aa", "#ef4444", "#f59e0b", "#10b981"];

const DONUT_COLORS: Record<string, string> = {
  blink:     "#7c3aed",
  lipsync:   "#0ea5e9",
  visual:    "#00d4aa",
  watermark: "#f59e0b",
  nlp:       "#10b981",
};

const MODULE_META: Record<string, { label: string; icon: string; weight: number }> = {
  blink:     { label: "Blink EAR",   icon: "👁️",  weight: 0.20 },
  lipsync:   { label: "Lip-Sync",    icon: "🎬",  weight: 0.25 },
  visual:    { label: "CNN Visual",  icon: "🔬",  weight: 0.35 },
  watermark: { label: "Watermark",   icon: "🔍",  weight: 0.10 },
  nlp:       { label: "NLP Speech",  icon: "🎤",  weight: 0.10 },
};

const CHART_NAMES: Record<string, string> = {
  blink:     "Blink",
  lipsync:   "Lip-Sync",
  visual:    "Visual",
  watermark: "Watermark",
  nlp:       "Speech",
};

// SVG Donut ring component
function DonutRing({ score, color }: { score: number; color: string }) {
  const r = 28;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;
  return (
    <div className="module-donut-wrap">
      <svg width="72" height="72" viewBox="0 0 72 72">
        <circle className="donut-track" cx="36" cy="36" r={r} strokeWidth="6" fill="none" />
        <circle
          className="donut-fill"
          cx="36" cy="36" r={r}
          strokeWidth="6"
          fill="none"
          stroke={color}
          strokeDasharray={circ}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="module-donut-label" style={{ color }}>{score}%</div>
    </div>
  );
}

// Tooltip style reads computed CSS variables so it works in both themes
const getDarkTooltipStyle = () => {
  const style = getComputedStyle(document.documentElement);
  return {
    background:   style.getPropertyValue("--tooltip-bg").trim()      || "#161b22",
    border:       `1px solid ${style.getPropertyValue("--tooltip-border").trim() || "#30363d"}`,
    borderRadius: 8,
    color:        style.getPropertyValue("--text-primary").trim()     || "#e6edf3",
    fontSize:     13,
  };
};
const getChartColors = () => {
  const s = getComputedStyle(document.documentElement);
  return {
    grid: s.getPropertyValue("--chart-grid").trim()  || "#21262d",
    tick: s.getPropertyValue("--chart-tick").trim()  || "#484f58",
  };
};

export default function AnalysisResults({ data, onReset }: AnalysisResultsProps) {
  const colors = getChartColors();
  const analysis = data?.analysis ?? data ?? {};
  const scoring = analysis.scoring || {};
  const moduleScores: Record<string, number> = scoring.module_scores || {};
  const confidence = scoring.confidence_score || 0;
  const riskLevel = scoring.risk_level || "unknown";
  const riskClass = getRiskClass(riskLevel);
  const reasonList: string[] = scoring.reasons || [];

  const blinkUnavailable = (analysis.behavioral?.blink_count ?? 0) === 0
    && (analysis.behavioral?.blink_irregularity ?? 1) >= 0.95;
  const lipsyncUnavailable = (analysis.behavioral?.lipsync_error ?? 1) >= 0.99;
  const transcriptUnavailable = (analysis.transcript?.method || "").toLowerCase() === "unavailable";

  const evidenceSignals: Record<string, boolean> = {
    "Face Detected":    Boolean(analysis.visual?.face_path),
    "Blink Analysis":   !blinkUnavailable,
    "Lip-Sync":         !lipsyncUnavailable,
    "Speech Transcript":!transcriptUnavailable,
    "Watermark Scan":   (analysis.watermark?.confidence || 0) > 0,
  };
  const availableCount = Object.values(evidenceSignals).filter(Boolean).length;
  const evidenceQuality = Math.round((availableCount / Object.keys(evidenceSignals).length) * 100);

  const confPct = toPercent(confidence);

  // Chart data
  const barData = Object.entries(moduleScores).map(([k, v]) => ({
    name: CHART_NAMES[k] || k,
    value: toPercent(v),
    color: DONUT_COLORS[k] || "#00d4aa",
  }));

  const pieData = Object.entries(moduleScores).map(([k, v]) => ({
    name: CHART_NAMES[k] || k,
    value: toPercent(v * (MODULE_META[k]?.weight ?? 0)),
    color: DONUT_COLORS[k] || "#00d4aa",
  }));

  const radarData = Object.entries(moduleScores).map(([k, v]) => ({
    subject: CHART_NAMES[k] || k,
    A: toPercent(v),
    fullMark: 100,
  }));

  const verdictMap: Record<string, string> = {
    high:   "Evidence of Manipulation Detected",
    medium: "Inconclusive — Manual Review Advised",
    low:    "No Strong Manipulation Indicators",
  };

  const handleExport = () => {
    const blob = new Blob([JSON.stringify({ data }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `realityguard_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="results-root">

      {/* ── Verdict Banner ── */}
      <div className={`verdict-banner ${riskClass}`}>
        <div className={`verdict-risk-badge ${riskClass}`}>
          <span className="verdict-risk-level">RISK</span>
          <span className="verdict-risk-word">{riskLevel.toUpperCase()}</span>
        </div>

        <div className="verdict-details">
          <div className="verdict-title">{verdictMap[riskLevel.toLowerCase()] ?? "Analysis Complete"}</div>
          <div className="verdict-file">📁 {data.video_file || "Unknown file"}</div>
          <div className="verdict-meta-row">
            <div className="verdict-meta-item">
              <span className="verdict-meta-label">Confidence</span>
              <span className="verdict-meta-value" style={{ color: riskClass === "high" ? "var(--red)" : riskClass === "medium" ? "var(--amber)" : "var(--green)" }}>
                {confPct}%
              </span>
            </div>
            <div className="verdict-meta-item">
              <span className="verdict-meta-label">Evidence Quality</span>
              <span className="verdict-meta-value" style={{ color: evidenceQuality >= 75 ? "var(--green)" : evidenceQuality >= 50 ? "var(--amber)" : "var(--red)" }}>
                {availableCount}/5 signals
              </span>
            </div>
            <div className="verdict-meta-item">
              <span className="verdict-meta-label">Modules</span>
              <span className="verdict-meta-value" style={{ color: "var(--teal)" }}>
                {Object.keys(moduleScores).length} active
              </span>
            </div>
            {data.completed_at && (
              <div className="verdict-meta-item">
                <span className="verdict-meta-label">Completed</span>
                <span className="verdict-meta-value" style={{ color: "var(--text-secondary)", fontSize: "0.82rem" }}>
                  {new Date(data.completed_at).toLocaleTimeString()}
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="verdict-actions">
          <button id="btn-new-analysis" className="btn btn-primary" onClick={onReset}>
            ⚡ New Analysis
          </button>
          <button id="btn-export-json" className="btn btn-secondary" onClick={handleExport}>
            📥 Export JSON
          </button>
        </div>
      </div>

      {/* ── Module Donut Grid ── */}
      <div>
        <div className="section-header">
          <div className="section-title">
            🔬 Module Suspicion Scores
            <span className="section-title-label">Per-Signal Breakdown</span>
          </div>
        </div>
        <div className="module-grid">
          {Object.entries(moduleScores).map(([key, val]) => {
            const meta = MODULE_META[key] || { label: key, icon: "⚙️", weight: 0 };
            const score = toPercent(val);
            const color = DONUT_COLORS[key] || "#00d4aa";
            return (
              <div key={key} className="module-score-card">
                <DonutRing score={score} color={color} />
                <div className="module-score-name">{meta.icon} {meta.label}</div>
                <div className="module-score-weight">weight {(meta.weight * 100).toFixed(0)}%</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Charts ── */}
      <div>
        <div className="section-header">
          <div className="section-title">
            📊 Visual Analysis
            <span className="section-title-label">Recharts Multi-View</span>
          </div>
        </div>
        <div className="charts-grid">
          {/* Bar Chart */}
          <div className="chart-card">
            <div className="chart-card-title">Suspicion Score by Module</div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={barData} margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: colors.tick }} />
                <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11, fill: colors.tick }} />
                <Tooltip
                  contentStyle={getDarkTooltipStyle()}
                  itemStyle={{ color: "var(--text-primary)" }}
                  formatter={(v) => [`${v}%`, "Suspicion"]}
                />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {barData.map((entry, idx) => (
                    <Cell key={`cell-${idx}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Pie Chart */}
          <div className="chart-card">
            <div className="chart-card-title">Weighted Score Contribution</div>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%" cy="50%"
                  outerRadius="70%"
                  dataKey="value"
                  labelLine={false}
                  label={false}
                >
                  {pieData.map((_, idx) => (
                    <Cell key={`pc-${idx}`} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={getDarkTooltipStyle()} 
                  itemStyle={{ color: "var(--text-primary)" }}
                  formatter={(v) => [`${v}%`]} 
                />
              </PieChart>
            </ResponsiveContainer>
            <ul className="pie-legend" aria-label="Risk distribution legend">
              {pieData.map((item, idx) => (
                <li key={item.name} className="pie-legend-item">
                  <span className="pie-legend-swatch" style={{ background: CHART_COLORS[idx % CHART_COLORS.length] }} />
                  <span className="pie-legend-label">{item.name}</span>
                  <span className="pie-legend-value">{item.value}%</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Radar Chart */}
          <div className="chart-card full-width">
            <div className="chart-card-title">Multi-Signal Forensic Profile Radar</div>
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={radarData}>
                <PolarGrid stroke={colors.grid} />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 12, fill: colors.tick }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10, fill: colors.tick }} />
                <Radar name="Suspicion" dataKey="A" stroke="#00d4aa" fill="#00d4aa" fillOpacity={0.15} strokeWidth={2} />
                <Tooltip
                  contentStyle={getDarkTooltipStyle()}
                  itemStyle={{ color: "var(--text-primary)" }}
                  formatter={(v) => [`${v}%`, "Suspicion"]}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ── Evidence Chain ── */}
      <div>
        <div className="section-header">
          <div className="section-title">
            🔗 Evidence Signal Chain
            <span className="section-title-label">Availability</span>
          </div>
        </div>
        <div className="evidence-chain">
          {Object.entries(evidenceSignals).map(([label, active]) => (
            <div key={label} className="evidence-item">
              <div className={`evidence-dot ${active ? "active" : "inactive"}`} />
              <span className="evidence-label">{label}</span>
              <span className={`evidence-status ${active ? "ok" : "na"}`}>
                {active ? "ACTIVE" : "N/A"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Findings ── */}
      <div>
        <div className="section-header">
          <div className="section-title">
            📋 Forensic Findings
            <span className="section-title-label">{reasonList.length} finding(s)</span>
          </div>
        </div>
        <div className="findings-list">
          {reasonList.length > 0 ? reasonList.map((r, i) => {
            const isFlag = !r.toLowerCase().includes("no single") && !r.toLowerCase().includes("no strong");
            return (
              <div key={i} className={`finding-item ${isFlag ? "flag" : "clear"}`}>
                <span className="finding-icon">{isFlag ? "⚠️" : "✓"}</span>
                <span className="finding-text">{r}</span>
              </div>
            );
          }) : (
            <div className="finding-item clear">
              <span className="finding-icon">✓</span>
              <span className="finding-text">No suspicious signals exceeded thresholds across all modules.</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Technical Details ── */}
      <div className="glass-panel" style={{ padding: 20 }}>
        <div className="section-header" style={{ marginBottom: 12 }}>
          <div className="section-title">
            🛠️ Technical Measurements
            <span className="section-title-label">Raw Metrics</span>
          </div>
        </div>
        <table className="detail-table">
          <thead>
            <tr>
              <th>Signal</th>
              <th>Key Metric</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Blink EAR</td>
              <td>Blink Count</td>
              <td>{blinkUnavailable ? "N/A" : (analysis.behavioral?.blink_count ?? 0)}</td>
            </tr>
            <tr>
              <td>Blink EAR</td>
              <td>Irregularity</td>
              <td>{blinkUnavailable ? "N/A" : ((analysis.behavioral?.blink_irregularity ?? 0) * 100).toFixed(1) + "%"}</td>
            </tr>
            <tr>
              <td>Lip-Sync</td>
              <td>Correlation Score</td>
              <td>{lipsyncUnavailable ? "N/A" : ((analysis.behavioral?.lipsync_correlation ?? 0) * 100).toFixed(1) + "%"}</td>
            </tr>
            <tr>
              <td>Visual</td>
              <td>Lighting Asymmetry</td>
              <td>{(analysis.visual?.lighting_asymmetry ?? 0).toFixed(3)}</td>
            </tr>
            <tr>
              <td>Visual</td>
              <td>Sharpness Score</td>
              <td>{(analysis.visual?.sharpness_score ?? 0).toFixed(2)}</td>
            </tr>
            <tr>
              <td>Watermark</td>
              <td>Confidence</td>
              <td>{((analysis.watermark?.confidence ?? 0) * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td>NLP Speech</td>
              <td>Speech Segments</td>
              <td>{transcriptUnavailable ? "N/A" : (analysis.transcript?.speech_segments ?? 0)}</td>
            </tr>
            <tr>
              <td>NLP Speech</td>
              <td>Transcript Method</td>
              <td>{transcriptUnavailable ? "N/A" : (analysis.transcript?.method || "Unknown")}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ── Recommendations ── */}
      <div className="recommendation-panel">
        <div className="section-title" style={{ marginBottom: 8 }}>
          💡 Investigator Recommendations
        </div>
        {riskLevel.toLowerCase() === "high" && (
          <div className="alert-banner danger" style={{ marginBottom: 12 }}>
            <span>🚨</span>
            <span><strong>High Manipulation Risk</strong> — Do NOT share this content without independent forensic verification. Consider escalating to platform authorities.</span>
          </div>
        )}
        {riskLevel.toLowerCase() === "medium" && (
          <div className="alert-banner warning" style={{ marginBottom: 12 }}>
            <span>⚠️</span>
            <span><strong>Inconclusive Result</strong> — Cross-validate with additional tools and corroborating physical evidence before making a determination.</span>
          </div>
        )}
        {riskLevel.toLowerCase() === "low" && (
          <div className="alert-banner success" style={{ marginBottom: 12 }}>
            <span>✅</span>
            <span><strong>Likely Authentic</strong> — No strong synthetic signals detected. Maintain standard chain-of-custody documentation.</span>
          </div>
        )}
        <ul className="recommendation-list">
          <li>Verify original source and metadata provenance</li>
          <li>Test multiple clips from the same subject</li>
          <li>Check file EXIf and codec fingerprints</li>
          <li>Cross-reference with known voice/video samples</li>
          <li>Consider environmental context and timeline</li>
          <li>Document analysis in chain-of-custody record</li>
        </ul>
      </div>

    </div>
  );
}
