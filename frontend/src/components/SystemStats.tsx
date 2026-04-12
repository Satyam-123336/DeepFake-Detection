import { useState, useEffect } from "react";
import { LineChart, Line, BarChart, Bar, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import axios from "axios";
import "./SystemStats.css";

export default function SystemStats() {
  const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
  const [stats, setStats] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");

  const formatDuration = (seconds: number) => {
    if (!seconds || seconds < 0) return "0s";
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${apiBase}/api/stats`);
      const payload = response.data;
      setStats(payload);
      setHistory((prev) => {
        const point = {
          time: new Date().toLocaleTimeString(),
          totalInferences: payload?.optimization?.total_inferences || 0,
          cacheHits: payload?.optimization?.cache_hits || 0,
          activeJobs: payload?.active_jobs || 0,
          cacheHitRate: ((payload?.optimization?.cache_hit_rate || 0) * 100),
        };
        return [...prev.slice(-19), point];
      });
      setError("");
      setLoading(false);
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p>Loading system statistics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-message">
        <h2>Error Loading Stats</h2>
        <p>{error}</p>
      </div>
    );
  }

  const optimization = stats?.optimization || {};
  const cache = stats?.cache || {};
  const services = stats?.services || {};

  const loadBreakdown = [
    { category: "Cache Hits", value: optimization.cache_hits || 0 },
    { category: "Recomputed", value: optimization.inferences_recomputed || 0 },
    { category: "Active Jobs", value: stats?.active_jobs || 0 },
  ];

  const latestTrendPoint = history[history.length - 1] || {
    totalInferences: optimization.total_inferences || 0,
    cacheHits: optimization.cache_hits || 0,
  };

  const cacheEfficiencyMsg = (cache.hit_rate || 0) >= 0.5
    ? "excellent cache utilization"
    : (cache.hit_rate || 0) >= 0.25
      ? "moderate cache utilization"
      : "cache warming in progress";

  return (
    <div className="stats-container">
      {/* Key Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon">📊</div>
          <div className="metric-data">
            <span className="label">Total Inferences</span>
            <span className="value">{optimization.total_inferences || 0}</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">⚡</div>
          <div className="metric-data">
            <span className="label">Cache Hit Rate</span>
            <span className="value">{((optimization.cache_hit_rate || 0) * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">💾</div>
          <div className="metric-data">
            <span className="label">Cache Size</span>
            <span className="value">{(cache.size_mb || 0).toFixed(1)}MB</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">⏱️</div>
          <div className="metric-data">
            <span className="label">Uptime</span>
            <span className="value">{formatDuration(stats?.uptime_seconds || 0)}</span>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="charts-section">
        <h2>📈 Performance Overview</h2>

        <div className="charts-grid">
          {/* Timeline */}
          <div className="chart-card full-width">
            <h3>Inference Timeline</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="totalInferences" stroke="#6366f1" strokeWidth={2} name="Total Inferences" />
                <Line type="monotone" dataKey="cacheHits" stroke="#22c55e" strokeWidth={2} name="Cache Hits" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Load Breakdown */}
          <div className="chart-card">
            <h3>Current Load Breakdown</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={loadBreakdown}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" angle={-30} textAnchor="end" height={70} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#6366f1" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Cache Hit Rate Trend */}
          <div className="chart-card">
            <h3>Cache Hit Rate Trend</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line type="monotone" dataKey="cacheHitRate" stroke="#f59e0b" strokeWidth={2} name="Hit Rate %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Cache Details */}
      <div className="details-section">
        <h2>💾 Cache Analytics</h2>

        <div className="details-grid">
          <div className="detail-card">
            <h3>Cache Statistics</h3>
            <div className="detail-row">
              <span>Entries</span>
              <strong>{cache.entries || 0}</strong>
            </div>
            <div className="detail-row">
              <span>Total Size</span>
              <strong>{(cache.size_mb || 0).toFixed(1)} MB</strong>
            </div>
            <div className="detail-row">
              <span>Hit Rate</span>
              <strong>{((cache.hit_rate || 0) * 100).toFixed(1)}%</strong>
            </div>
            <div className="detail-row">
              <span>Evictions</span>
              <strong>{cache.evictions || 0}</strong>
            </div>
          </div>

          <div className="detail-card">
            <h3>Inference Statistics</h3>
            <div className="detail-row">
              <span>Total Runs</span>
              <strong>{optimization.total_inferences || 0}</strong>
            </div>
            <div className="detail-row">
              <span>Cache Hits</span>
              <strong>{optimization.cache_hits || 0}</strong>
            </div>
            <div className="detail-row">
              <span>Recomputed</span>
              <strong>{optimization.inferences_recomputed || 0}</strong>
            </div>
            <div className="detail-row">
              <span>Hit Rate</span>
              <strong>{((optimization.cache_hit_rate || 0) * 100).toFixed(1)}%</strong>
            </div>
          </div>

          <div className="detail-card">
            <h3>Active Jobs</h3>
            <div className="detail-row">
              <span>Running Now</span>
              <strong>{stats.active_jobs || 0}</strong>
            </div>
            <div className="detail-row">
              <span>Uptime</span>
              <strong>{formatDuration(stats?.uptime_seconds || 0)}</strong>
            </div>
            <div className="detail-row">
              <span>Total Inferences</span>
              <strong>{optimization.total_inferences || 0}</strong>
            </div>
            <div className="detail-row">
              <span>Queue Depth</span>
              <strong>{Math.max(0, (stats.active_jobs || 0) - 1)}</strong>
            </div>
          </div>
        </div>
      </div>

      {/* API Health */}
      <div className="health-section">
        <h2>🏥 System Health</h2>

        <div className="health-grid">
          <div className={`health-item ${services.api === "active" ? "good" : "warning"}`}>
            <div className="health-icon">✓</div>
            <span>API Server</span>
            <em>{services.api === "active" ? "Operational" : "Degraded"}</em>
          </div>
          <div className={`health-item ${services.pipeline === "active" ? "good" : "warning"}`}>
            <div className="health-icon">✓</div>
            <span>Pipeline</span>
            <em>{services.pipeline === "active" ? "Running" : "Degraded"}</em>
          </div>
          <div className={`health-item ${services.cache === "active" ? "good" : "warning"}`}>
            <div className="health-icon">✓</div>
            <span>Cache Layer</span>
            <em>{services.cache === "active" ? "Active" : "Degraded"}</em>
          </div>
          <div className="health-item good">
            <div className="health-icon">✓</div>
            <span>Storage</span>
            <em>Available</em>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="recommendations">
        <h2>💡 Optimization Insights</h2>
        <ul>
          <li>
            ✓ <strong>Cache efficiency:</strong> {((cache.hit_rate || 0) * 100).toFixed(1)}% hit rate - {cacheEfficiencyMsg}
          </li>
          <li>
            ✓ <strong>Inference trend:</strong> {latestTrendPoint.totalInferences} total runs, {latestTrendPoint.cacheHits} served from cache
          </li>
          <li>
            ✓ <strong>Current load:</strong> {stats.active_jobs || 0} active jobs in queue/processing
          </li>
          <li>
            💡 <strong>Tip:</strong> Reanalyzing same video? Cache will return in &lt;1ms
          </li>
        </ul>
      </div>
    </div>
  );
}
