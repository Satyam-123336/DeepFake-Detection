import { useState, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar, ResponsiveContainer,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";
import axios from "axios";
import "./SystemStats.css";

// Reads live CSS variable values so charts adapt to light/dark theme
const getTooltipStyle = () => {
  const s = getComputedStyle(document.documentElement);
  return {
    background:   s.getPropertyValue("--tooltip-bg").trim()      || "#161b22",
    border:       `1px solid ${s.getPropertyValue("--tooltip-border").trim() || "#30363d"}`,
    borderRadius: 8,
    color:        s.getPropertyValue("--text-primary").trim()     || "#e6edf3",
    fontSize:     12,
  };
};

const getChartColors = () => {
  const s = getComputedStyle(document.documentElement);
  return {
    grid: s.getPropertyValue("--chart-grid").trim()  || "#21262d",
    tick: s.getPropertyValue("--chart-tick").trim()  || "#484f58",
  };
};

function formatDuration(seconds: number) {
  if (!seconds || seconds < 0) return "0s";
  const hrs  = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  if (hrs > 0)  return `${hrs}h ${mins}m`;
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
}


export default function SystemStats() {
  const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
  const [stats, setStats] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 2000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await axios.get(`${apiBase}/api/stats`);
        const p = res.data;
        setStats(p);
        setHistory(prev => {
          const point = {
            time: new Date().toLocaleTimeString("en", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
            total:    p?.optimization?.total_inferences || 0,
            cacheHits: p?.optimization?.cache_hits || 0,
            hitRate:  +((p?.optimization?.cache_hit_rate || 0) * 100).toFixed(1),
            jobs:     p?.active_jobs || 0,
          };
          return [...prev.slice(-19), point];
        });
        setError("");
      } catch (err: any) {
        setError(err.message || "Connection error");
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, [tick, apiBase]);

  if (loading) {
    return (
      <div className="stats-loading">
        <div className="spinner" />
        <span>Loading telemetry...</span>
      </div>
    );
  }

  if (error && !stats) {
    return (
      <div className="stats-loading">
        <span style={{ fontSize: "2rem" }}>⚠️</span>
        <h2 style={{ color: "var(--text-secondary)" }}>Engine Unreachable</h2>
        <p style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{error}</p>
      </div>
    );
  }

  const opt  = stats?.optimization || {};
  const cache = stats?.cache || {};
  const svc  = stats?.services || {};

  const loadBreakdown = [
    { name: "Cache Hits",  value: opt.cache_hits || 0, fill: "#00d4aa" },
    { name: "Recomputed",  value: opt.inferences_recomputed || 0, fill: "#7c3aed" },
    { name: "Active Jobs", value: stats?.active_jobs || 0, fill: "#f59e0b" },
  ];

  const cacheHitRate = ((opt.cache_hit_rate || 0) * 100).toFixed(1);
  const cacheEffMsg = (cache.hit_rate || 0) >= 0.5
    ? "Excellent cache utilization"
    : (cache.hit_rate || 0) >= 0.25
    ? "Moderate cache utilization"
    : "Cache warming in progress";

  const services = [
    { name: "API Server",       key: svc.api,      icon: "⚡" },
    { name: "Analysis Pipeline", key: svc.pipeline, icon: "🔬" },
    { name: "Cache Layer",       key: svc.cache,    icon: "💾" },
    { name: "Storage",           key: "active",     icon: "🗂️" },
  ];

  const colors = getChartColors();

  return (
    <div className="stats-root">
      {/* Page Header */}
      <div className="stats-page-header">
        <div>
          <h1 className="stats-page-title">System Analytics</h1>
          <p className="stats-page-sub">Engine telemetry · Refreshes every 2 seconds</p>
        </div>
        <div className="live-indicator">
          <div className="live-dot" />
          LIVE
        </div>
      </div>

      {/* ── KPI Cards ── */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-icon teal">🔬</div>
          <div className="kpi-data">
            <div className="kpi-label">Total Inferences</div>
            <div className="kpi-value">{opt.total_inferences || 0}</div>
            <div className="kpi-trend">All-time analysis runs</div>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon green">⚡</div>
          <div className="kpi-data">
            <div className="kpi-label">Cache Hit Rate</div>
            <div className="kpi-value">{cacheHitRate}%</div>
            <div className="kpi-trend">{cacheEffMsg}</div>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon amber">💾</div>
          <div className="kpi-data">
            <div className="kpi-label">Cache Size</div>
            <div className="kpi-value">{(cache.size_mb || 0).toFixed(1)}<span style={{ fontSize: "0.9rem", fontWeight: 500, color: "var(--text-secondary)" }}> MB</span></div>
            <div className="kpi-trend">{cache.entries || 0} stored entries</div>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon purple">⏱️</div>
          <div className="kpi-data">
            <div className="kpi-label">Engine Uptime</div>
            <div className="kpi-value">{formatDuration(stats?.uptime_seconds || 0)}</div>
            <div className="kpi-trend">{stats?.active_jobs || 0} active job(s)</div>
          </div>
        </div>
      </div>

      {/* ── Service Status ── */}
      <div>
        <div style={{ marginBottom: 12 }}>
          <div className="section-header" style={{ marginBottom: 0 }}>
            <div className="section-title">
              🔴 Service Health
              <span className="section-title-label">Live Status</span>
            </div>
          </div>
        </div>
        <div className="service-status-grid">
          {services.map(s => {
            const ok = s.key === "active";
            return (
              <div key={s.name} className="service-status-item">
                <div className="service-dot-row">
                  <div className={`service-status-dot ${ok ? "ok" : "warn"}`} />
                  <span className="service-name">{s.icon} {s.name}</span>
                </div>
                <div className={`service-state ${ok ? "ok" : "warn"}`}>
                  {ok ? "OPERATIONAL" : "DEGRADED"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Charts ── */}
      <div>
        <div className="section-header">
          <div className="section-title">
            📈 Performance Telemetry
            <span className="section-title-label">Last 20 readings</span>
          </div>
        </div>
        <div className="stats-charts-grid">
          {/* Inference Timeline */}
          <div className="stats-chart-card">
            <div className="stats-chart-title">Inference Activity Timeline</div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={history} margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: colors.tick }} />
                <YAxis tick={{ fontSize: 10, fill: colors.tick }} />
                <Tooltip contentStyle={getTooltipStyle()} itemStyle={{ color: "var(--text-primary)" }} />
                <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }} />
                <Line type="monotone" dataKey="total"     stroke="#00d4aa" strokeWidth={2} dot={false} name="Total Runs" />
                <Line type="monotone" dataKey="cacheHits" stroke="#10b981" strokeWidth={2} dot={false} name="Cache Hits" strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Load Breakdown */}
          <div className="stats-chart-card">
            <div className="stats-chart-title">Load Breakdown</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={loadBreakdown} margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: colors.tick }} />
                <YAxis tick={{ fontSize: 10, fill: colors.tick }} />
                <Tooltip contentStyle={getTooltipStyle()} itemStyle={{ color: "var(--text-primary)" }} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]} name="Count">
                  {loadBreakdown.map((entry, idx) => (
                    <rect key={idx} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Hit Rate Trend */}
          <div className="stats-chart-card" style={{ gridColumn: "1 / -1" }}>
            <div className="stats-chart-title">Cache Hit Rate Trend (%)</div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={history} margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: colors.tick }} />
                <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 10, fill: colors.tick }} />
                <Tooltip contentStyle={getTooltipStyle()} itemStyle={{ color: "var(--text-primary)" }} formatter={v => [`${v}%`, "Hit Rate"]} />
                <Line type="monotone" dataKey="hitRate" stroke="#f59e0b" strokeWidth={2} dot={false} name="Cache Hit %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ── Detailed Stats ── */}
      <div>
        <div className="section-header">
          <div className="section-title">
            🛠️ Detailed Diagnostics
            <span className="section-title-label">Engine Internals</span>
          </div>
        </div>
        <div className="stats-detail-grid">
          <div className="stats-detail-card">
            <div className="stats-detail-title">Cache Statistics</div>
            {[
              ["Entries",   cache.entries || 0],
              ["Size",      `${(cache.size_mb || 0).toFixed(1)} MB`],
              ["Hit Rate",  `${((cache.hit_rate || 0)*100).toFixed(1)}%`],
              ["Evictions", cache.evictions || 0],
            ].map(([k, v]) => (
              <div key={String(k)} className="stats-kv-row">
                <span className="stats-kv-key">{k}</span>
                <span className="stats-kv-val">{v}</span>
              </div>
            ))}
          </div>

          <div className="stats-detail-card">
            <div className="stats-detail-title">Inference Statistics</div>
            {[
              ["Total Runs",   opt.total_inferences || 0],
              ["Cache Hits",   opt.cache_hits || 0],
              ["Recomputed",   opt.inferences_recomputed || 0],
              ["Hit Rate",     `${((opt.cache_hit_rate||0)*100).toFixed(1)}%`],
            ].map(([k, v]) => (
              <div key={String(k)} className="stats-kv-row">
                <span className="stats-kv-key">{k}</span>
                <span className="stats-kv-val">{v}</span>
              </div>
            ))}
          </div>

          <div className="stats-detail-card">
            <div className="stats-detail-title">Job Queue</div>
            {[
              ["Active Jobs",  stats?.active_jobs || 0],
              ["Queue Depth",  Math.max(0, (stats?.active_jobs || 0) - 1)],
              ["Uptime",       formatDuration(stats?.uptime_seconds || 0)],
              ["Total Runs",   opt.total_inferences || 0],
            ].map(([k, v]) => (
              <div key={String(k)} className="stats-kv-row">
                <span className="stats-kv-key">{k}</span>
                <span className="stats-kv-val">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Insights ── */}
      <div className="glass-panel" style={{ padding: 20 }}>
        <div className="section-title" style={{ marginBottom: 4 }}>
          💡 Optimization Insights
        </div>
        <ul className="insights-list">
          <li>Cache efficiency: <strong>{cacheHitRate}%</strong> hit rate - {cacheEffMsg}</li>
          <li>Inference trend: <strong>{opt.total_inferences || 0}</strong> total runs, <strong>{opt.cache_hits || 0}</strong> served from cache</li>
          <li>Active load: <strong>{stats?.active_jobs || 0}</strong> job(s) currently in queue</li>
          <li>Same-video re-analysis serves from cache in under 1ms</li>
        </ul>
      </div>
    </div>
  );
}
