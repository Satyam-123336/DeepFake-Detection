/**
 * Resolves the backend API base URL dynamically.
 * 
 * Guarantees:
 * 1. Respects explicit VITE_API_BASE_URL environment variable if set.
 * 2. When served directly by FastAPI (port 8000 or production proxy), uses relative path ""
 *    so browser requests are same-origin (eliminating CORS and 127.0.0.1 vs localhost mismatches).
 * 3. When loaded in Vite dev server (e.g. port 3000 / 5173), dynamically connects to port 8000
 *    matching the current browser hostname (localhost or 127.0.0.1).
 */
export const getApiBase = (): string => {
  const envUrl = (import.meta as any).env?.VITE_API_BASE_URL;
  if (envUrl && typeof envUrl === "string" && envUrl.trim() !== "") {
    return envUrl.trim().replace(/\/+$/, "");
  }

  if (typeof window !== "undefined") {
    const port = window.location.port;
    // When served from FastAPI backend (port 8000) or standard web proxy (port 80/443 or none)
    if (port === "8000" || port === "" || port === "80" || port === "443") {
      return "";
    }
    // In Vite dev server (ports 3000, 5173, etc.): connect to backend on same host:8000
    const hostname = window.location.hostname || "localhost";
    return `http://${hostname}:8000`;
  }

  return "http://localhost:8000";
};

export const API_BASE = getApiBase();
