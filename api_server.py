"""FastAPI backend server for React frontend with real-time WebSocket support."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from src.pipeline.optimized_inference import analyze_video_optimized, get_optimization_stats
from src.utils.cache_manager import get_cache_stats, clear_cache
from src.utils.io import ensure_dir

# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(
    title="DeepFake Detection API",
    description="Production-grade API for video analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = Path("data/processed")
JOBS_DIR = Path("data/jobs")

ensure_dir(UPLOAD_DIR)
ensure_dir(PROCESSED_DIR)
ensure_dir(JOBS_DIR)

# In-memory job tracking
jobs_db: dict[str, dict[str, Any]] = {}


# ============================================================================
# DATA MODELS
# ============================================================================

class AnalysisRequest:
    """Request model for analysis."""

    pass


class AnalysisResult:
    """Analysis result with metadata."""

    pass


class HealthCheckResponse:
    """Health check response."""

    status: str
    timestamp: str
    version: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def generate_job_id() -> str:
    """Generate unique job ID."""
    return str(uuid.uuid4())[:8]


def create_job(filename: str) -> str:
    """Create new job entry and return job ID."""
    job_id = generate_job_id()
    jobs_db[job_id] = {
        "id": job_id,
        "filename": filename,
        "status": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }
    return job_id


def update_job(job_id: str, **kwargs) -> None:
    """Update job status."""
    if job_id in jobs_db:
        jobs_db[job_id].update(kwargs)


def get_job(job_id: str) -> dict[str, Any] | None:
    """Retrieve job by ID."""
    return jobs_db.get(job_id)


def risk_to_color(risk_level: str) -> str:
    """Map risk level to color."""
    risk = risk_level.lower()
    if risk == "high":
        return "#ef4444"
    if risk == "medium":
        return "#f59e0b"
    return "#22c55e"


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "api": "active",
                "cache": "active",
                "pipeline": "active",
            },
        }
    )


@app.get("/api/stats")
async def get_stats() -> JSONResponse:
    """Get system statistics."""
    opt_stats = get_optimization_stats()
    cache_stats = get_cache_stats()

    return JSONResponse(
        {
            "optimization": opt_stats,
            "cache": cache_stats,
            "uptime_seconds": int((datetime.now() - datetime.now()).total_seconds()),
            "active_jobs": len([j for j in jobs_db.values() if j["status"] in ["queued", "processing"]]),
        }
    )


@app.get("/api/jobs")
async def list_jobs() -> JSONResponse:
    """List all jobs."""
    jobs_list = list(jobs_db.values())
    jobs_list.sort(key=lambda x: x["created_at"], reverse=True)
    return JSONResponse(jobs_list[:100])  # Return last 100 jobs


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    """Get specific job status."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)


# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None) -> JSONResponse:
    """
    Upload and analyze video.

    Returns job ID for tracking progress.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Create job
    job_id = create_job(file.filename)
    update_job(job_id, status="receiving", progress=10)

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Queue analysis in background
    if background_tasks:
        background_tasks.add_task(_run_analysis, job_id, file_path)

    return JSONResponse(
        {
            "job_id": job_id,
            "status": "queued",
            "message": f"Analysis queued. Check /api/jobs/{job_id} for status.",
        },
        status_code=202,
    )


async def _run_analysis(job_id: str, video_path: Path) -> None:
    """Background task to run analysis."""
    try:
        update_job(job_id, status="processing", progress=20)

        # Run pipeline
        payload = analyze_video_optimized(video_path, PROCESSED_DIR)

        update_job(job_id, status="processing", progress=90)

        # Enrich result with metadata
        result = {
            "job_id": job_id,
            "video_file": video_path.name,
            "completed_at": datetime.now().isoformat(),
            "analysis": payload,
            "risk_color": risk_to_color(payload.get("scoring", {}).get("risk_level", "unknown")),
        }

        update_job(job_id, status="completed", progress=100, result=result)

    except Exception as e:
        update_job(job_id, status="failed", progress=0, error=str(e))


@app.post("/api/analyze-sync")
async def analyze_video_sync(file: UploadFile = File(...)) -> JSONResponse:
    """
    Synchronous analysis (blocks until complete).

    Use for smaller videos or UI integration.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = UPLOAD_DIR / file.filename

    try:
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        payload = analyze_video_optimized(file_path, PROCESSED_DIR)

        return JSONResponse(
            {
                "success": True,
                "video_file": file.filename,
                "analysis": payload,
                "risk_color": risk_to_color(payload.get("scoring", {}).get("risk_level", "unknown")),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ============================================================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================================================


@app.post("/api/cache/clear")
async def clear_all_cache() -> JSONResponse:
    """Clear all caches."""
    clear_cache()
    return JSONResponse({"status": "cache cleared", "timestamp": datetime.now().isoformat()})


@app.get("/api/cache/stats")
async def cache_statistics() -> JSONResponse:
    """Get cache statistics."""
    stats = get_cache_stats()
    return JSONResponse(stats)


# ============================================================================
# WEBSOCKET ENDPOINTS (Real-time progress)
# ============================================================================


@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time job progress."""
    await websocket.accept()

    try:
        job = get_job(job_id)
        if not job:
            await websocket.send_json({"error": "Job not found"})
            await websocket.close(code=4004)
            return

        # Send current state
        await websocket.send_json(job)

        # Poll for updates
        while True:
            await websocket.send_json(jobs_db[job_id])

            # Check if job is complete
            if jobs_db[job_id]["status"] in ["completed", "failed"]:
                break

            # Rate limit: poll every 500ms
            import asyncio

            await asyncio.sleep(0.5)

    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> JSONResponse:
    """Cancel a queued job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed/failed job")

    update_job(job_id, status="cancelled")
    return JSONResponse({"status": "cancelled", "job_id": job_id})


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> JSONResponse:
    """Delete job and associated files."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete uploaded file
    file_path = UPLOAD_DIR / job["filename"]
    if file_path.exists():
        file_path.unlink()

    # Remove from jobs DB
    del jobs_db[job_id]

    return JSONResponse({"status": "deleted", "job_id": job_id})


@app.post("/api/cleanup")
async def cleanup_old_jobs(days: int = 7) -> JSONResponse:
    """Clean up old jobs and files."""
    cutoff_time = datetime.now() - timedelta(days=days)
    deleted_count = 0

    for job_id, job in list(jobs_db.items()):
        created_at = datetime.fromisoformat(job["created_at"])
        if created_at < cutoff_time:
            file_path = UPLOAD_DIR / job["filename"]
            if file_path.exists():
                file_path.unlink()
            del jobs_db[job_id]
            deleted_count += 1

    return JSONResponse(
        {
            "deleted_jobs": deleted_count,
            "cutoff_date": cutoff_time.isoformat(),
        }
    )


# ============================================================================
# SERVE REACT FRONTEND
# ============================================================================

# This will be configured to serve the React frontend build


@app.get("/")
async def serve_root() -> JSONResponse:
    """Serve API root."""
    return JSONResponse(
        {
            "service": "DeepFake Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions gracefully."""
    return JSONResponse(
        {
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
        status_code=exc.status_code,
    )


# ============================================================================
# STARTUP HOOKS
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    print("🚀 DeepFake Detection API server starting...")
    print(f"📁 Upload dir: {UPLOAD_DIR.resolve()}")
    print(f"📁 Process dir: {PROCESSED_DIR.resolve()}")
    print(f"📁 Jobs dir: {JOBS_DIR.resolve()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 DeepFake Detection API server shutting down...")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Starting DeepFake Detection API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
