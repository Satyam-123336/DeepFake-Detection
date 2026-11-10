"""FastAPI backend server for React frontend with real-time WebSocket support."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import asyncio
import urllib.request
import os
import shutil

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from src.pipeline.optimized_inference import (
    analyze_video_optimized,
    get_optimization_stats,
    _optimized_pipeline,
    PIPELINE_VERSION,
)
from src.utils.cache_manager import get_cache_stats, clear_cache
from src.utils.video_cache import video_cache_manager
from src.utils.io import ensure_dir

# ============================================================================
# CONFIGURATION
# ============================================================================

async def keep_alive():
    """Background task to keep Render free tier awake by pinging itself."""
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        print("keep_alive: No RENDER_EXTERNAL_URL found, skipping.")
        return
    health_url = f"{url}/health"
    print(f"keep_alive: Starting ping loop for {health_url} every 14 minutes.")
    while True:
        await asyncio.sleep(14 * 60)  # Sleep 14 minutes
        try:
            # Run synchronous urllib in a thread to avoid blocking the event loop
            await asyncio.to_thread(urllib.request.urlopen, health_url)
            print(f"keep_alive: Successfully pinged {health_url}")
        except Exception as e:
            print(f"keep_alive error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown lifecycle."""
    print("DeepFake Detection API server starting...")
    print(f"Upload dir: {UPLOAD_DIR.resolve()}")
    print(f"Process dir: {PROCESSED_DIR.resolve()}")
    print(f"Jobs dir: {JOBS_DIR.resolve()}")
    
    # Start the keep-alive background task to bypass Render sleep
    ping_task = asyncio.create_task(keep_alive())
    
    yield
    print("DeepFake Detection API server shutting down...")
    ping_task.cancel()


app = FastAPI(
    title="DeepFake Detection API",
    description="Production-grade API for video analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware - allow all origins for flexible local development and remote deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
APP_START_TIME = datetime.now()

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


def create_job(
    filename: str,
    safe_filename: str | None = None,
    job_id: str | None = None,
    video_hash: str | None = None,
) -> str:
    """Create new job entry and return job ID."""
    jid = job_id or generate_job_id()
    jobs_db[jid] = {
        "id": jid,
        "filename": filename,
        "stored_filename": safe_filename or f"{jid}_{filename}",
        "video_hash": video_hash,
        "status": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }
    return jid


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
    video_cache_stats = video_cache_manager.get_stats()

    return JSONResponse(
        {
            "optimization": opt_stats,
            "cache": cache_stats,
            "video_cache": video_cache_stats,
            "uptime_seconds": int((datetime.now() - APP_START_TIME).total_seconds()),
            "active_jobs": len([j for j in jobs_db.values() if j["status"] in ["queued", "processing"]]),
            "services": {
                "api": "active",
                "cache": "active",
                "video_cache": "active",
                "pipeline": "active",
            },
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
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JSONResponse:
    """
    Upload and analyze video with deduplication and instant caching.

    Returns job ID for tracking progress.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Invalid video file: filename is required.")

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {str(e)}")

    if not contents or len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    try:
        # Content-addressed deduplication: store unique video file once
        video_path, video_hash, is_duplicate = video_cache_manager.store(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video storage/cache error: {str(e)}")

    # 1. In-flight deduplication: if identical video is already being analyzed, attach to existing job
    for existing_id, existing_job in jobs_db.items():
        if (
            existing_job.get("video_hash") == video_hash
            and existing_job.get("status") in ("queued", "processing", "receiving")
        ):
            return JSONResponse(
                {
                    "job_id": existing_id,
                    "status": existing_job.get("status"),
                    "message": "Duplicate video detected. Attached to active analysis job.",
                    "from_cache": False,
                    "duplicate_detected": True,
                },
                status_code=202,
            )

    # 2. Check persistent inference cache for instant hit
    cache_key = f"inference_{video_hash}_v{PIPELINE_VERSION}"
    cached_payload = _optimized_pipeline.persistent_cache.get(cache_key)

    job_id = generate_job_id()
    create_job(file.filename, safe_filename=video_path.name, job_id=job_id, video_hash=video_hash)

    if cached_payload is not None:
        _optimized_pipeline._cache_hits += 1
        _optimized_pipeline._inference_count += 1
        cached_payload_copy = dict(cached_payload)
        cached_payload_copy["from_cache"] = True
        cached_payload_copy["video_hash"] = video_hash
        cached_payload_copy["is_duplicate"] = is_duplicate
        result = {
            "job_id": job_id,
            "video_file": file.filename,
            "video_hash": video_hash,
            "completed_at": datetime.now().isoformat(),
            "analysis": cached_payload_copy,
            "risk_color": risk_to_color(cached_payload.get("scoring", {}).get("risk_level", "unknown")),
            "from_cache": True,
            "duplicate_detected": is_duplicate,
        }
        update_job(job_id, status="completed", progress=100, result=result)
        return JSONResponse(
            {
                "job_id": job_id,
                "status": "completed",
                "message": "Video recognized from cache (deduplicated). Result ready immediately.",
                "from_cache": True,
                "duplicate_detected": is_duplicate,
                "result": result,
            },
            status_code=200,
        )

    # 3. New video: schedule background analysis
    update_job(job_id, status="queued", progress=10)
    background_tasks.add_task(_run_analysis, job_id, video_path)

    return JSONResponse(
        {
            "job_id": job_id,
            "status": "queued",
            "message": f"Analysis queued. Check /api/jobs/{job_id} for status.",
            "from_cache": False,
            "duplicate_detected": is_duplicate,
        },
        status_code=202,
    )


def _run_analysis(job_id: str, video_path: Path) -> None:
    """Background task to run analysis."""
    payload = None
    try:
        update_job(job_id, status="processing", progress=20)

        # Run pipeline
        payload = analyze_video_optimized(video_path, PROCESSED_DIR)

        update_job(job_id, status="processing", progress=90)

        # Enrich result with metadata
        result = {
            "job_id": job_id,
            "video_file": video_path.name,
            "video_hash": payload.get("video_hash"),
            "completed_at": datetime.now().isoformat(),
            "analysis": payload,
            "risk_color": risk_to_color(payload.get("scoring", {}).get("risk_level", "unknown")),
            "from_cache": payload.get("from_cache", False),
            "duplicate_detected": payload.get("is_duplicate", False),
        }

        update_job(job_id, status="completed", progress=100, result=result)

    except Exception as e:
        update_job(job_id, status="failed", progress=0, error=str(e))
    finally:
        # CLEANUP: Delete uncompressed intermediate frames,
        # PRESERVE deduplicated video file in data/video_cache
        try:
            if payload and "preprocessing" in payload and "artifact_dir" in payload["preprocessing"]:
                artifact_dir = Path(payload["preprocessing"]["artifact_dir"])
                if artifact_dir.exists() and artifact_dir.is_dir():
                    shutil.rmtree(artifact_dir)
            else:
                for d in PROCESSED_DIR.glob(f"{video_path.stem}*"):
                    if d.is_dir():
                        shutil.rmtree(d)
        except Exception as e:
            print(f"Cleanup failed for {job_id}: {e}")


@app.post("/api/analyze-sync")
async def analyze_video_sync(file: UploadFile = File(...)) -> JSONResponse:
    """
    Synchronous analysis with deduplication and instant caching.

    Use for smaller videos or UI integration.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Invalid video file: filename is required.")

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

    if not contents or len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    try:
        video_path, video_hash, is_duplicate = video_cache_manager.store(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video storage/cache error: {str(e)}")

    # Check cache first for instant sub-millisecond response
    cache_key = f"inference_{video_hash}_v{PIPELINE_VERSION}"
    cached_payload = _optimized_pipeline.persistent_cache.get(cache_key)

    if cached_payload is not None:
        _optimized_pipeline._cache_hits += 1
        _optimized_pipeline._inference_count += 1
        cached_payload_copy = dict(cached_payload)
        cached_payload_copy["from_cache"] = True
        cached_payload_copy["video_hash"] = video_hash
        cached_payload_copy["is_duplicate"] = is_duplicate
        return JSONResponse(
            {
                "success": True,
                "video_file": file.filename,
                "video_hash": video_hash,
                "analysis": cached_payload_copy,
                "risk_color": risk_to_color(cached_payload.get("scoring", {}).get("risk_level", "unknown")),
                "from_cache": True,
                "duplicate_detected": is_duplicate,
            }
        )

    try:
        # Offload CPU-heavy pipeline to threadpool so event loop is never blocked
        payload = await asyncio.to_thread(analyze_video_optimized, video_path, PROCESSED_DIR)

        return JSONResponse(
            {
                "success": True,
                "video_file": file.filename,
                "video_hash": video_hash,
                "analysis": payload,
                "risk_color": risk_to_color(payload.get("scoring", {}).get("risk_level", "unknown")),
                "from_cache": payload.get("from_cache", False),
                "duplicate_detected": is_duplicate,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Cleanup intermediate processed frames, keeping cached video
        try:
            for d in PROCESSED_DIR.glob(f"{video_path.stem}*"):
                if d.is_dir():
                    shutil.rmtree(d)
        except Exception:
            pass


# ============================================================================
# CACHE & VIDEO STORAGE MANAGEMENT ENDPOINTS
# ============================================================================


@app.post("/api/cache/clear")
async def clear_all_cache() -> JSONResponse:
    """Clear in-memory and persistent inference caches."""
    clear_cache()
    _optimized_pipeline.persistent_cache.clear()
    return JSONResponse({"status": "all caches cleared", "timestamp": datetime.now().isoformat()})


@app.get("/api/cache/stats")
async def cache_statistics() -> JSONResponse:
    """Get cache statistics."""
    stats = get_cache_stats()
    return JSONResponse(stats)


@app.get("/api/videos/cache")
async def get_video_cache_stats() -> JSONResponse:
    """Get video deduplication and storage statistics."""
    return JSONResponse(video_cache_manager.get_stats())


@app.post("/api/videos/cache/clear")
async def clear_video_cache() -> JSONResponse:
    """Clear all stored deduplicated videos."""
    video_cache_manager.clear()
    return JSONResponse({"status": "video cache cleared", "timestamp": datetime.now().isoformat()})


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
            current_job = get_job(job_id)
            if not current_job:
                await websocket.send_json({"error": "Job no longer exists"})
                break

            await websocket.send_json(current_job)

            # Check if job is complete
            if current_job["status"] in ["completed", "failed", "cancelled"]:
                break

            # Rate limit: poll every 500ms
            await asyncio.sleep(0.5)

    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


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

    # Delete uploaded file using stored safe filename
    stored_name = job.get("stored_filename", job["filename"])
    file_path = UPLOAD_DIR / stored_name
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
            stored_name = job.get("stored_filename", job["filename"])
            file_path = UPLOAD_DIR / stored_name
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

DIST_DIR = Path("frontend/dist")
if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def serve_root() -> JSONResponse:
        """Serve API root if frontend build is not present."""
        return JSONResponse(
            {
                "service": "DeepFake Detection API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health",
                "note": "Frontend build not found at frontend/dist. Run 'npm run build' inside frontend/ to build UI.",
            }
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions gracefully returning detail and error."""
    return JSONResponse(
        {
            "detail": exc.detail,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all error handler for unexpected backend failures."""
    err_str = str(exc) or "Internal server error"
    return JSONResponse(
        {
            "detail": err_str,
            "error": err_str,
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
        },
        status_code=500,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Starting DeepFake Detection API server...")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Never reload=True in production
        log_level="info",
    )
