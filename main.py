# main.py
# Smart Paws — AI Behaviour Checker API
# Deploy on Railway.app

import os
import uuid
import shutil
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import PetBehaviorEngine

# ── Model paths (relative to project root) ───────────────────────────
MODEL_DIR   = os.getenv("MODEL_DIR", "./models")
XGB_PATH    = os.path.join(MODEL_DIR, "xgboost_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "flow_scaler.pkl")
CNN_PATH    = os.path.join(MODEL_DIR, "cnn_scripted.pt")

# ── Upload temp directory ─────────────────────────────────────────────
UPLOAD_DIR  = "/tmp/smart_paws_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Max video size: 50 MB ─────────────────────────────────────────────
MAX_VIDEO_BYTES = 50 * 1024 * 1024

# ── Global engine (loaded once at startup) ────────────────────────────
engine: PetBehaviorEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up at shutdown."""
    global engine
    print("Loading models …")
    try:
        engine = PetBehaviorEngine(
            xgb_path    = XGB_PATH,
            scaler_path = SCALER_PATH,
            cnn_path    = CNN_PATH,
        )
        print("All models loaded. API is ready.")
    except Exception as e:
        print(f"ERROR loading models: {e}")
        print("API will start but /analyze will return errors.")
    yield
    # Cleanup on shutdown
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    print("API shut down cleanly.")


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "Smart Paws — AI Behaviour Checker",
    description = "Analyzes pet behavior from text descriptions and video clips.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


# ═════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "service": "Smart Paws AI Behaviour Checker",
        "version": "1.0.0",
        "status":  "running",
        "endpoints": {
            "analyze": "POST /behavior/analyze",
            "health":  "GET  /health",
            "docs":    "GET  /docs",
        },
    }


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Flutter calls this on app startup to verify the API is live.
    """
    return {
        "status":        "ok",
        "model_loaded":  engine is not None,
        "model_version": "1.0.0",
        "timestamp":     int(time.time()),
    }


@app.post("/behavior/analyze")
async def analyze_behavior(
    description: Optional[str]        = Form(default=""),
    animal:      Optional[str]        = Form(default="unknown"),
    breed:       Optional[str]        = Form(default="unknown"),
    video:       Optional[UploadFile] = File(default=None),
):
    """
    Main analysis endpoint.

    Flutter sends a multipart/form-data POST with:
      description : str  — text describing the behaviour (optional)
      animal      : str  — "dog" or "cat" (from your breed module)
      breed       : str  — e.g. "german_shepherd" (from your breed module)
      video       : file — short video clip, mp4/mov/avi (optional)

    Returns JSON with diagnosis, confidence, severity, and suggestions.

    At least one of description or video must be provided.
    """
    if engine is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Please retry in a few seconds.",
        )

    has_text  = bool(description and description.strip())
    has_video = video is not None and video.filename

    # Must have at least one input
    if not has_text and not has_video:
        return JSONResponse(
            status_code = 400,
            content     = {
                "success": False,
                "error":   "Please provide a description or upload a video.",
                "detected_behavior": "No input provided",
                "suggestions": [
                    "Describe your pet's behaviour in the text box.",
                    "Or upload a short video clip for visual analysis.",
                ],
            },
        )

    tmp_video_path = None
    start_time     = time.time()

    try:
        # ── Save uploaded video to temp ───────────────────────────────
        if has_video:
            content = await video.read()

            if len(content) > MAX_VIDEO_BYTES:
                return JSONResponse(
                    status_code = 413,
                    content     = {
                        "success": False,
                        "error":   "Video too large. Please upload a clip under 50 MB.",
                        "suggestions": [
                            "Trim your video to 10–15 seconds.",
                            "Record in lower resolution (720p is sufficient).",
                        ],
                    },
                )

            ext            = os.path.splitext(video.filename)[-1].lower() or ".mp4"
            allowed_ext    = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
            if ext not in allowed_ext:
                return JSONResponse(
                    status_code = 415,
                    content     = {
                        "success": False,
                        "error":   f"Unsupported format '{ext}'. Use MP4, MOV, or AVI.",
                    },
                )

            tmp_video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
            with open(tmp_video_path, "wb") as f:
                f.write(content)

        # ── Run inference ─────────────────────────────────────────────
        result = engine.predict(
            text       = description or "",
            video_path = tmp_video_path,
            breed      = breed or "unknown",
            animal     = animal or "unknown",
        )

        result["processing_time_ms"] = round((time.time() - start_time) * 1000)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code = 500,
            content     = {
                "success": False,
                "error":   f"Analysis failed: {str(e)}",
                "detected_behavior": "Analysis error",
                "suggestions": [
                    "Please try again.",
                    "If the problem persists, try with a shorter video clip.",
                    "You can also describe the behaviour using text only.",
                ],
            },
        )

    finally:
        # Always clean up the temp file
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)


# ── Local dev entry point ─────────────────────────────────────────────
# Run with:  python main.py
# Or:        uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
