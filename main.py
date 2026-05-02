# main.py
# Smart Paws — AI Behaviour Checker API
# Deploy on Railway.app

import os
import uuid
import shutil
import time
import json # Added for Groq
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq # Added Groq

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

# ── Groq Client Initialization ────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None
if not groq_client:
    print("WARNING: GROQ_API_KEY not found. Will fall back to raw model output.")

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
    lifspan     = lifespan,
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
    """Health check endpoint."""
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
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please retry in a few seconds.")

    has_text  = bool(description and description.strip())
    has_video = video is not None and video.filename

    if not has_text and not has_video:
        return JSONResponse(
            status_code=400,
            content={"detail": "Please provide a description or upload a video."}
        )

    tmp_video_path = None
    start_time     = time.time()

    try:
        # ── Save uploaded video to temp ───────────────────────────────
        if has_video:
            content = await video.read()
            if len(content) > MAX_VIDEO_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Video too large. Please upload a clip under 50 MB."}
                )

            ext = os.path.splitext(video.filename)[-1].lower() or ".mp4"
            allowed_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
            if ext not in allowed_ext:
                return JSONResponse(
                    status_code=415,
                    content={"detail": f"Unsupported format '{ext}'. Use MP4, MOV, or AVI."}
                )

            tmp_video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
            with open(tmp_video_path, "wb") as f:
                f.write(content)

        # ── 1. Run inference using your local PyTorch/XGBoost engine ───
        raw_result = engine.predict(
            text       = description or "",
            video_path = tmp_video_path,
            breed      = breed or "unknown",
            animal     = animal or "unknown",
        )

        # ── 2. Format output with Groq for the Flutter UI ──────────────
        final_ui_result = None

        if groq_client:
            try:
                system_prompt = (
                    "You are an expert veterinary behaviorist. "
                    "You will receive Raw AI Output from a computer vision model. "
                    "Your job is to translate that raw data into a clean JSON response. "
                    "CRITICAL DIAGNOSIS RULE: You MUST map the 'diagnosis' to one of these exact 3 conditions: "
                    "1. 'Separation Anxiety' "
                    "2. 'Aggression / Hostility' "
                    "3. 'Depression / Stress'. "
                    "If the raw behavior describes something else (e.g., 'excessive barking' or 'hiding'), "
                    "you MUST map it to the closest matching condition from the 3 listed above. "
                    "Respond ONLY with a raw JSON object. "
                    "You MUST use this exact schema:\n"
                    "{\n"
                    '  "diagnosis": "String (Strictly one of the 3 conditions)",\n'
                    '  "confidence": "String (e.g., 85%)",\n'
                    '  "indicators": [\n'
                    '    {"icon": "warning|volume|run|pets", "text": "Short warning text", "color": "orange|red|amber"}\n'
                    "  ],\n"
                    '  "actions": [\n'
                    '    {"title": "Action title", "desc": "Detailed explanation of what the owner should do."}\n'
                    "  ]\n"
                    "}"
                )

                user_prompt = f"User Description: {description}\nRaw AI Output: {raw_result}"

                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    response_format={"type": "json_object"} 
                )
                
                final_ui_result = json.loads(chat_completion.choices[0].message.content)
            
            except Exception as e:
                print(f"Groq formatting failed, using fallback logic: {e}")
        
        # ── 3. Fallback logic if Groq fails or is missing ──────────────
        if not final_ui_result:
            final_ui_result = {
                "diagnosis": str(raw_result.get("detected_behavior", "Unknown Behavior")).title(),
                "confidence": "Analysis Complete",
                "indicators": [{"icon": "warning", "text": "See recommendations below", "color": "orange"}],
                "actions": [
                    {"title": "Suggestion", "desc": s} for s in raw_result.get("suggestions", ["Consult a vet for further advice."])
                ]
            }

        final_ui_result["processing_time_ms"] = round((time.time() - start_time) * 1000)
        return JSONResponse(content=final_ui_result)

    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Analysis failed: {str(e)}"}
        )

    finally:
        # Always clean up the temp file
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)