from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
import json
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class ClipRequest(BaseModel):
    video_url: Optional[str] = None
    prompt: str
    num_clips: int = 5
    min_duration: int = 10  # seconds
    max_duration: int = 60  # seconds
    style: str = "engaging"  # engaging, educational, funny, dramatic

class ClipResponse(BaseModel):
    id: str
    start_time: float
    end_time: float
    duration: float
    caption: str
    score: float
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

class ProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str
    clips: List[ClipResponse] = []

# Initialize FastAPI
app = FastAPI(
    title="ClipMaker Pro API",
    description="AI-powered video clipping service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with Redis/DB in production)
jobs_db = {}
clips_db = {}
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Mock AI processing (replace with actual Whisper + GPT)
async def mock_ai_processing(video_path: str, prompt: str, num_clips: int = 5) -> List[Dict]:
    """Mock AI that returns sample clips"""
    import random
    
    clips = []
    for i in range(num_clips):
        start = random.uniform(0, 300)  # Random start within 5 minutes
        duration = random.uniform(15, 45)
        
        clips.append({
            "id": str(uuid.uuid4()),
            "start_time": start,
            "end_time": start + duration,
            "duration": duration,
            "caption": f"This is an engaging clip about {prompt.split()[0]}!",
            "score": random.uniform(0.7, 0.95),
            "video_url": f"/processed/clip_{i}.mp4",
            "thumbnail_url": f"/processed/thumb_{i}.jpg"
        })
    
    # Simulate processing time
    await asyncio.sleep(2)
    return clips

async def process_video_background(job_id: str, video_path: str, request: ClipRequest):
    """Background task to process video"""
    try:
        jobs_db[job_id] = {"status": "processing", "progress": 30}
        
        # Step 1: Extract audio and transcribe (Mock)
        logger.info(f"Processing video: {video_path}")
        
        # Step 2: AI analysis for clip selection
        clips = await mock_ai_processing(video_path, request.prompt, request.num_clips)
        
        # Step 3: Generate clips (Mock - in real implementation, use FFmpeg)
        for clip in clips:
            clip_id = clip["id"]
            clips_db[clip_id] = clip
            
            # Create mock files
            with open(PROCESSED_DIR / f"clip_{clip_id}.txt", "w") as f:
                f.write(f"Clip data for {clip_id}")
        
        jobs_db[job_id] = {
            "status": "completed",
            "progress": 100,
            "clips": clips,
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        jobs_db[job_id] = {"status": "failed", "error": str(e)}

# Routes
@app.get("/")
async def root():
    return {"message": "ClipMaker Pro API", "status": "online"}

@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a video file"""
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "url": f"/uploads/{filename}",
            "message": "Upload successful"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def process_video(
    request: ClipRequest,
    background_tasks: BackgroundTasks
):
    """Process video and generate clips"""
    try:
        job_id = str(uuid.uuid4())
        
        # Create job entry
        jobs_db[job_id] = {
            "status": "queued",
            "progress": 10,
            "request": request.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        # Start background processing
        if request.video_url:
            video_path = request.video_url
        else:
            # Use last uploaded file
            files = list(UPLOAD_DIR.glob("*"))
            if not files:
                raise HTTPException(status_code=400, detail="No video uploaded")
            video_path = str(files[-1])
        
        background_tasks.add_task(
            process_video_background,
            job_id,
            video_path,
            request
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Processing started",
            "status_url": f"/api/jobs/{job_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "clips": job.get("clips", []),
        "error": job.get("error")
    }

@app.get("/api/clips")
async def get_all_clips():
    """Get all generated clips"""
    return {"clips": list(clips_db.values())}

@app.get("/api/clips/{clip_id}")
async def get_clip(clip_id: str):
    """Get specific clip"""
    if clip_id not in clips_db:
        raise HTTPException(status_code=404, detail="Clip not found")
    return clips_db[clip_id]

@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """Serve uploaded files"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
