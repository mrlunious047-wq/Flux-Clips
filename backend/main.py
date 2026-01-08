"""
ClipMaker Pro Backend API
Complete video processing with OpenAI Whisper + GPT-4
"""

import os
import uuid
import json
import tempfile
import shutil
import asyncio
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
import aiofiles

# AI/ML imports
import openai
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Database/Queue imports (optional - using in-memory for simplicity)
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clipmaker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ClipMaker Pro API",
    description="AI-powered video clipping service with Whisper + GPT-4",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # In production: ["your-domain.com"]
)

# ==================== MODELS ====================
class ClipRequest(BaseModel):
    video_url: Optional[str] = None
    prompt: str = Field(..., min_length=5, max_length=1000)
    num_clips: int = Field(default=5, ge=1, le=20)
    min_duration: int = Field(default=15, ge=5, le=60)
    max_duration: int = Field(default=60, ge=15, le=180)
    style: str = Field(default="engaging", regex="^(engaging|educational|funny|dramatic|inspirational)$")
    include_captions: bool = True
    caption_style: str = Field(default="modern", regex="^(modern|classic|bold|minimal)$")
    
    @validator('max_duration')
    def max_duration_greater_than_min(cls, v, values):
        if 'min_duration' in values and v <= values['min_duration']:
            raise ValueError('max_duration must be greater than min_duration')
        return v

class Clip(BaseModel):
    id: str
    start_time: float
    end_time: float
    duration: float
    caption: str
    confidence: float = Field(..., ge=0, le=1)
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    transcription: Optional[str] = None
    keywords: List[str] = []
    sentiment: Optional[str] = None

class ProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str
    progress: int = Field(..., ge=0, le=100)
    estimated_time: Optional[float] = None
    clips: List[Clip] = []

class VideoInfo(BaseModel):
    id: str
    filename: str
    duration: float
    size: int
    resolution: Optional[str] = None
    fps: Optional[float] = None
    created_at: datetime

# ==================== CONFIGURATION ====================
# Load environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-demo-key")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4-1106-preview")
JWT_SECRET = os.getenv("JWT_SECRET", "development-secret-key-change-in-production")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 2048)) * 1024 * 1024  # 2GB default
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "mp4,mov,avi,mkv,webm").split(",")

# Initialize services
openai.api_key = OPENAI_API_KEY

# Load Whisper model (cache it)
try:
    whisper_model = whisper.load_model(WHISPER_MODEL)
    logger.info(f"Loaded Whisper model: {WHISPER_MODEL}")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

for dir_path in [UPLOAD_DIR, PROCESSED_DIR, THUMBNAIL_DIR]:
    dir_path.mkdir(exist_ok=True)

# In-memory storage (replace with Redis/DB in production)
jobs_db = {}
clips_db = {}
videos_db = {}

# ==================== UTILITIES ====================
def generate_id() -> str:
    return str(uuid.uuid4())

def validate_video_file(file: UploadFile) -> bool:
    """Validate video file before processing"""
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    ext = Path(file.filename).suffix.lower()[1:]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    return True

def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Extract video metadata using FFmpeg"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        video_stream = next(
            (s for s in info["streams"] if s["codec_type"] == "video"), 
            None
        )
        
        return {
            "duration": float(info["format"]["duration"]),
            "size": int(info["format"]["size"]),
            "resolution": f"{video_stream['width']}x{video_stream['height']}" if video_stream else None,
            "fps": eval(video_stream["avg_frame_rate"]) if video_stream and "avg_frame_rate" in video_stream else None,
            "codec": video_stream["codec_name"] if video_stream else None
        }
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return {"duration": 0, "size": 0}

# ==================== AI PROCESSING FUNCTIONS ====================
async def transcribe_audio(audio_path: str) -> List[Dict]:
    """Transcribe audio using Whisper"""
    try:
        if not whisper_model:
            # Fallback to OpenAI Whisper API
            with open(audio_path, "rb") as audio_file:
                transcript = await openai.Audio.atranscribe(
                    "whisper-1",
                    audio_file,
                    response_format="verbose_json"
                )
            return transcript.get("segments", [])
        
        # Use local Whisper
        result = whisper_model.transcribe(audio_path, word_timestamps=True)
        return result.get("segments", [])
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        # Return mock data for development
        return [
            {"start": 0, "end": 5, "text": "Sample transcription for testing."},
            {"start": 5, "end": 10, "text": "This is where interesting content happens."}
        ]

async def analyze_with_gpt(segments: List[Dict], request: ClipRequest) -> List[Dict]:
    """Use GPT to find best clips based on prompt"""
    
    # Prepare transcript with timestamps
    transcript = "\n".join([
        f"[{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}"
        for segment in segments[:50]  # Limit to prevent token overflow
    ])
    
    system_prompt = f"""You are an expert video editor AI. Your task is to find {request.num_clips} engaging clips from a video transcript.

USER REQUEST: {request.prompt}
CLIP STYLE: {request.style}
CLIP DURATION RANGE: {request.min_duration}-{request.max_duration} seconds

RULES:
1. Each clip must be between {request.min_duration} and {request.max_duration} seconds
2. Select moments that would perform well on social media (TikTok, Instagram Reels, YouTube Shorts)
3. Clips should have clear audio and emotional impact
4. Provide a compelling caption for each clip
5. Rate each clip's potential from 0.0 to 1.0
6. Extract 3-5 keywords for each clip
7. Analyze sentiment (positive, negative, neutral, mixed)

TRANSCRIPT WITH TIMESTAMPS:
{transcript}

Return ONLY valid JSON in this exact format:
{{
    "clips": [
        {{
            "start_time": 123.45,
            "end_time": 145.67,
            "duration": 22.22,
            "caption": "Engaging caption here",
            "confidence": 0.85,
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "sentiment": "positive"
        }}
    ]
}}
"""
    
    try:
        response = await openai.ChatCompletion.acreate(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Find the best clips from this transcript."}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        clips = result.get("clips", [])
        
        # Add IDs and validate
        for clip in clips:
            clip["id"] = generate_id()
            clip["duration"] = clip["end_time"] - clip["start_time"]
            
            # Ensure duration is within bounds
            if clip["duration"] < request.min_duration:
                clip["end_time"] = clip["start_time"] + request.min_duration
                clip["duration"] = request.min_duration
            elif clip["duration"] > request.max_duration:
                clip["end_time"] = clip["start_time"] + request.max_duration
                clip["duration"] = request.max_duration
        
        return clips[:request.num_clips]  # Ensure we only return requested number
    
    except Exception as e:
        logger.error(f"GPT analysis failed: {e}")
        # Return mock clips for development
        return [
            {
                "id": generate_id(),
                "start_time": i * 60.0,
                "end_time": (i * 60.0) + 30.0,
                "duration": 30.0,
                "caption": f"Engaging clip #{i+1} about {request.prompt.split()[0]}",
                "confidence": 0.8 + (i * 0.03),
                "keywords": ["viral", "trending", "engaging"],
                "sentiment": "positive"
            }
            for i in range(request.num_clips)
        ]

def extract_audio(video_path: str) -> str:
    """Extract audio from video using FFmpeg"""
    audio_path = f"temp_audio_{uuid.uuid4()}.mp3"
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-q:a", "0", "-map", "a",
        audio_path, "-y",
        "-loglevel", "error"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio extraction failed: {e.stderr}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio extraction failed: {e.stderr}"
        )

def generate_clip(video_path: str, clip_data: Dict, include_captions: bool = True) -> Dict:
    """Generate actual video clip with optional captions"""
    clip_id = clip_data["id"]
    start_time = clip_data["start_time"]
    end_time = clip_data["end_time"]
    caption = clip_data["caption"]
    
    output_path = PROCESSED_DIR / f"clip_{clip_id}.mp4"
    thumbnail_path = THUMBNAIL_DIR / f"thumb_{clip_id}.jpg"
    
    try:
        # Load video
        video = VideoFileClip(str(video_path))
        
        # Extract subclip
        clip = video.subclip(start_time, min(end_time, video.duration))
        
        # Resize for vertical format (9:16)
        target_width = 1080
        target_height = 1920
        
        # Calculate resize maintaining aspect ratio
        clip_resized = clip.resize(height=target_height)
        
        # Crop if too wide
        if clip_resized.w > target_width:
            x_center = clip_resized.w / 2
            clip_resized = clip_resized.crop(
                x1=x_center - target_width/2,
                width=target_width
            )
        elif clip_resized.w < target_width:
            # Pad if too narrow
            from moviepy.editor import ColorClip
            background = ColorClip(
                size=(target_width, target_height),
                color=(0, 0, 0)
            ).set_duration(clip.duration)
            
            clip_resized = CompositeVideoClip([
                background,
                clip_resized.set_position("center")
            ])
        
        # Add captions if requested
        if include_captions:
            caption_clip = create_caption_clip(caption, clip.duration, clip_resized.size)
            final_clip = CompositeVideoClip([
                clip_resized,
                caption_clip.set_position(("center", target_height * 0.8))
            ])
        else:
            final_clip = clip_resized
        
        # Export video
        final_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(PROCESSED_DIR / f"temp_audio_{clip_id}.m4a"),
            remove_temp=True,
            fps=30,
            logger=None
        )
        
        # Generate thumbnail
        generate_thumbnail(str(video_path), start_time, str(thumbnail_path))
        
        # Cleanup
        clip.close()
        video.close()
        if 'final_clip' in locals():
            final_clip.close()
        
        return {
            "video_url": f"/processed/clip_{clip_id}.mp4",
            "thumbnail_url": f"/thumbnails/thumb_{clip_id}.jpg"
        }
        
    except Exception as e:
        logger.error(f"Clip generation failed: {e}")
        # Return placeholder URLs
        return {
            "video_url": None,
            "thumbnail_url": None
        }

def create_caption_clip(text: str, duration: float, video_size: tuple) -> TextClip:
    """Create styled caption text clip"""
    try:
        # Create text clip with styling
        txt_clip = TextClip(
            text,
            fontsize=48,
            color='white',
            font='Arial-Bold',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(video_size[0] * 0.9, None),
            align='center'
        )
        
        # Set duration
        txt_clip = txt_clip.set_duration(duration)
        
        # Add fade in/out
        txt_clip = txt_clip.crossfadein(0.5).crossfadeout(0.5)
        
        return txt_clip
        
    except Exception as e:
        logger.error(f"Caption creation failed: {e}")
        # Return empty clip as fallback
        return TextClip("", size=(1, 1)).set_duration(duration)

def generate_thumbnail(video_path: str, timestamp: float, output_path: str):
    """Generate thumbnail from video at specific timestamp"""
    try:
        # Use OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        
        if success:
            # Resize and save
            frame = cv2.resize(frame, (320, 568))  # Thumbnail size
            cv2.imwrite(output_path, frame)
        
        cap.release()
        
    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}")
        # Create placeholder thumbnail
        img = Image.new('RGB', (320, 568), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((160, 284), "Clip", fill=(255, 255, 255), anchor="mm")
        img.save(output_path)

# ==================== BACKGROUND PROCESSING ====================
async def process_video_background(job_id: str, video_path: str, request: ClipRequest):
    """Background task to process video"""
    try:
        jobs_db[job_id]["status"] = "extracting_audio"
        jobs_db[job_id]["progress"] = 20
        
        # Step 1: Extract audio
        audio_path = extract_audio(video_path)
        jobs_db[job_id]["status"] = "transcribing"
        jobs_db[job_id]["progress"] = 40
        
        # Step 2: Transcribe
        segments = await transcribe_audio(audio_path)
        jobs_db[job_id]["status"] = "analyzing"
        jobs_db[job_id]["progress"] = 60
        
        # Step 3: AI analysis
        clips = await analyze_with_gpt(segments, request)
        jobs_db[job_id]["status"] = "generating_clips"
        jobs_db[job_id]["progress"] = 80
        
        # Step 4: Generate clips
        for i, clip in enumerate(clips):
            # Update progress per clip
            clip_progress = 80 + int((i + 1) / len(clips) * 15)
            jobs_db[job_id]["progress"] = clip_progress
            
            # Generate clip video
            media_urls = generate_clip(
                video_path,
                clip,
                request.include_captions
            )
            
            clip.update(media_urls)
            
            # Store clip
            clips_db[clip["id"]] = clip
        
        # Cleanup temp audio
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Update job status
        jobs_db[job_id].update({
            "status": "completed",
            "progress": 100,
            "clips": clips,
            "completed_at": datetime.now().isoformat(),
            "estimated_time": None
        })
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        jobs_db[job_id].update({
            "status": "failed",
            "error": str(e),
            "progress": 0
        })

# ==================== API ENDPOINTS ====================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ClipMaker Pro API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "upload": "POST /api/upload",
            "process": "POST /api/process",
            "jobs": "GET /api/jobs/{job_id}",
            "clips": "GET /api/clips"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "openai": "connected" if OPENAI_API_KEY.startswith("sk-") else "demo",
            "whisper": "loaded" if whisper_model else "not_loaded",
            "storage": {
                "uploads": sum(f.stat().st_size for f in UPLOAD_DIR.rglob('*')),
                "processed": sum(f.stat().st_size for f in PROCESSED_DIR.rglob('*'))
            }
        }
    }

@app.post("/api/upload", response_model=Dict[str, Any])
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a video file"""
    try:
        # Validate file
        validate_video_file(file)
        
        # Generate unique filename
        file_id = generate_id()
        filename = f"{file_id}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        # Save file with progress tracking
        total_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            while chunk := await file.read(chunk_size):
                total_size += len(chunk)
                await out_file.write(chunk)
        
        # Get video info
        video_info = get_video_info(file_path)
        
        # Store video metadata
        video_data = {
            "id": file_id,
            "filename": filename,
            "original_name": file.filename,
            "path": str(file_path),
            "size": total_size,
            "created_at": datetime.now().isoformat(),
            **video_info
        }
        
        videos_db[file_id] = video_data
        
        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "url": f"/uploads/{filename}",
            "download_url": f"/api/download/{file_id}",
            "info": video_info,
            "message": "Upload successful"
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@app.post("/api/process", response_model=ProcessResponse)
async def process_video(
    request: ClipRequest,
    background_tasks: BackgroundTasks
):
    """Process video and generate clips"""
    try:
        # Generate job ID
        job_id = generate_id()
        
        # Determine video source
        if request.video_url:
            video_path = request.video_url
        else:
            # Get latest uploaded video
            if not videos_db:
                raise HTTPException(
                    status_code=400,
                    detail="No video uploaded. Please upload a video first."
                )
            latest_video = list(videos_db.values())[-1]
            video_path = latest_video["path"]
        
        # Create job entry
        jobs_db[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 10,
            "request": request.dict(),
            "video_path": video_path,
            "created_at": datetime.now().isoformat(),
            "estimated_time": 120,  # 2 minutes estimate
            "clips": [],
            "error": None
        }
        
        # Start background processing
        background_tasks.add_task(
            process_video_background,
            job_id,
            video_path,
            request
        )
        
        return ProcessResponse(
            job_id=job_id,
            status="queued",
            message="Processing started. Use GET /api/jobs/{job_id} to check status.",
            progress=10,
            estimated_time=120,
            clips=[]
        )
        
    except Exception as e:
        logger.error(f"Process request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.get("/api/jobs/{job_id}", response_model=ProcessResponse)
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    
    job = jobs_db[job_id]
    
    return ProcessResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        message=f"Job is {job.get('status', 'unknown')}",
        progress=job.get("progress", 0),
        estimated_time=job.get("estimated_time"),
        clips=job.get("clips", [])
    )

@app.get("/api/clips")
async def get_all_clips(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get all generated clips with pagination"""
    clips_list = list(clips_db.values())
    
    # Apply pagination
    paginated_clips = clips_list[offset:offset + limit]
    
    return {
        "clips": paginated_clips,
        "total": len(clips_list),
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < len(clips_list)
    }

@app.get("/api/clips/{clip_id}")
async def get_clip(clip_id: str):
    """Get specific clip"""
    if clip_id not in clips_db:
        raise HTTPException(
            status_code=404,
            detail="Clip not found"
        )
    
    return clips_db[clip_id]

@app.get("/api/videos")
async def get_all_videos():
    """Get all uploaded videos"""
    return {
        "videos": list(videos_db.values()),
        "total": len(videos_db)
    }

@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    """Get specific video"""
    if video_id not in videos_db:
        raise HTTPException(
            status_code=404,
            detail="Video not found"
        )
    
    return videos_db[video_id]

@app.get("/api/download/{file_id}")
async def download_file(file_id: str):
    """Download a file (video or clip)"""
    # Check if it's a video
    if file_id in videos_db:
        video = videos_db[file_id]
        file_path = Path(video["path"])
        if file_path.exists():
            return FileResponse(
                file_path,
                filename=video["original_name"],
                media_type="video/mp4"
            )
    
    # Check if it's a clip
    if file_id in clips_db:
        clip = clips_db[file_id]
        if clip.get("video_url"):
            file_path = PROCESSED_DIR / Path(clip["video_url"]).name
            if file_path.exists():
                return FileResponse(
                    file_path,
                    filename=f"clip_{file_id}.mp4",
                    media_type="video/mp4"
                )
    
    raise HTTPException(
        status_code=404,
        detail="File not found"
    )

@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """Serve uploaded files"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )
    return FileResponse(file_path)

@app.get("/processed/{filename}")
async def get_processed_file(filename: str):
    """Serve processed clips"""
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )
    return FileResponse(file_path)

@app.get("/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    """Serve thumbnails"""
    file_path = THUMBNAIL_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Thumbnail not found"
        )
    return FileResponse(file_path)

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its clips"""
    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    
    # Delete associated clips
    job = jobs_db[job_id]
    for clip in job.get("clips", []):
        clip_id = clip.get("id")
        if clip_id in clips_db:
            # Delete clip files
            clip_data = clips_db[clip_id]
            if clip_data.get("video_url"):
                video_path = PROCESSED_DIR / Path(clip_data["video_url"]).name
                if video_path.exists():
                    video_path.unlink()
            
            if clip_data.get("thumbnail_url"):
                thumb_path = THUMBNAIL_DIR / Path(clip_data["thumbnail_url"]).name
                if thumb_path.exists():
                    thumb_path.unlink()
            
            # Remove from clips DB
            del clips_db[clip_id]
    
    # Remove job
    del jobs_db[job_id]
    
    return {"success": True, "message": f"Job {job_id} deleted"}

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "jobs": {
            "total": len(jobs_db),
            "completed": sum(1 for j in jobs_db.values() if j.get("status") == "completed"),
            "failed": sum(1 for j in jobs_db.values() if j.get("status") == "failed"),
            "processing": sum(1 for j in jobs_db.values() if j.get("status") in ["processing", "queued"])
        },
        "clips": {
            "total": len(clips_db),
            "total_duration": sum(c.get("duration", 0) for c in clips_db.values()),
            "avg_confidence": sum(c.get("confidence", 0) for c in clips_db.values()) / max(len(clips_db), 1)
        },
        "videos": {
            "total": len(videos_db),
            "total_size": sum(v.get("size", 0) for v in videos_db.values()),
            "total_duration": sum(v.get("duration", 0) for v in videos_db.values())
        },
        "storage": {
            "uploads_mb": sum(f.stat().st_size for f in UPLOAD_DIR.rglob('*')) / (1024*1024),
            "processed_mb": sum(f.stat().st_size for f in PROCESSED_DIR.rglob('*')) / (1024*1024),
            "thumbnails_mb": sum(f.stat().st_size for f in THUMBNAIL_DIR.rglob('*')) / (1024*1024)
        }
    }

# ==================== ERROR HANDLERS ====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("ENVIRONMENT") == "development" else None,
            "status_code": 500
        }
    )

# ==================== STARTUP/SHUTDOWN ====================
@app.on_event("startup")
async def startup_event():
    """Run on startup"""
    logger.info("ClipMaker Pro API starting up...")
    
    # Check dependencies
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logger.info("FFmpeg is available")
    except Exception as e:
        logger.warning(f"FFmpeg not found: {e}")
    
    # Cleanup old temp files
    for temp_file in Path(".").glob("temp_audio_*"):
        try:
            temp_file.unlink()
        except:
            pass

@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown"""
    logger.info("ClipMaker Pro API shutting down...")
    
    # Cleanup
    for temp_file in Path(".").glob("temp_audio_*"):
        try:
            temp_file.unlink()
        except:
            pass

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )
