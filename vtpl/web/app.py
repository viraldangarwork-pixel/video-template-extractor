import asyncio
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from vtpl.pipeline import CRITICAL, STAGES
from vtpl.util.workdir import WorkDir, video_id

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
WORK_DIR = Path(os.environ.get("VTPL_WORK_DIR", "work")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = WORK_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

AD_CLIENT_ID = os.environ.get("AD_CLIENT_ID", "")
CANONICAL_URL = os.environ.get("CANONICAL_URL", "https://vtpl.app")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))

app = FastAPI(
    title="Video Template Extractor",
    description="Reverse-engineer short-form videos into editable templates",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
executor = ThreadPoolExecutor(max_workers=int(os.environ.get("MAX_CONCURRENT_JOBS", "2")))

STAGE_META = [
    {"name": "ingest", "label": "Normalizing video & reading metadata", "weight": 5},
    {"name": "frames", "label": "Extracting frame cache", "weight": 5},
    {"name": "shots", "label": "Detecting scene cuts & shot boundaries", "weight": 10},
    {"name": "motion", "label": "Measuring camera motion & zoom", "weight": 10},
    {"name": "grade", "label": "Profiling color grading & brightness", "weight": 5},
    {"name": "audio", "label": "Analyzing audio beats & BPM tempo", "weight": 15},
    {"name": "speech", "label": "Transcribing speech & subtitles (Whisper)", "weight": 15},
    {"name": "text", "label": "OCR typography, font & animation tracking", "weight": 30},
    {"name": "merge", "label": "Compiling final template.json", "weight": 5},
]
STAGE_LOOKUP = {s["name"]: s for s in STAGE_META}

# In-memory jobs store
jobs: Dict[str, Dict[str, Any]] = {}
job_subscribers: Dict[str, list[asyncio.Queue]] = {}


def _calc_progress(stage_status: dict) -> int:
    total_pct = 0
    for meta in STAGE_META:
        st = stage_status.get(meta["name"], "pending")
        if st == "ok":
            total_pct += meta["weight"]
        elif st == "running":
            total_pct += meta["weight"] // 2
    return min(100, total_pct)


def _notify_job(job_id: str):
    if job_id not in job_subscribers:
        return
    job_data = jobs.get(job_id)
    if not job_data:
        return
    payload = json.dumps(job_data)
    for q in job_subscribers[job_id]:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def _run_job_sync(job_id: str, source_path: Path):
    job = jobs[job_id]
    job["status"] = "running"
    job["started_at"] = time.time()

    try:
        vid = video_id(source_path)
        job["video_id"] = vid
        wd = WorkDir(WORK_DIR / vid)

        # Check if pre-cached
        template_file = wd.path("template.json")
        if template_file.exists():
            try:
                tmpl = json.loads(template_file.read_text(encoding="utf-8"))
                for s in STAGE_META:
                    job["stages_status"][s["name"]] = "ok"
                job["progress"] = 100
                job["status"] = "done"
                job["template"] = tmpl
                job["message"] = "Analysis retrieved from cache successfully!"
                job["finished_at"] = time.time()
                _notify_job(job_id)
                return
            except Exception:
                pass

        merge_inputs = [s.output for s in STAGES if s.name in CRITICAL and s.name != "merge"]
        dirty = False

        for st in STAGES:
            job["current_stage"] = st.name
            job["stages_status"][st.name] = "running"
            job["progress"] = _calc_progress(job["stages_status"])
            job["message"] = STAGE_LOOKUP[st.name]["label"]
            _notify_job(job_id)

            t0 = time.perf_counter()
            try:
                # If cached output exists and valid, skip re-computation
                if wd.has(st.output) and st.name != "merge":
                    job["stages_status"][st.name] = "ok"
                    job["stage_times"][st.name] = 0.0
                    dirty = True
                    continue

                warnings = st.fn(wd, source_path)
                elapsed = time.perf_counter() - t0
                wd.record(st.name, "ok", elapsed, warnings)
                job["stages_status"][st.name] = "ok"
                job["stage_times"][st.name] = round(elapsed, 2)
                dirty = True
            except Exception as e:
                elapsed = time.perf_counter() - t0
                wd.record(st.name, "failed", elapsed, [str(e)])
                job["stages_status"][st.name] = "failed"
                job["stage_times"][st.name] = round(elapsed, 2)
                if st.name in CRITICAL:
                    raise RuntimeError(f"Critical stage '{st.name}' failed: {e}")

            job["progress"] = _calc_progress(job["stages_status"])
            _notify_job(job_id)

        if not template_file.exists():
            raise RuntimeError("Pipeline completed but template.json was not created.")

        tmpl = json.loads(template_file.read_text(encoding="utf-8"))
        job["template"] = tmpl
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Analysis completed successfully!"
        job["finished_at"] = time.time()
        _notify_job(job_id)

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["message"] = f"Error: {e}"
        job["finished_at"] = time.time()
        _notify_job(job_id)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "ad_client_id": AD_CLIENT_ID,
            "canonical_url": CANONICAL_URL,
            "max_upload_mb": MAX_UPLOAD_MB,
        },
    )


@app.get("/robots.txt")
async def robots():
    content = f"""User-agent: *
Allow: /

Sitemap: {CANONICAL_URL.rstrip('/')}/sitemap.xml
"""
    return Response(content=content, media_type="text/plain")


@app.get("/sitemap.xml")
async def sitemap():
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>{CANONICAL_URL.rstrip('/')}/</loc>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>
"""
    return Response(content=content, media_type="application/xml")


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "time": time.time()}


@app.post("/api/upload")
async def upload_video(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".m4v", ".webm", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {ext}. Use .mp4, .mov, or .webm.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    # 1. Early rejection via Content-Length header to block massive uploads (e.g. 1000MB) before transfer
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            cl = int(content_length)
            if cl > max_bytes:
                mb_received = cl / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_MB}MB (attempted {mb_received:.1f}MB)."
                )
        except ValueError:
            pass

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    target_path = job_dir / f"source{ext}"

    # 2. Stream write with strict chunked size enforcement
    size = 0
    try:
        with open(target_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    buffer.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size ({MAX_UPLOAD_MB}MB)."
                    )
                buffer.write(chunk)
    except HTTPException:
        target_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

    if size == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    init_stages = {s["name"]: "pending" for s in STAGE_META}
    jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "filesize": size,
        "status": "queued",
        "current_stage": "",
        "stages_status": init_stages,
        "stage_times": {},
        "progress": 0,
        "message": "File uploaded. Preparing analysis...",
        "error": None,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "template": None,
    }

    executor.submit(_run_job_sync, job_id, target_path)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/sample")
async def sample_reel():
    sample_id = "18210285d518"
    sample_tmpl = WORK_DIR / sample_id / "template.json"
    if not sample_tmpl.exists():
        raise HTTPException(status_code=404, detail="Sample template not found")

    job_id = "sample_" + sample_id
    if job_id not in jobs:
        tmpl = json.loads(sample_tmpl.read_text(encoding="utf-8"))
        jobs[job_id] = {
            "job_id": job_id,
            "video_id": sample_id,
            "filename": "sample_reel.mp4",
            "filesize": 16010012,
            "status": "done",
            "current_stage": "merge",
            "stages_status": {s["name"]: "ok" for s in STAGE_META},
            "stage_times": {
                "ingest": 10.5,
                "frames": 2.08,
                "shots": 11.06,
                "motion": 1.72,
                "grade": 0.04,
                "audio": 44.76,
                "speech": 6.08,
                "text": 316.92,
                "merge": 0.01,
            },
            "progress": 100,
            "message": "Sample template loaded instantly!",
            "error": None,
            "created_at": time.time(),
            "started_at": time.time(),
            "finished_at": time.time(),
            "template": tmpl,
        }
    return {"job_id": job_id, "status": "done"}


@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    queue = asyncio.Queue(maxsize=30)
    job_subscribers.setdefault(job_id, []).append(queue)

    async def event_generator():
        # First send immediate current state
        yield f"data: {json.dumps(jobs[job_id])}\n\n"
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=2.0)
                    yield f"data: {data}\n\n"
                    item = json.loads(data)
                    if item.get("status") in ("done", "failed"):
                        break
                except asyncio.TimeoutError:
                    # Heartbeat
                    if job_id in jobs:
                        yield f"data: {json.dumps(jobs[job_id])}\n\n"
                        if jobs[job_id].get("status") in ("done", "failed"):
                            break
        finally:
            if job_id in job_subscribers and queue in job_subscribers[job_id]:
                job_subscribers[job_id].remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/template")
async def get_template(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    tmpl = jobs[job_id].get("template")
    if not tmpl:
        raise HTTPException(status_code=400, detail="Template is not ready yet")
    return tmpl


@app.get("/api/jobs/{job_id}/download")
async def download_template(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    tmpl = jobs[job_id].get("template")
    if not tmpl:
        raise HTTPException(status_code=400, detail="Template is not ready yet")
    content = json.dumps(tmpl, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="template.json"'},
    )


@app.post("/api/template/yaml")
async def generate_yaml(request: Request):
    import yaml
    body = await request.json()
    texts_dict = body.get("texts", {})
    yaml_content = "# Auto-generated texts.yaml for vtpl render\n"
    yaml_content += "# Place in ./mine/texts.yaml along with your clip_1.mp4 and music.mp3\n\n"
    yaml_content += yaml.safe_dump(texts_dict, sort_keys=False, allow_unicode=True)
    return Response(
        content=yaml_content,
        media_type="text/yaml",
        headers={"Content-Disposition": 'attachment; filename="texts.yaml"'},
    )


@app.post("/api/template/decode")
async def decode_template(request: Request):
    body = await request.json()
    tmpl = body.get("template")
    if not tmpl or not isinstance(tmpl, dict):
        raise HTTPException(status_code=400, detail="Invalid template JSON structure")
    
    job_id = "decoded_" + uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "video_id": tmpl.get("meta", {}).get("source_id", "decoded"),
        "filename": "decoded_template.json",
        "filesize": len(json.dumps(tmpl)),
        "status": "done",
        "current_stage": "merge",
        "stages_status": {s["name"]: "ok" for s in STAGE_META},
        "stage_times": {},
        "progress": 100,
        "message": "Template decoded and ready for editing!",
        "error": None,
        "created_at": time.time(),
        "started_at": time.time(),
        "finished_at": time.time(),
        "template": tmpl,
    }
    return {"job_id": job_id, "status": "done", "template": tmpl}


RENDERS_DIR = WORK_DIR / "renders"
RENDERS_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/render")
async def render_video_endpoint(request: Request):
    from vtpl.render.run import render as run_render
    import yaml
    
    body = await request.json()
    tmpl = body.get("template")
    texts = body.get("texts", {})
    job_id = body.get("job_id")
    
    if not tmpl:
        raise HTTPException(status_code=400, detail="Missing template")
    
    render_id = uuid.uuid4().hex[:10]
    render_media_dir = RENDERS_DIR / render_id
    render_media_dir.mkdir(parents=True, exist_ok=True)
    
    template_path = render_media_dir / "template.json"
    template_path.write_text(json.dumps(tmpl, indent=2), encoding="utf-8")
    
    texts_path = render_media_dir / "texts.yaml"
    texts_path.write_text(yaml.safe_dump(texts, sort_keys=False, allow_unicode=True), encoding="utf-8")
    
    source_found = False
    if job_id and job_id in jobs:
        vid = jobs[job_id].get("video_id")
        if vid and (WORK_DIR / vid / "norm.mp4").exists():
            shutil.copy(WORK_DIR / vid / "norm.mp4", render_media_dir / "clip_1.mp4")
            if (WORK_DIR / vid / "audio.wav").exists():
                shutil.copy(WORK_DIR / vid / "audio.wav", render_media_dir / "music.wav")
            source_found = True
            
    if not source_found:
        sample_dir = WORK_DIR / "18210285d518"
        if (sample_dir / "norm.mp4").exists():
            shutil.copy(sample_dir / "norm.mp4", render_media_dir / "clip_1.mp4")
            if (sample_dir / "audio.wav").exists():
                shutil.copy(sample_dir / "audio.wav", render_media_dir / "music.wav")
            source_found = True
            
    if not source_found:
        raise HTTPException(status_code=400, detail="No base video clip found to render. Please upload media.")
        
    out_mp4 = render_media_dir / "out.mp4"
    try:
        loop = asyncio.get_event_loop()
        warnings = await loop.run_in_executor(executor, run_render, template_path, render_media_dir, out_mp4, False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {e}")
        
    if not out_mp4.exists():
        raise HTTPException(status_code=500, detail="Rendering failed to create MP4.")
        
    return {
        "status": "done",
        "render_id": render_id,
        "video_url": f"/api/renders/{render_id}/video",
        "download_url": f"/api/renders/{render_id}/download",
        "warnings": warnings,
        "size": out_mp4.stat().st_size
    }


@app.get("/api/renders/{render_id}/video")
async def get_rendered_video(render_id: str):
    out_mp4 = RENDERS_DIR / render_id / "out.mp4"
    if not out_mp4.exists():
        raise HTTPException(status_code=404, detail="Rendered video not found")
    return FileResponse(out_mp4, media_type="video/mp4")


@app.get("/api/renders/{render_id}/download")
async def download_rendered_video(render_id: str):
    out_mp4 = RENDERS_DIR / render_id / "out.mp4"
    if not out_mp4.exists():
        raise HTTPException(status_code=404, detail="Rendered video not found")
    return FileResponse(out_mp4, media_type="video/mp4", filename="rendered_video.mp4")
