# Decoupled Deployment Guide: Vercel (Frontend) + Render (Backend)

This guide explains how to host **vtpl** (Video Template Extractor & Studio) for **100% free** using a decoupled architecture:
- **Frontend**: Hosted on [Vercel](https://vercel.com) (Edge CDN, instantaneous static page delivery, SEO-optimized, customizable typography & color palette editor, in-browser MP4 player).
- **Backend**: Hosted on [Render](https://render.com) (Docker container with FFmpeg, libass, OpenCV, Librosa, and Whisper AI).

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │          Vercel (Free Tier)            │
                      │  https://your-project.vercel.app       │
                      │  • Global Edge CDN (Ultra-fast)        │
                      │  • SEO Tags, AdSense, Studio UI        │
                      │  • Preset Palettes & MP4 Preview       │
                      │  • Instant 50MB Client-side Validation │
                      └──────────────────┬─────────────────────┘
                                         │
                 Direct API Calls / SSE  │  (CORS Enabled: allow_origins=["*"])
                                         ▼
                      ┌────────────────────────────────────────┐
                      │          Render (Free Tier)            │
                      │  https://your-backend.onrender.com     │
                      │  • Docker Container (512MB RAM)        │
                      │  • FFmpeg + libass Typography          │
                      │  • Whisper AI (tiny model: ~75MB RAM)  │
                      │  • Scene Detect, Beat & Motion Engine  │
                      │  • Early 50MB Content-Length Abort     │
                      └────────────────────────────────────────┘
```

---

## 50 MB Upload Validation (Frontend & Backend)

To protect the server against large file uploads (e.g. 100MB or 1,000MB files that would exceed Render's 512MB memory limit):

1. **Frontend Validation (`frontend/index.html`)**:
   - Checks `file.size` immediately when a file is selected or dragged into the dropzone.
   - If `file.size > 50 * 1024 * 1024`, it rejects the file **before any network bytes are sent**, alerting the user with an exact size readout.
   - Verifies valid video file extensions (`.mp4`, `.mov`, `.m4v`, `.webm`, `.mkv`).
   - Rejects empty 0-byte files.

2. **Backend Validation (`vtpl/web/app.py`)**:
   - Inspects `request.headers.get("content-length")` upfront. If the client attempts to send $> 50\text{ MB}$, the server immediately aborts with **HTTP 413 (Payload Too Large)** without reading the body.
   - During stream chunk writing, validates accumulated size and aborts if it exceeds 50MB, deleting any partial file immediately.
   - Rejects empty files with **HTTP 400**.

---

## Step 1: Deploy Backend to Render (Free Version)

1. Sign in to [render.com](https://render.com) using your GitHub account.
2. Click **New +** $\to$ **Web Service**.
3. Select your repository: `viraldangarwork-pixel/video-template-extractor`.
4. Configure service details:
   - **Name**: `vtpl-backend`
   - **Runtime**: `Docker`
   - **Instance Type**: `Free` ($0/month, 512MB RAM)
   - **Health Check Path**: `/healthz`
5. Verify Environment Variables:
   - `PORT`: `8000`
   - `VTPL_WHISPER_MODEL`: `tiny` (uses only ~75MB RAM to prevent 512MB OOM)
   - `MAX_UPLOAD_MB`: `50`
   - `MAX_CONCURRENT_JOBS`: `1`
6. Click **Create Web Service**. Wait 3–5 minutes for the Docker image to build.
7. Once live, copy your backend URL (e.g. `https://vtpl-backend.onrender.com`).

---

## Step 2: Deploy Frontend to Vercel (Free Version)

1. Sign in to [vercel.com](https://vercel.com) using your GitHub account.
2. Click **Add New...** $\to$ **Project**.
3. Import `viraldangarwork-pixel/video-template-extractor`.
4. In Build Settings:
   - **Root Directory**: `frontend` (or keep `./` as root `vercel.json` already points to `frontend`)
5. Click **Deploy**.
6. Your web studio is live on `https://<your-project>.vercel.app`!

---

## Step 3: Link Frontend to Backend via Environment Variable (.env)

Choose any of the following methods:

- **Method A (Vercel Environment Variable - Recommended)**:
  1. In Vercel, go to your project $\to$ **Settings** $\to$ **Environment Variables**.
  2. Add:
     - **Key**: `VTPL_BACKEND_URL`
     - **Value**: `https://vtpl-backend.onrender.com` *(your live Render backend URL)*
  3. Redeploy. Vercel's build step automatically injects this into `config.js`!

- **Method B (Local or File-based `.env`)**:
  Create `frontend/.env`:
  ```env
  VTPL_BACKEND_URL=https://vtpl-backend.onrender.com
  ```
  Run `npm run build` (or `node build.js`) in `frontend/`.

- **Method C (Direct `frontend/config.js` or `vercel.json` rewrites)**:
  You can set `window.VTPL_BACKEND_URL = "https://vtpl-backend.onrender.com";` directly in `frontend/config.js` or let `vercel.json` rewrites proxy `/api/*` calls.

---

## Render Free Tier Sleeping & Keep-Awake

- Render free tier instances sleep after 15 minutes of inactivity. First wake-up takes ~30–50s.
- To prevent cold starts, you can set up a free uptime monitor (e.g., [UptimeRobot](https://uptimerobot.com)) to ping `https://<your-service>.onrender.com/healthz` (supports both `GET` and `HEAD` methods) every 10 minutes.
