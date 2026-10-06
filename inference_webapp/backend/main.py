"""FastAPI backend for live inference with the archived Stage 2 checkpoint.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Expects the checkpoint at $CHECKPOINT_DIR (default ./checkpoint), with
train_state.pt and a qwen_lora/ folder inside it -- see the README in
this folder for exactly what to download and where to put it.
"""
from __future__ import annotations

import io
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drive_utils import download_drive_file
from pipeline import pipeline

app = FastAPI(title="Duplex Speech LM -- Live Inference")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

jobs_lock = threading.Lock()
jobs: Dict[str, dict] = {}


class JobStatus(BaseModel):
    status: str  # "running" | "done" | "error"
    progress: float
    error: Optional[str] = None
    stage: Optional[str] = None


@app.on_event("startup")
def load_model_on_startup() -> None:
    print("Loading model -- this downloads/loads WavLM, Qwen2.5-1.5B, Vocos, and your checkpoint. " "May take a few minutes the first time.")
    try:
        pipeline.load()
    except FileNotFoundError as e:
        print(f"[startup] WARNING: model not loaded yet -- {e}")
        print("[startup] the server will still start; /api/health will report not-ready until you fix this and restart.")


@app.get("/api/health")
def health():
    return {
        "ready": pipeline._loaded,
        "checkpoint_step": pipeline.step,
        "device": str(pipeline.device) if pipeline._loaded else None,
        "checkpoint_dir": str(pipeline.checkpoint_dir),
    }


def _read_wav_bytes(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    """soundfile, not torchaudio.load, for reading from an in-memory buffer --
    more reliably documented for file-like objects across versions than
    torchaudio's backend dispatch."""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)  # (n_samples, n_channels)
    waveform = torch.from_numpy(data).mean(dim=1)  # mono, (n_samples,)
    return waveform, sr


def _write_wav_bytes(waveform: torch.Tensor, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, waveform.numpy(), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _run_generation(job_id: str, wav_bytes: bytes, window_s: float, max_duration_s: float) -> None:
    try:
        waveform, sr = _read_wav_bytes(wav_bytes)

        def on_progress(frac: float) -> None:
            with jobs_lock:
                jobs[job_id]["progress"] = frac

        output = pipeline.generate(waveform, sr, window_s=window_s, max_duration_s=max_duration_s, progress_cb=on_progress)

        with jobs_lock:
            jobs[job_id].update(
                status="done",
                progress=1.0,
                input_audio=_write_wav_bytes(waveform, sr),
                output_audio=_write_wav_bytes(output, 24000),
            )
    except Exception as e:  # noqa: BLE001 -- surface any failure to the polling client, don't crash the thread silently
        with jobs_lock:
            jobs[job_id].update(status="error", error=str(e))


@app.post("/api/generate")
async def generate(
    file: UploadFile = File(...),
    window_s: float = Form(0.4),
    max_duration_s: float = Form(30.0),
):
    if not pipeline._loaded:
        raise HTTPException(503, "Model is not loaded -- check /api/health and the server console for why.")

    wav_bytes = await file.read()
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "running", "progress": 0.0}

    thread = threading.Thread(target=_run_generation, args=(job_id, wav_bytes, window_s, max_duration_s), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        return JobStatus(status=job["status"], progress=job["progress"], error=job.get("error"), stage=job.get("stage"))


@app.get("/api/jobs/{job_id}/input_audio")
def job_input_audio(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None or "input_audio" not in job:
        raise HTTPException(404, "Audio not available yet")
    return Response(content=job["input_audio"], media_type="audio/wav")


@app.get("/api/jobs/{job_id}/output_audio")
def job_output_audio(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None or "output_audio" not in job:
        raise HTTPException(404, "Audio not available yet")
    return Response(content=job["output_audio"], media_type="audio/wav")


@app.get("/api/jobs/{job_id}/ground_truth_audio")
def job_ground_truth_audio(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None or "ground_truth_audio" not in job:
        raise HTTPException(404, "Audio not available yet (only present for teacher-forced jobs)")
    return Response(content=job["ground_truth_audio"], media_type="audio/wav")


def _run_teacher_forced(job_id: str, user_bytes: bytes, agent_bytes: bytes) -> None:
    try:
        user_wave, user_sr = _read_wav_bytes(user_bytes)
        agent_wave, agent_sr = _read_wav_bytes(agent_bytes)
        if user_sr != agent_sr:
            raise ValueError(f"User clip ({user_sr}Hz) and agent clip ({agent_sr}Hz) must share a sample rate.")
        n = min(user_wave.shape[0], agent_wave.shape[0])
        user_wave, agent_wave = user_wave[:n], agent_wave[:n]

        output = pipeline.generate_teacher_forced(user_wave, agent_wave, user_sr)

        with jobs_lock:
            jobs[job_id].update(
                status="done",
                progress=1.0,
                input_audio=_write_wav_bytes(user_wave, user_sr),
                output_audio=_write_wav_bytes(output, 24000),
                ground_truth_audio=_write_wav_bytes(agent_wave, agent_sr),
            )
    except Exception as e:  # noqa: BLE001
        with jobs_lock:
            jobs[job_id].update(status="error", error=str(e))


@app.post("/api/generate_teacher_forced")
async def generate_teacher_forced(user_file: UploadFile = File(...), agent_file: UploadFile = File(...)):
    """Diagnostic mode: requires the REAL agent audio as input (teacher
    forcing), not something available at real deployment time. See
    pipeline.generate_teacher_forced's docstring for why this exists --
    this checkpoint's self-feedback output was noisy, but this path
    confirmed the model/training weren't the bottleneck, the duplex-fusion
    design was. Both clips must be the same real two-channel recording's
    user/agent tracks over the SAME time window."""
    if not pipeline._loaded:
        raise HTTPException(503, "Model is not loaded -- check /api/health and the server console for why.")

    user_bytes = await user_file.read()
    agent_bytes = await agent_file.read()
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "running", "progress": 0.0}

    thread = threading.Thread(target=_run_teacher_forced, args=(job_id, user_bytes, agent_bytes), daemon=True)
    thread.start()

    return {"job_id": job_id}


class TeacherForcedFromDriveRequest(BaseModel):
    user_drive_link: str
    agent_drive_link: str
    offset_s: float = 60.0
    duration_s: float = 8.0


def _read_wav_segment(path: Path, offset_s: float, duration_s: float) -> tuple[torch.Tensor, int]:
    info = sf.info(str(path))
    start = int(offset_s * info.samplerate)
    frames = int(duration_s * info.samplerate)
    if start >= info.frames:
        raise ValueError(
            f"offset_s={offset_s} is past the end of this file ({info.frames / info.samplerate:.1f}s long)."
        )
    data, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError(f"Got 0 frames reading offset_s={offset_s}, duration_s={duration_s} -- check these values.")
    return torch.from_numpy(data).mean(dim=1), sr


def _run_teacher_forced_from_drive(
    job_id: str, user_link: str, agent_link: str, offset_s: float, duration_s: float
) -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="drive_dl_"))
    try:
        with jobs_lock:
            jobs[job_id]["stage"] = "downloading user clip from Drive..."
        user_path = download_drive_file(user_link, tmp_dir)

        with jobs_lock:
            jobs[job_id]["stage"] = "downloading agent clip from Drive..."
        agent_path = download_drive_file(agent_link, tmp_dir)

        with jobs_lock:
            jobs[job_id]["stage"] = "trimming and running inference..."
        user_wave, user_sr = _read_wav_segment(user_path, offset_s, duration_s)
        agent_wave, agent_sr = _read_wav_segment(agent_path, offset_s, duration_s)
        if user_sr != agent_sr:
            raise ValueError(f"User clip ({user_sr}Hz) and agent clip ({agent_sr}Hz) must share a sample rate.")
        n = min(user_wave.shape[0], agent_wave.shape[0])
        user_wave, agent_wave = user_wave[:n], agent_wave[:n]

        output = pipeline.generate_teacher_forced(user_wave, agent_wave, user_sr)

        with jobs_lock:
            jobs[job_id].update(
                status="done",
                progress=1.0,
                input_audio=_write_wav_bytes(user_wave, user_sr),
                output_audio=_write_wav_bytes(output, 24000),
                ground_truth_audio=_write_wav_bytes(agent_wave, agent_sr),
            )
    except Exception as e:  # noqa: BLE001
        with jobs_lock:
            jobs[job_id].update(status="error", error=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # never leave full-session downloads lying around


@app.post("/api/generate_teacher_forced_from_drive")
async def generate_teacher_forced_from_drive(req: TeacherForcedFromDriveRequest):
    """Same as /api/generate_teacher_forced, but pulls the user/agent clips
    directly from Google Drive share links instead of a browser upload, and
    trims them to [offset_s, offset_s + duration_s) itself -- saves you from
    manually downloading a full session and cutting a clip out of it first.

    Requires both files to be shared as "Anyone with the link" -- gdown has
    no OAuth session to use, this is the tradeoff for not needing one."""
    if not pipeline._loaded:
        raise HTTPException(503, "Model is not loaded -- check /api/health and the server console for why.")

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "running", "progress": 0.0, "stage": "starting..."}

    thread = threading.Thread(
        target=_run_teacher_forced_from_drive,
        args=(job_id, req.user_drive_link, req.agent_drive_link, req.offset_s, req.duration_s),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
