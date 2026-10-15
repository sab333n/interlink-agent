"""
Tiny HTTP server that lets the InterLink frontend trigger the AI apply agent.

Endpoints
---------
POST  /api/agent/run     → spawn `python agent.py` as a background process.
                           Body JSON: { profile, resume_b64, resume_name,
                                        apply_url, target_company, dry_run }
GET   /api/agent/status  → { running: bool, pid: int|null, log_tail: str }
GET   /api/agent/log     → full text log of the latest run (tail -N lines)
POST  /api/agent/stop    → kill the running agent process (if any)

Run it with:
    cd agent && source .venv/bin/activate
    python server.py                    # http://localhost:8001

Then click the "Run AI Agent" button on the jobs page.
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import tempfile
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
AGENT_SCRIPT = ROOT / "agent.py"
QUEUE_SCRIPT = ROOT / "queue_runner.py"
LOG_FILE = ROOT / "agent_run.log"

app = FastAPI(title="InterLink AI Agent API")

from inline_apply import router as inline_apply_router  # noqa: E402
from email_finder import router as email_finder_router  # noqa: E402
from chat import router as chat_router  # noqa: E402
import imessage  # noqa: E402
from extension import router as extension_router  # noqa: E402
from auth import auth_required  # noqa: E402

# Hosted, the agent serves many users, so the single shared local queue (which
# drives a visible browser on this machine) is switched off; in-site apply is used instead.
HOSTED = auth_required()
LOCAL_ONLY = {"/api/agent/run", "/api/agent/queue", "/api/agent/log", "/api/agent/stop"}


@app.middleware("http")
async def block_local_only(request: Request, call_next):
    if HOSTED and request.url.path in LOCAL_ONLY:
        return JSONResponse({"detail": "Not available on the hosted agent."}, status_code=403)
    return await call_next(request)

app.include_router(inline_apply_router)
app.include_router(email_finder_router)
app.include_router(chat_router)
app.include_router(imessage.router)
app.include_router(extension_router)


@app.on_event("startup")
async def _start_imessage() -> None:
    imessage.start_background()

app.add_middleware(
    CORSMiddleware,
    # Allow any localhost / 127.0.0.1 port — the Vite dev server may land on
    # 5173, 5174, 5175, etc. depending on what's free. We use a regex so we
    # don't have to maintain an explicit allow-list.
    allow_origin_regex=os.environ.get("AGENT_ALLOWED_ORIGINS")
    or r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    target_company: Optional[str] = None
    apply_url: Optional[str] = None
    dry_run: Optional[bool] = True  # default: safe — never clicks final Submit
    # The signed-in user's profile (agent format) and resume, sent on every run.
    profile: Optional[dict[str, Any]] = None
    resume_b64: Optional[str] = None
    resume_name: Optional[str] = None


class QueueJob(BaseModel):
    id: str
    company: str
    role: str
    apply_url: Optional[str] = None


class QueueRequest(BaseModel):
    jobs: list[QueueJob]
    profile: dict[str, Any]
    resume_b64: Optional[str] = None
    resume_name: Optional[str] = None


RUN_DIR = Path(tempfile.gettempdir()) / "interlink-agent-run"


def _write_run_inputs(req: RunRequest | QueueRequest, env: dict[str, str]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for old in RUN_DIR.iterdir():
        old.unlink(missing_ok=True)
    profile_path = RUN_DIR / "profile.json"
    profile_path.write_text(json.dumps(req.profile), encoding="utf-8")
    env["PROFILE_PATH"] = str(profile_path)
    if req.profile and req.profile.get("email"):
        env["APPLICANT_EMAIL"] = str(req.profile["email"])
    if req.resume_b64:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", req.resume_name or "resume.pdf") or "resume.pdf"
        resume_path = RUN_DIR / name
        resume_path.write_bytes(base64.b64decode(req.resume_b64))
        env["RESUME_PATH"] = str(resume_path)


class AgentState:
    proc: Optional[subprocess.Popen] = None
    tail: deque[str] = deque(maxlen=400)
    lock = threading.Lock()
    mode: str = "single"
    # Queue progress keyed by job id: {status, detail}
    jobs: dict[str, dict[str, str]] = {}
    queue_done: bool = False

    @classmethod
    def is_running(cls) -> bool:
        return cls.proc is not None and cls.proc.poll() is None


def _reader_thread(proc: subprocess.Popen, fh):
    """Read the agent's stdout line-by-line, echo to a log file + in-memory tail."""
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if line.startswith("@@JOB "):
                try:
                    ev = json.loads(line[6:])
                    AgentState.jobs[ev["id"]] = {"status": ev["status"], "detail": ev.get("detail", "")}
                except Exception:
                    pass
                continue
            if line.startswith("@@DONE"):
                AgentState.queue_done = True
            AgentState.tail.append(line)
            try:
                fh.write(line + "\n")
                fh.flush()
            except Exception:
                pass
    except Exception as e:
        AgentState.tail.append(f"[server] reader thread error: {e}")
    finally:
        try:
            fh.close()
        except Exception:
            pass


@app.post("/api/agent/run")
def run_agent(req: RunRequest):
    with AgentState.lock:
        if AgentState.is_running():
            return {
                "ok": False,
                "running": True,
                "pid": AgentState.proc.pid if AgentState.proc else None,
                "message": "Agent is already running.",
            }

        if not AGENT_SCRIPT.exists():
            raise HTTPException(500, f"agent.py not found at {AGENT_SCRIPT}")

        if not req.profile:
            raise HTTPException(400, "profile is required — sign in and complete your profile first")

        env = os.environ.copy()
        _write_run_inputs(req, env)
        if req.target_company:
            env["TARGET_COMPANY"] = req.target_company
        if req.apply_url:
            env["APPLY_URL"] = req.apply_url
        env["AGENT_DRY_RUN"] = "1" if (req.dry_run is None or req.dry_run) else "0"
        if req.dry_run is False:
            env["AGENT_SUBMIT"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        AgentState.mode = "single"
        AgentState.jobs = {}
        python_exe = sys.executable
        cmd = [python_exe, str(AGENT_SCRIPT)]

        log_fh = open(LOG_FILE, "w", encoding="utf-8", buffering=1)

        try:
            AgentState.tail.clear()
            AgentState.tail.append(
                f"[server] spawning: {' '.join(cmd)}"
            )
            AgentState.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            log_fh.close()
            raise HTTPException(500, f"failed to start agent: {e}")

        t = threading.Thread(
            target=_reader_thread, args=(AgentState.proc, log_fh), daemon=True
        )
        t.start()

        return {
            "ok": True,
            "running": True,
            "pid": AgentState.proc.pid,
            "message": f"Agent started (PID {AgentState.proc.pid}).",
        }


@app.post("/api/agent/queue")
def run_queue(req: QueueRequest):
    """Fill every queued application in review mode (never clicks final Submit)."""
    with AgentState.lock:
        if AgentState.is_running():
            return {"ok": False, "running": True, "message": "Agent is already running."}
        if not req.jobs:
            raise HTTPException(400, "queue is empty")

        env = os.environ.copy()
        _write_run_inputs(req, env)
        jobs_path = RUN_DIR / "jobs.json"
        jobs_path.write_text(json.dumps([j.model_dump() for j in req.jobs]), encoding="utf-8")
        env["JOBS_PATH"] = str(jobs_path)
        env["AGENT_DRY_RUN"] = "1"
        env.pop("AGENT_SUBMIT", None)
        env["PYTHONUNBUFFERED"] = "1"

        AgentState.mode = "queue"
        AgentState.queue_done = False
        AgentState.jobs = {j.id: {"status": "queued", "detail": ""} for j in req.jobs}
        AgentState.tail.clear()
        log_fh = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
        try:
            AgentState.proc = subprocess.Popen(
                [sys.executable, str(QUEUE_SCRIPT)],
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            log_fh.close()
            raise HTTPException(500, f"failed to start queue: {e}")
        threading.Thread(target=_reader_thread, args=(AgentState.proc, log_fh), daemon=True).start()
        return {"ok": True, "running": True, "message": f"Queue started with {len(req.jobs)} job(s)."}


@app.get("/api/agent/status")
def agent_status():
    if HOSTED:
        return {"running": False, "hosted": True, "ai_enabled": bool(os.environ.get("XAI_API_KEY"))}
    running = AgentState.is_running()
    exit_code = None
    if AgentState.proc is not None and not running:
        exit_code = AgentState.proc.returncode
    return {
        "running": running,
        "pid": AgentState.proc.pid if AgentState.proc else None,
        "exit_code": exit_code,
        "mode": AgentState.mode,
        "jobs": AgentState.jobs,
        "queue_done": AgentState.queue_done,
        "ai_enabled": bool(os.environ.get("XAI_API_KEY")),
        "log_tail": "\n".join(list(AgentState.tail)[-60:]),
    }


@app.get("/api/agent/log")
def agent_log(lines: int = 400):
    return {
        "lines": list(AgentState.tail)[-max(1, min(lines, 2000)) :],
    }


@app.post("/api/agent/stop")
def agent_stop():
    with AgentState.lock:
        if not AgentState.is_running():
            return {"ok": True, "running": False, "message": "No agent running."}
        try:
            assert AgentState.proc is not None
            AgentState.proc.send_signal(signal.SIGTERM)
            for _ in range(10):
                if AgentState.proc.poll() is not None:
                    break
                time.sleep(0.3)
            if AgentState.proc.poll() is None:
                AgentState.proc.kill()
            return {"ok": True, "running": False, "message": "Agent stopped."}
        except Exception as e:
            raise HTTPException(500, f"stop failed: {e}")


@app.get("/api/agent/health")
def health():
    return {"ok": True, "service": "interlink-agent"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT") or os.environ.get("AGENT_SERVER_PORT", "8001"))
    print(f"\n🚀  InterLink Agent API listening on http://localhost:{port}")
    print("    POST /api/agent/run       → start the agent")
    print("    GET  /api/agent/status    → check progress")
    print("    POST /api/agent/stop      → kill the agent")
    print("    GET  /api/agent/log       → recent log lines\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
