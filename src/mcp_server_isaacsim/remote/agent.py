"""
Remote agent that runs on the Nebius GPU instance.

Exposes a FastAPI HTTP interface for the MCP server to call through an SSH tunnel.
Manages Isaac Lab simulation sessions, training jobs, and asset/sensor operations.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("isaacsim-remote-agent")

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

@dataclass
class TrainingJob:
    job_id: str
    task: str
    framework: str
    process: subprocess.Popen | None = None
    log_path: str = ""
    started_at: float = 0.0
    status: str = "pending"  # pending | running | completed | failed | stopped
    config: dict = field(default_factory=dict)


@dataclass
class SimSession:
    session_id: str
    task: str
    num_envs: int
    created_at: float = 0.0
    step_count: int = 0
    status: str = "active"
    last_obs: dict | None = None
    last_reward: list | None = None
    last_info: dict | None = None


class AgentState:
    """Global mutable state for the remote agent."""

    def __init__(self):
        self.sessions: dict[str, SimSession] = {}
        self.training_jobs: dict[str, TrainingJob] = {}
        self.isaac_lab_path: str = self._find_isaaclab()
        self.log_dir: str = str(Path.home() / "isaaclab_logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def _find_isaaclab(self) -> str:
        """Locate Isaac Lab installation."""
        candidates = [
            os.environ.get("ISAACLAB_PATH", ""),
            str(Path.home() / "IsaacLab"),
            "/opt/isaaclab",
            "/workspace/isaaclab",
            "/workspace/IsaacLab",
        ]
        for p in candidates:
            if p and Path(p).exists() and (Path(p) / "isaaclab.sh").exists():
                return p
        return ""

    @property
    def isaaclab_sh(self) -> str:
        if not self.isaac_lab_path:
            raise RuntimeError(
                "Isaac Lab not found. Set ISAACLAB_PATH env var or install to ~/IsaacLab"
            )
        return str(Path(self.isaac_lab_path) / "isaaclab.sh")


state = AgentState()

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class ConnectRequest(BaseModel):
    isaac_lab_path: str | None = None


class CreateSessionRequest(BaseModel):
    task: str
    num_envs: int = 16
    device: str = "cuda:0"
    headless: bool = True
    enable_cameras: bool = False
    extra_args: dict[str, Any] = {}


class StepRequest(BaseModel):
    session_id: str
    num_steps: int = 1
    actions: list[list[float]] | None = None  # None = random actions


class ResetRequest(BaseModel):
    session_id: str
    seed: int | None = None


class TrainRequest(BaseModel):
    task: str
    framework: str = "skrl"  # skrl | rsl_rl | sb3 | rl_games
    num_envs: int = 4096
    max_iterations: int = 1000
    device: str = "cuda:0"
    headless: bool = True
    seed: int = 42
    checkpoint: str | None = None
    distributed: bool = False
    num_gpus: int = 1
    extra_args: dict[str, str] = {}
    run_name: str | None = None


class EvalRequest(BaseModel):
    task: str
    framework: str = "skrl"
    checkpoint: str
    num_envs: int = 16
    num_steps: int = 200
    record_video: bool = True
    device: str = "cuda:0"


class RunScriptRequest(BaseModel):
    script_path: str
    args: list[str] = []
    headless: bool = True
    timeout: int = 300


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Isaac Lab Remote Agent starting...")
    logger.info("Isaac Lab path: %s", state.isaac_lab_path or "NOT FOUND")
    yield
    # Cleanup: stop all training jobs
    for job in state.training_jobs.values():
        if job.process and job.process.poll() is None:
            job.process.terminate()
    logger.info("Agent shut down.")


app = FastAPI(title="Isaac Lab Remote Agent", version="0.1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Health & info
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    gpu_info = await _get_gpu_info()
    return {
        "status": "ok",
        "isaac_lab_path": state.isaac_lab_path,
        "isaac_lab_found": bool(state.isaac_lab_path),
        "active_sessions": len(state.sessions),
        "active_training_jobs": sum(
            1 for j in state.training_jobs.values() if j.status == "running"
        ),
        "gpu": gpu_info,
    }


@app.get("/info")
async def info():
    return {
        "isaac_lab_path": state.isaac_lab_path,
        "log_dir": state.log_dir,
        "python": sys.executable,
        "sessions": {k: _session_summary(v) for k, v in state.sessions.items()},
        "training_jobs": {k: _job_summary(v) for k, v in state.training_jobs.items()},
    }


# ---------------------------------------------------------------------------
# Environment listing
# ---------------------------------------------------------------------------

@app.get("/envs")
async def list_environments():
    """List all registered Isaac Lab environments."""
    try:
        result = await _run_isaaclab_script(
            "scripts/environments/list_envs.py",
            timeout=60,
        )
        return {"environments": result["stdout"], "return_code": result["return_code"]}
    except Exception as e:
        # Fallback: try to parse from the installed packages
        return {"environments": str(e), "hint": "Could not list environments automatically."}


# ---------------------------------------------------------------------------
# Simulation sessions (interactive stepping)
# ---------------------------------------------------------------------------

@app.post("/session/create")
async def create_session(req: CreateSessionRequest):
    """Create a new interactive simulation session.

    This launches an Isaac Lab environment as a subprocess with a control socket.
    For simplicity, we use a script-based approach where the session runs a
    step server that accepts commands via a local socket.
    """
    session_id = f"sim-{uuid.uuid4().hex[:8]}"

    # Write a temporary session runner script
    session_script = Path(state.log_dir) / f"{session_id}_runner.py"
    session_socket = Path(state.log_dir) / f"{session_id}.sock"
    session_script.write_text(_generate_session_script(
        session_id=session_id,
        task=req.task,
        num_envs=req.num_envs,
        device=req.device,
        headless=req.headless,
        enable_cameras=req.enable_cameras,
        socket_path=str(session_socket),
    ))

    session = SimSession(
        session_id=session_id,
        task=req.task,
        num_envs=req.num_envs,
        created_at=time.time(),
    )
    state.sessions[session_id] = session

    # Launch the session subprocess
    cmd = [state.isaaclab_sh, "-p", str(session_script)]
    env = {**os.environ, "HEADLESS": "1" if req.headless else "0"}

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for the socket to appear (session is ready)
        for _ in range(60):
            if session_socket.exists():
                break
            await asyncio.sleep(1)
        else:
            session.status = "failed"
            stderr = await proc.stderr.read() if proc.stderr else b""
            return {
                "session_id": session_id,
                "status": "failed",
                "error": f"Session did not start within 60s. stderr: {stderr.decode()[-500:]}",
            }

        session.status = "active"
        return {"session_id": session_id, "status": "active", "task": req.task, "num_envs": req.num_envs}

    except Exception as e:
        session.status = "failed"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/step")
async def step_session(req: StepRequest):
    """Step the simulation forward."""
    session = state.sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")

    socket_path = Path(state.log_dir) / f"{req.session_id}.sock"
    try:
        result = await _send_session_command(socket_path, {
            "command": "step",
            "num_steps": req.num_steps,
            "actions": req.actions,
        })
        session.step_count += req.num_steps
        session.last_obs = result.get("obs")
        session.last_reward = result.get("reward")
        session.last_info = result.get("info")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/reset")
async def reset_session(req: ResetRequest):
    """Reset the simulation environment."""
    session = state.sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")

    socket_path = Path(state.log_dir) / f"{req.session_id}.sock"
    try:
        result = await _send_session_command(socket_path, {
            "command": "reset",
            "seed": req.seed,
        })
        session.step_count = 0
        session.last_obs = result.get("obs")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/obs")
async def get_observation(session_id: str):
    """Get the current observation from a session."""
    session = state.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    socket_path = Path(state.log_dir) / f"{session_id}.sock"
    result = await _send_session_command(socket_path, {"command": "get_obs"})
    return result


@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    """Close and clean up a simulation session."""
    session = state.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    socket_path = Path(state.log_dir) / f"{session_id}.sock"
    try:
        await _send_session_command(socket_path, {"command": "close"})
    except Exception:
        pass

    session.status = "closed"
    # Clean up temp files
    for suffix in ["_runner.py", ".sock"]:
        p = Path(state.log_dir) / f"{session_id}{suffix}"
        if p.exists():
            p.unlink()

    del state.sessions[session_id]
    return {"status": "closed", "session_id": session_id}


# ---------------------------------------------------------------------------
# Training jobs
# ---------------------------------------------------------------------------

@app.post("/train/start")
async def start_training(req: TrainRequest):
    """Start a training job."""
    job_id = f"train-{uuid.uuid4().hex[:8]}"

    framework_scripts = {
        "skrl": "scripts/reinforcement_learning/skrl/train.py",
        "rsl_rl": "scripts/reinforcement_learning/rsl_rl/train.py",
        "sb3": "scripts/reinforcement_learning/sb3/train.py",
        "rl_games": "scripts/reinforcement_learning/rl_games/train.py",
    }
    if req.framework not in framework_scripts:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown framework: {req.framework}. Use: {list(framework_scripts.keys())}",
        )

    script = framework_scripts[req.framework]
    log_file = Path(state.log_dir) / f"{job_id}.log"

    args = [
        "--task", req.task,
        "--num_envs", str(req.num_envs),
        "--max_iterations", str(req.max_iterations),
        "--seed", str(req.seed),
        "--headless",
    ]
    if req.checkpoint:
        args.extend(["--checkpoint", req.checkpoint])
    if req.run_name:
        args.extend(["--run_name", req.run_name])
    for k, v in req.extra_args.items():
        args.extend([f"--{k}", str(v)])

    # Build command
    if req.distributed and req.num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={req.num_gpus}",
            str(Path(state.isaac_lab_path) / script),
            "--distributed",
            *args,
        ]
    else:
        cmd = [state.isaaclab_sh, "-p", str(Path(state.isaac_lab_path) / script), *args]

    env = {**os.environ, "HEADLESS": "1"}

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

    job = TrainingJob(
        job_id=job_id,
        task=req.task,
        framework=req.framework,
        process=proc,
        log_path=str(log_file),
        started_at=time.time(),
        status="running",
        config=req.model_dump(),
    )
    state.training_jobs[job_id] = job

    return {
        "job_id": job_id,
        "status": "running",
        "task": req.task,
        "framework": req.framework,
        "log_path": str(log_file),
        "pid": proc.pid,
    }


@app.get("/train/{job_id}")
async def get_training_status(job_id: str):
    """Get the status and recent logs of a training job."""
    job = state.training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Update status from process
    if job.process and job.process.poll() is not None:
        job.status = "completed" if job.process.returncode == 0 else "failed"

    # Read last N lines of log
    tail_lines = []
    if Path(job.log_path).exists():
        with open(job.log_path) as f:
            lines = f.readlines()
            tail_lines = lines[-50:]

    # Find latest checkpoint
    checkpoint = _find_latest_checkpoint(job)

    return {
        "job_id": job_id,
        "status": job.status,
        "task": job.task,
        "framework": job.framework,
        "elapsed_seconds": time.time() - job.started_at if job.started_at else 0,
        "recent_logs": "".join(tail_lines),
        "latest_checkpoint": checkpoint,
        "config": job.config,
    }


@app.get("/train/{job_id}/logs")
async def get_training_logs(job_id: str, tail: int = 100):
    """Get training logs."""
    job = state.training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not Path(job.log_path).exists():
        return {"logs": "", "lines": 0}

    with open(job.log_path) as f:
        lines = f.readlines()
    return {"logs": "".join(lines[-tail:]), "total_lines": len(lines), "showing_last": tail}


@app.post("/train/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a training job."""
    job = state.training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.process and job.process.poll() is None:
        os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
        try:
            job.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(job.process.pid), signal.SIGKILL)
        job.status = "stopped"

    return {"job_id": job_id, "status": job.status}


@app.get("/train")
async def list_training_jobs():
    """List all training jobs."""
    jobs = {}
    for jid, job in state.training_jobs.items():
        if job.process and job.process.poll() is not None:
            job.status = "completed" if job.process.returncode == 0 else "failed"
        jobs[jid] = _job_summary(job)
    return {"jobs": jobs}


# ---------------------------------------------------------------------------
# Evaluate / play
# ---------------------------------------------------------------------------

@app.post("/eval")
async def evaluate_policy(req: EvalRequest):
    """Evaluate a trained policy and optionally record video."""
    framework_scripts = {
        "skrl": "scripts/reinforcement_learning/skrl/play.py",
        "rsl_rl": "scripts/reinforcement_learning/rsl_rl/play.py",
        "sb3": "scripts/reinforcement_learning/sb3/play.py",
        "rl_games": "scripts/reinforcement_learning/rl_games/play.py",
    }
    script = framework_scripts.get(req.framework)
    if not script:
        raise HTTPException(status_code=400, detail=f"Unknown framework: {req.framework}")

    args = [
        "--task", req.task,
        "--num_envs", str(req.num_envs),
        "--checkpoint", req.checkpoint,
        "--headless",
    ]
    if req.record_video:
        args.extend(["--video", "--video_length", str(req.num_steps)])

    result = await _run_isaaclab_script(script, args=args, timeout=300)
    return {
        "status": "completed" if result["return_code"] == 0 else "failed",
        "output": result["stdout"][-2000:],
        "errors": result["stderr"][-1000:] if result["stderr"] else None,
    }


# ---------------------------------------------------------------------------
# Custom script execution
# ---------------------------------------------------------------------------

@app.post("/run")
async def run_script(req: RunScriptRequest):
    """Run an arbitrary Isaac Lab Python script."""
    if not Path(req.script_path).exists():
        # Try relative to isaac lab path
        full_path = Path(state.isaac_lab_path) / req.script_path
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Script not found: {req.script_path}")
        script_path = str(full_path)
    else:
        script_path = req.script_path

    args = list(req.args)
    if req.headless and "--headless" not in args:
        args.append("--headless")

    result = await _run_isaaclab_script(script_path, args=args, timeout=req.timeout, absolute=True)
    return result


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

@app.get("/files/logs")
async def list_log_dirs():
    """List training log directories."""
    log_base = Path.home() / "isaaclab_logs"
    # Also check common Isaac Lab log locations
    dirs = []
    for base in [log_base, Path("logs"), Path(state.isaac_lab_path) / "logs" if state.isaac_lab_path else Path("/nonexistent")]:
        if base.exists():
            for framework_dir in sorted(base.iterdir()):
                if framework_dir.is_dir():
                    for task_dir in sorted(framework_dir.iterdir()):
                        if task_dir.is_dir():
                            for run_dir in sorted(task_dir.iterdir()):
                                if run_dir.is_dir():
                                    dirs.append({
                                        "path": str(run_dir),
                                        "framework": framework_dir.name,
                                        "task": task_dir.name,
                                        "run": run_dir.name,
                                        "files": [f.name for f in run_dir.iterdir()][:20],
                                    })
    return {"log_dirs": dirs[-50:]}


@app.get("/files/checkpoints")
async def list_checkpoints(task: str | None = None, framework: str | None = None):
    """List available checkpoints."""
    checkpoints = []
    log_base = Path("logs")
    if not log_base.exists() and state.isaac_lab_path:
        log_base = Path(state.isaac_lab_path) / "logs"

    if log_base.exists():
        for fdir in log_base.iterdir():
            if not fdir.is_dir():
                continue
            if framework and fdir.name != framework:
                continue
            for tdir in fdir.iterdir():
                if not tdir.is_dir():
                    continue
                if task and task not in tdir.name:
                    continue
                for rdir in sorted(tdir.iterdir(), reverse=True):
                    ckpt_dir = rdir / "checkpoints"
                    if ckpt_dir.exists():
                        for ckpt in sorted(ckpt_dir.iterdir()):
                            checkpoints.append({
                                "path": str(ckpt),
                                "framework": fdir.name,
                                "task": tdir.name,
                                "run": rdir.name,
                                "filename": ckpt.name,
                                "size_mb": round(ckpt.stat().st_size / 1e6, 1),
                            })
    return {"checkpoints": checkpoints[-100:]}


@app.get("/files/read")
async def read_file(path: str, tail: int = 100):
    """Read a file from the remote instance."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if p.suffix in (".png", ".jpg", ".jpeg"):
        data = p.read_bytes()
        return {"type": "image", "encoding": "base64", "data": base64.b64encode(data).decode()}

    text = p.read_text(errors="replace")
    lines = text.splitlines()
    if tail and len(lines) > tail:
        lines = lines[-tail:]
    return {"type": "text", "content": "\n".join(lines), "total_lines": len(text.splitlines())}


@app.get("/files/videos")
async def list_videos():
    """List recorded videos."""
    videos = []
    for base in [Path("logs"), Path(state.isaac_lab_path) / "logs" if state.isaac_lab_path else Path("/nonexistent")]:
        if base.exists():
            for vid in base.rglob("*.mp4"):
                videos.append({
                    "path": str(vid),
                    "size_mb": round(vid.stat().st_size / 1e6, 1),
                    "modified": vid.stat().st_mtime,
                })
    return {"videos": sorted(videos, key=lambda v: v["modified"], reverse=True)[:50]}


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

@app.get("/gpu")
async def gpu_info():
    return await _get_gpu_info()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_gpu_info() -> dict:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        gpus = []
        for line in stdout.decode().strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "name": parts[0],
                    "memory_total_mb": int(parts[1]),
                    "memory_used_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                    "temperature_c": int(parts[5]),
                })
        return {"gpus": gpus}
    except Exception as e:
        return {"gpus": [], "error": str(e)}


async def _run_isaaclab_script(
    script: str,
    args: list[str] | None = None,
    timeout: int = 120,
    absolute: bool = False,
) -> dict:
    """Run a Python script through Isaac Lab's launcher."""
    if absolute:
        script_path = script
    else:
        script_path = str(Path(state.isaac_lab_path) / script)

    cmd = [state.isaaclab_sh, "-p", script_path]
    if args:
        cmd.extend(args)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "HEADLESS": "1"},
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return {"return_code": -1, "stdout": "", "stderr": "Timeout", "timed_out": True}

    return {
        "return_code": proc.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


async def _send_session_command(socket_path: Path, command: dict) -> dict:
    """Send a command to a running session via Unix socket."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        payload = json.dumps(command).encode() + b"\n"
        writer.write(payload)
        await writer.drain()
        response = await asyncio.wait_for(reader.readline(), timeout=120)
        return json.loads(response.decode())
    finally:
        writer.close()
        await writer.wait_closed()


def _generate_session_script(
    session_id: str,
    task: str,
    num_envs: int,
    device: str,
    headless: bool,
    enable_cameras: bool,
    socket_path: str,
) -> str:
    """Generate a Python script that runs an interactive Isaac Lab session with a control socket."""
    return f'''"""Auto-generated interactive session: {session_id}"""
import asyncio
import json
import os
import sys
import signal

# Isaac Lab requires AppLauncher before any other imports
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(
    headless={headless},
    enable_cameras={enable_cameras},
)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np

# Now create the environment
env = gym.make("{task}", num_envs={num_envs}, device="{device}")
obs, info = env.reset()

def tensor_to_list(t):
    """Convert tensor/array to JSON-serializable list."""
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy().tolist()
    if isinstance(t, np.ndarray):
        return t.tolist()
    if isinstance(t, dict):
        return {{k: tensor_to_list(v) for k, v in t.items()}}
    return t

def summarize_obs(obs):
    """Summarize observations to avoid sending huge tensors."""
    if isinstance(obs, dict):
        summary = {{}}
        for k, v in obs.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                arr = v if isinstance(v, np.ndarray) else v.cpu().numpy()
                summary[k] = {{
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "sample": arr[0].tolist() if len(arr) > 0 else [],
                }}
            else:
                summary[k] = tensor_to_list(v)
        return summary
    if isinstance(obs, (torch.Tensor, np.ndarray)):
        arr = obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()
        return {{
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "sample": arr[0].tolist() if len(arr) > 0 else [],
        }}
    return obs

async def handle_client(reader, writer):
    global obs, info, env
    try:
        data = await reader.readline()
        cmd = json.loads(data.decode())

        if cmd["command"] == "step":
            num_steps = cmd.get("num_steps", 1)
            actions_input = cmd.get("actions")
            total_reward = None

            for _ in range(num_steps):
                if actions_input is not None:
                    action = torch.tensor(actions_input, device="{device}")
                else:
                    action = torch.tensor(
                        env.action_space.sample(), device="{device}"
                    )
                obs, reward, terminated, truncated, info = env.step(action)
                r = reward.cpu().numpy().tolist() if isinstance(reward, torch.Tensor) else reward
                if total_reward is None:
                    total_reward = r
                else:
                    total_reward = [a + b for a, b in zip(total_reward, r)]

            response = {{
                "obs": summarize_obs(obs),
                "reward": total_reward,
                "terminated": tensor_to_list(terminated),
                "truncated": tensor_to_list(truncated),
                "info": {{k: tensor_to_list(v) for k, v in info.items()}} if isinstance(info, dict) else str(info),
                "step_count": num_steps,
            }}

        elif cmd["command"] == "reset":
            seed = cmd.get("seed")
            if seed is not None:
                obs, info = env.reset(seed=seed)
            else:
                obs, info = env.reset()
            response = {{"obs": summarize_obs(obs), "status": "reset"}}

        elif cmd["command"] == "get_obs":
            response = {{
                "obs": summarize_obs(obs),
                "action_space": {{
                    "shape": list(env.action_space.shape),
                    "low": env.action_space.low.tolist() if hasattr(env.action_space, "low") else None,
                    "high": env.action_space.high.tolist() if hasattr(env.action_space, "high") else None,
                }},
                "observation_space": str(env.observation_space),
            }}

        elif cmd["command"] == "close":
            env.close()
            simulation_app.close()
            response = {{"status": "closed"}}

        else:
            response = {{"error": f"Unknown command: {{cmd['command']}}"}}

        writer.write(json.dumps(response).encode() + b"\\n")
        await writer.drain()

        if cmd["command"] == "close":
            await asyncio.sleep(0.5)
            sys.exit(0)

    except Exception as e:
        error_resp = json.dumps({{"error": str(e)}}).encode() + b"\\n"
        writer.write(error_resp)
        await writer.drain()
    finally:
        writer.close()

async def main():
    # Remove socket if it exists
    import pathlib
    sock = pathlib.Path("{socket_path}")
    if sock.exists():
        sock.unlink()

    server = await asyncio.start_unix_server(handle_client, path="{socket_path}")
    print(f"Session {{"{session_id}"}} ready on {socket_path}", flush=True)

    def shutdown(sig, frame):
        env.close()
        simulation_app.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)

    async with server:
        await server.serve_forever()

asyncio.run(main())
'''


def _find_latest_checkpoint(job: TrainingJob) -> str | None:
    """Find the most recent checkpoint for a training job."""
    log_base = Path("logs")
    if not log_base.exists() and state.isaac_lab_path:
        log_base = Path(state.isaac_lab_path) / "logs"

    if not log_base.exists():
        return None

    framework_dir = log_base / job.framework
    if not framework_dir.exists():
        return None

    # Find the most recent run dir for this task
    latest = None
    latest_time = 0
    for task_dir in framework_dir.iterdir():
        if job.task.replace("-", "_").lower() in task_dir.name.replace("-", "_").lower():
            for run_dir in task_dir.iterdir():
                ckpt_dir = run_dir / "checkpoints"
                if ckpt_dir.exists():
                    for ckpt in ckpt_dir.iterdir():
                        mtime = ckpt.stat().st_mtime
                        if mtime > latest_time:
                            latest_time = mtime
                            latest = str(ckpt)
    return latest


def _session_summary(s: SimSession) -> dict:
    return {
        "session_id": s.session_id,
        "task": s.task,
        "num_envs": s.num_envs,
        "status": s.status,
        "step_count": s.step_count,
        "age_seconds": round(time.time() - s.created_at),
    }


def _job_summary(j: TrainingJob) -> dict:
    return {
        "job_id": j.job_id,
        "task": j.task,
        "framework": j.framework,
        "status": j.status,
        "elapsed_seconds": round(time.time() - j.started_at) if j.started_at else 0,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    port = int(os.environ.get("ISAACSIM_AGENT_PORT", "8421"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
