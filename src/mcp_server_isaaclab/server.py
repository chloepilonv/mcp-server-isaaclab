"""
MCP Server for NVIDIA Isaac Lab.

Runs locally and communicates with a remote agent on a Nebius GPU instance
through an SSH tunnel. Exposes tools for environment management, training,
evaluation, asset browsing, and more.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap

from mcp.server.fastmcp import FastMCP

from mcp_server_isaaclab.connection import connection

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Isaac Lab",
    instructions=textwrap.dedent("""\
        MCP server for NVIDIA Isaac Lab robotics simulation.
        Isaac Lab runs on a remote Nebius GPU instance. You must connect first
        with `connect_instance`, then use the other tools to interact with it.

        Typical workflow:
        1. connect_instance — establish SSH tunnel to Nebius
        2. list_environments — see available tasks
        3. start_training / create_session — train or interactively step
        4. monitor_training / step_session — observe progress
        5. list_checkpoints / evaluate_policy — use trained models

        All heavy computation runs on the GPU instance. This server is a bridge.
    """),
)


# ===========================================================================
# Connection management
# ===========================================================================

@mcp.tool()
async def connect_instance(
    host: str,
    user: str = "ubuntu",
    key_path: str | None = None,
) -> str:
    """Connect to a Nebius GPU instance running the Isaac Lab remote agent.

    This establishes an SSH tunnel for all subsequent communication.
    The remote agent (`isaacsim-remote-agent`) must be running on the instance.

    Args:
        host: IP address or hostname of the Nebius instance
        user: SSH username (default: ubuntu)
        key_path: Path to SSH private key file (optional, uses default keys if not set)
    """
    result = await connection.connect(host=host, user=user, key_path=key_path)
    return json.dumps(result, indent=2)


@mcp.tool()
async def disconnect_instance() -> str:
    """Disconnect from the Nebius instance and tear down the SSH tunnel."""
    result = await connection.disconnect()
    return json.dumps(result, indent=2)


@mcp.tool()
async def instance_status() -> str:
    """Check the status of the remote instance, GPU utilization, and active jobs."""
    if not connection.connected:
        return json.dumps({"status": "disconnected", "hint": "Use connect_instance first."})
    try:
        health = await connection.get("/health")
        return json.dumps(health, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ===========================================================================
# Environment discovery
# ===========================================================================

@mcp.tool()
async def list_environments() -> str:
    """List all available Isaac Lab environments/tasks.

    Returns registered Gymnasium environment IDs like 'Isaac-Cartpole-v0',
    'Isaac-Ant-v0', 'Isaac-Lift-Cube-Franka-v0', etc.
    """
    result = await connection.get("/envs")
    return json.dumps(result, indent=2)


# ===========================================================================
# Interactive simulation sessions
# ===========================================================================

@mcp.tool()
async def create_session(
    task: str,
    num_envs: int = 16,
    device: str = "cuda:0",
    enable_cameras: bool = False,
) -> str:
    """Create an interactive simulation session for stepping through an environment.

    Use this for hands-on exploration — stepping the sim, observing states,
    testing actions. For training, use start_training instead.

    Args:
        task: Environment ID (e.g. 'Isaac-Cartpole-v0', 'Isaac-Lift-Cube-Franka-v0')
        num_envs: Number of parallel environments (default: 16)
        device: Compute device (default: 'cuda:0')
        enable_cameras: Enable camera sensors (slower but needed for vision tasks)
    """
    result = await connection.post("/session/create", json={
        "task": task,
        "num_envs": num_envs,
        "device": device,
        "headless": True,
        "enable_cameras": enable_cameras,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def step_session(
    session_id: str,
    num_steps: int = 1,
    actions: list[list[float]] | None = None,
) -> str:
    """Step the simulation forward.

    Args:
        session_id: Session ID from create_session
        num_steps: Number of steps to advance (default: 1)
        actions: Actions to apply — shape [num_envs, action_dim]. If None, uses random actions.

    Returns observation summary (shapes, stats, sample), rewards, terminated/truncated flags.
    """
    result = await connection.post("/session/step", json={
        "session_id": session_id,
        "num_steps": num_steps,
        "actions": actions,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def reset_session(session_id: str, seed: int | None = None) -> str:
    """Reset the simulation environment to initial state.

    Args:
        session_id: Session ID from create_session
        seed: Random seed for reproducibility (optional)
    """
    result = await connection.post("/session/reset", json={
        "session_id": session_id,
        "seed": seed,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_observation(session_id: str) -> str:
    """Get the current observation and action/observation space info from a session.

    Args:
        session_id: Session ID from create_session
    """
    result = await connection.get(f"/session/{session_id}/obs")
    return json.dumps(result, indent=2)


@mcp.tool()
async def close_session(session_id: str) -> str:
    """Close an interactive simulation session and free GPU resources.

    Args:
        session_id: Session ID from create_session
    """
    result = await connection.delete(f"/session/{session_id}")
    return json.dumps(result, indent=2)


# ===========================================================================
# Training
# ===========================================================================

@mcp.tool()
async def start_training(
    task: str,
    framework: str = "skrl",
    num_envs: int = 4096,
    max_iterations: int = 1000,
    device: str = "cuda:0",
    seed: int = 42,
    checkpoint: str | None = None,
    distributed: bool = False,
    num_gpus: int = 1,
    run_name: str | None = None,
    extra_args: dict[str, str] | None = None,
) -> str:
    """Start a reinforcement learning training job.

    This runs asynchronously on the GPU instance. Use monitor_training to check progress.

    Args:
        task: Environment ID (e.g. 'Isaac-Ant-v0')
        framework: RL framework — 'skrl' | 'rsl_rl' | 'sb3' | 'rl_games'
        num_envs: Number of parallel environments (default: 4096)
        max_iterations: Training iterations (default: 1000)
        device: Compute device (default: 'cuda:0')
        seed: Random seed (default: 42)
        checkpoint: Path to checkpoint to resume from (optional)
        distributed: Use multi-GPU distributed training (default: False)
        num_gpus: Number of GPUs for distributed training (default: 1)
        run_name: Custom name for this run (optional)
        extra_args: Additional CLI arguments as key-value pairs (optional)
    """
    payload = {
        "task": task,
        "framework": framework,
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "device": device,
        "seed": seed,
        "distributed": distributed,
        "num_gpus": num_gpus,
        "extra_args": extra_args or {},
    }
    if checkpoint:
        payload["checkpoint"] = checkpoint
    if run_name:
        payload["run_name"] = run_name

    result = await connection.post("/train/start", json=payload)
    return json.dumps(result, indent=2)


@mcp.tool()
async def monitor_training(job_id: str) -> str:
    """Get status, recent logs, and latest checkpoint of a training job.

    Args:
        job_id: Job ID from start_training
    """
    result = await connection.get(f"/train/{job_id}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_training_logs(job_id: str, tail: int = 100) -> str:
    """Get training logs from a job.

    Args:
        job_id: Job ID from start_training
        tail: Number of lines from the end to return (default: 100)
    """
    result = await connection.get(f"/train/{job_id}/logs", params={"tail": tail})
    return json.dumps(result, indent=2)


@mcp.tool()
async def stop_training(job_id: str) -> str:
    """Stop a running training job.

    Args:
        job_id: Job ID from start_training
    """
    result = await connection.post(f"/train/{job_id}/stop")
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_training_jobs() -> str:
    """List all training jobs (running, completed, and failed)."""
    result = await connection.get("/train")
    return json.dumps(result, indent=2)


# ===========================================================================
# Evaluation
# ===========================================================================

@mcp.tool()
async def evaluate_policy(
    task: str,
    checkpoint: str,
    framework: str = "skrl",
    num_envs: int = 16,
    num_steps: int = 200,
    record_video: bool = True,
) -> str:
    """Evaluate a trained policy checkpoint.

    Args:
        task: Environment ID (e.g. 'Isaac-Ant-v0')
        checkpoint: Path to the checkpoint file on the remote instance
        framework: RL framework used for training (default: 'skrl')
        num_envs: Number of evaluation environments (default: 16)
        num_steps: Number of evaluation steps (default: 200)
        record_video: Record evaluation video (default: True)
    """
    result = await connection.post("/eval", json={
        "task": task,
        "checkpoint": checkpoint,
        "framework": framework,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "record_video": record_video,
    })
    return json.dumps(result, indent=2)


# ===========================================================================
# Files & artifacts
# ===========================================================================

@mcp.tool()
async def list_checkpoints(
    task: str | None = None,
    framework: str | None = None,
) -> str:
    """List available trained model checkpoints.

    Args:
        task: Filter by task name (optional, partial match)
        framework: Filter by framework name (optional)
    """
    params = {}
    if task:
        params["task"] = task
    if framework:
        params["framework"] = framework
    result = await connection.get("/files/checkpoints", params=params)
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_log_dirs() -> str:
    """List training log directories on the remote instance."""
    result = await connection.get("/files/logs")
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_videos() -> str:
    """List recorded simulation/evaluation videos on the remote instance."""
    result = await connection.get("/files/videos")
    return json.dumps(result, indent=2)


@mcp.tool()
async def read_remote_file(path: str, tail: int = 100) -> str:
    """Read a file from the remote Nebius instance.

    Supports text files (returns last N lines) and images (returns base64).

    Args:
        path: Absolute path to the file on the remote instance
        tail: Number of lines from end for text files (default: 100)
    """
    result = await connection.get("/files/read", params={"path": path, "tail": tail})
    return json.dumps(result, indent=2)


# ===========================================================================
# Script execution
# ===========================================================================

@mcp.tool()
async def run_isaaclab_script(
    script_path: str,
    args: list[str] | None = None,
    headless: bool = True,
    timeout: int = 300,
) -> str:
    """Run a Python script through Isaac Lab's launcher on the remote instance.

    Use this for custom scripts, demos, or any Isaac Lab operation not covered
    by other tools.

    Args:
        script_path: Path to the script (absolute, or relative to Isaac Lab install)
        args: Command-line arguments to pass to the script
        headless: Run without GUI (default: True)
        timeout: Maximum execution time in seconds (default: 300)
    """
    result = await connection.post("/run", json={
        "script_path": script_path,
        "args": args or [],
        "headless": headless,
        "timeout": timeout,
    })
    return json.dumps(result, indent=2)


# ===========================================================================
# GPU monitoring
# ===========================================================================

@mcp.tool()
async def gpu_status() -> str:
    """Get GPU utilization, memory usage, and temperature from the remote instance."""
    result = await connection.get("/gpu")
    return json.dumps(result, indent=2)


# ===========================================================================
# Resources — contextual information for the LLM
# ===========================================================================

@mcp.resource("isaaclab://guide")
async def isaaclab_guide() -> str:
    """Quick reference guide for Isaac Lab concepts and common tasks."""
    return textwrap.dedent("""\
        # Isaac Lab Quick Reference

        ## Architecture
        - Isaac Lab is built on NVIDIA Isaac Sim (Omniverse)
        - Uses GPU-parallelized environments via gymnasium API
        - Two workflow styles: Manager-Based (modular, config-driven) and Direct (monolithic, JIT-friendly)

        ## Common Tasks (Gymnasium IDs)
        **Classic:** Isaac-Cartpole-v0, Isaac-Ant-v0, Isaac-Humanoid-v0
        **Manipulation:** Isaac-Reach-Franka-v0, Isaac-Lift-Cube-Franka-v0, Isaac-Open-Drawer-Franka-v0
        **Locomotion:** Isaac-Velocity-Flat-Anymal-D-v0, Isaac-Velocity-Rough-Anymal-D-v0, Isaac-Velocity-Flat-Unitree-Go2-v0
        **Navigation:** Isaac-Navigation-Flat-Anymal-C-v0

        ## RL Frameworks
        - **skrl**: Modern, modular, good default choice
        - **rsl_rl**: Optimized for locomotion, ETH RSL's framework
        - **sb3**: Stable Baselines 3, great for prototyping
        - **rl_games**: NVIDIA's framework, supports multi-GPU well

        ## Key Concepts
        - **num_envs**: Number of parallel environments (GPU-vectorized). More = faster training. 4096 is common.
        - **Observations**: State info the agent sees (joint positions, velocities, base pose, etc.)
        - **Actions**: What the agent outputs (joint positions/velocities/torques, IK targets)
        - **Domain Randomization**: EventManager randomizes mass, friction, gravity, etc.
        - **Curriculum**: CurriculumManager adjusts difficulty during training

        ## Training Tips
        - Start with smaller num_envs (512-1024) for debugging
        - Use skrl or rsl_rl for locomotion tasks
        - Use sb3 for quick prototyping
        - Monitor GPU memory with gpu_status — OOM is common with large num_envs
        - Checkpoints are saved periodically in logs/{framework}/{task}/{timestamp}/checkpoints/

        ## Remote Execution
        - All computation runs headless on the Nebius GPU instance
        - Livestreaming available but not needed for training
        - Videos are recorded during evaluation with --video flag
    """)


@mcp.resource("isaaclab://environments")
async def environments_resource() -> str:
    """Dynamic list of available environments from the connected instance."""
    if not connection.connected:
        return "Not connected to a Nebius instance. Use connect_instance first."
    try:
        result = await connection.get("/envs")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching environments: {e}"


# ===========================================================================
# Prompts — common workflows
# ===========================================================================

@mcp.prompt()
async def train_locomotion(
    robot: str = "Anymal-D",
    terrain: str = "flat",
) -> str:
    """Set up a locomotion training run for a quadruped robot."""
    task_map = {
        ("Anymal-D", "flat"): "Isaac-Velocity-Flat-Anymal-D-v0",
        ("Anymal-D", "rough"): "Isaac-Velocity-Rough-Anymal-D-v0",
        ("Anymal-C", "flat"): "Isaac-Velocity-Flat-Anymal-C-v0",
        ("Anymal-C", "rough"): "Isaac-Velocity-Rough-Anymal-C-v0",
        ("Unitree-Go2", "flat"): "Isaac-Velocity-Flat-Unitree-Go2-v0",
        ("Unitree-Go2", "rough"): "Isaac-Velocity-Rough-Unitree-Go2-v0",
        ("Unitree-H1", "flat"): "Isaac-Velocity-Flat-Unitree-H1-v0",
        ("Unitree-H1", "rough"): "Isaac-Velocity-Rough-Unitree-H1-v0",
    }
    task = task_map.get((robot, terrain), f"Isaac-Velocity-{terrain.title()}-{robot}-v0")
    return textwrap.dedent(f"""\
        Train a {robot} locomotion policy on {terrain} terrain.

        Recommended setup:
        1. Connect to Nebius instance
        2. Start training:
           - Task: {task}
           - Framework: rsl_rl (best for locomotion)
           - num_envs: 4096
           - max_iterations: 1500
        3. Monitor every ~60s for progress
        4. Evaluate the best checkpoint with video recording
    """)


@mcp.prompt()
async def train_manipulation(
    task_type: str = "reach",
    robot: str = "Franka",
) -> str:
    """Set up a manipulation training run."""
    task_map = {
        "reach": f"Isaac-Reach-{robot}-v0",
        "lift": f"Isaac-Lift-Cube-{robot}-v0",
        "open_drawer": f"Isaac-Open-Drawer-{robot}-v0",
    }
    task = task_map.get(task_type, f"Isaac-{task_type.title()}-{robot}-v0")
    return textwrap.dedent(f"""\
        Train a {robot} arm for {task_type} task.

        Recommended setup:
        1. Connect to Nebius instance
        2. Start training:
           - Task: {task}
           - Framework: skrl
           - num_envs: 2048
           - max_iterations: 1000
        3. Monitor training progress
        4. Evaluate with video to verify behavior
    """)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
