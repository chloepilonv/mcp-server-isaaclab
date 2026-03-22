# mcp-server-isaacsim

MCP server for **NVIDIA Isaac Lab** — control robotics simulations from Claude.

Runs locally on your Mac and communicates with Isaac Lab on a remote **Nebius GPU instance** through an SSH tunnel. All heavy simulation stays on the GPU; Claude just sends commands.

## Architecture

```
┌──────────┐    stdio    ┌──────────────┐   SSH tunnel   ┌─────────────────┐
│  Claude   │◄──────────►│  MCP Server  │◄──────────────►│  Remote Agent   │
│  (local)  │            │  (local Mac) │   port 8421    │  (Nebius GPU)   │
└──────────┘            └──────────────┘                └────────┬────────┘
                                                                  │
                                                          ┌───────▼────────┐
                                                          │   Isaac Lab    │
                                                          │  (Isaac Sim)   │
                                                          └────────────────┘
```

**MCP Server** (this repo) runs on your Mac as a stdio MCP server. It opens an SSH tunnel to the Nebius instance and forwards all requests to the **Remote Agent** — a FastAPI service running next to Isaac Lab on the GPU box.

## Prerequisites

- Python 3.10+
- A Nebius GPU instance (or any Linux box with an NVIDIA GPU)
- SSH access to the instance
- Isaac Lab installed on the instance (setup script included)

## Quick Start

### 1. Install locally

```bash
git clone git@github.com:chloepilonv/mcp-server-isaacsim.git
cd mcp-server-isaacsim
pip install -e .
```

### 2. Install Isaac Lab on your Nebius instance

```bash
scp scripts/setup-nebius-isaaclab.sh ubuntu@<NEBIUS_IP>:~
ssh ubuntu@<NEBIUS_IP> bash ~/setup-nebius-isaaclab.sh
```

This installs Isaac Lab + the skrl, rsl_rl, and sb3 RL frameworks.

### 3. Deploy the remote agent

```bash
./scripts/deploy-remote-agent.sh <NEBIUS_IP> ubuntu ~/.ssh/your_key
```

This copies the agent code, installs it, and starts it as a systemd service on port 8421.

### 4. Configure Claude

The project includes `.mcp.json` so Claude Code automatically picks up the server when you're in this directory.

For **Claude Desktop**, add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "isaaclab": {
      "command": "mcp-server-isaacsim"
    }
  }
}
```

For **Claude Code** in other projects, add to the project's `.mcp.json`:

```json
{
  "mcpServers": {
    "isaaclab": {
      "command": "mcp-server-isaacsim"
    }
  }
}
```

## Tools

### Connection

| Tool | Description |
|------|-------------|
| `connect_instance` | Establish SSH tunnel to Nebius GPU instance |
| `disconnect_instance` | Tear down the connection |
| `instance_status` | GPU utilization, active sessions & jobs |
| `gpu_status` | Detailed GPU memory, temperature, utilization |

### Simulation (Interactive)

| Tool | Description |
|------|-------------|
| `list_environments` | List all registered Isaac Lab tasks |
| `create_session` | Create an interactive simulation session |
| `step_session` | Step simulation forward (random or specified actions) |
| `reset_session` | Reset environment to initial state |
| `get_observation` | Get current observations + action/obs space info |
| `close_session` | Close session and free GPU memory |

### Training

| Tool | Description |
|------|-------------|
| `start_training` | Launch an async RL training job |
| `monitor_training` | Get status, recent logs, latest checkpoint |
| `get_training_logs` | Read full training logs |
| `stop_training` | Stop a running training job |
| `list_training_jobs` | List all jobs (running, completed, failed) |

### Evaluation & Files

| Tool | Description |
|------|-------------|
| `evaluate_policy` | Evaluate a checkpoint, optionally record video |
| `list_checkpoints` | Browse saved model checkpoints |
| `list_log_dirs` | Browse training log directories |
| `list_videos` | List recorded simulation videos |
| `read_remote_file` | Read any text/image file on the instance |
| `run_isaaclab_script` | Run arbitrary Isaac Lab Python scripts |

## Example Conversations

**Train a locomotion policy:**
```
> Connect to my Nebius instance at 203.0.113.42
> What environments are available for quadruped locomotion?
> Train Anymal-D on rough terrain with rsl_rl, 4096 envs, 1500 iterations
> Check on the training
> Evaluate the best checkpoint and record a video
```

**Explore an environment interactively:**
```
> Connect to 203.0.113.42
> Create a session with Isaac-Cartpole-v0, 32 envs
> What does the observation space look like?
> Step 100 times with random actions — what are the rewards?
> Reset and try again
> Close the session
```

**Monitor GPU and manage jobs:**
```
> What's the GPU status?
> List all training jobs
> Stop the Ant training — it's not converging
> Show me the last 200 lines of logs from the Franka training
```

## Supported RL Frameworks

| Framework | Best For | Notes |
|-----------|----------|-------|
| **skrl** | General purpose | Modern, modular, good default choice |
| **rsl_rl** | Locomotion | ETH RSL's framework, optimized for legged robots |
| **sb3** | Prototyping | Stable Baselines 3, easy to use |
| **rl_games** | Multi-GPU | NVIDIA's framework, scales well |

## Available Environments (selection)

| Category | Examples |
|----------|---------|
| Classic | `Isaac-Cartpole-v0`, `Isaac-Ant-v0`, `Isaac-Humanoid-v0` |
| Manipulation | `Isaac-Reach-Franka-v0`, `Isaac-Lift-Cube-Franka-v0`, `Isaac-Open-Drawer-Franka-v0` |
| Locomotion | `Isaac-Velocity-Flat-Anymal-D-v0`, `Isaac-Velocity-Rough-Unitree-Go2-v0` |
| Navigation | `Isaac-Navigation-Flat-Anymal-C-v0` |

Use `list_environments` to get the full list from your installation.

## Project Structure

```
mcp-server-isaacsim/
├── src/mcp_server_isaacsim/
│   ├── server.py            # MCP server (runs locally, exposes tools)
│   ├── connection.py        # SSH tunnel + HTTP client manager
│   └── remote/
│       └── agent.py         # FastAPI agent (runs on Nebius GPU)
├── scripts/
│   ├── deploy-remote-agent.sh      # Deploy agent to Nebius
│   └── setup-nebius-isaaclab.sh    # Install Isaac Lab on instance
├── .mcp.json                # Claude Code MCP config
├── pyproject.toml
└── README.md
```

## Development

```bash
pip install -e ".[dev]"
ruff check src/
pytest
```

## License

MIT
