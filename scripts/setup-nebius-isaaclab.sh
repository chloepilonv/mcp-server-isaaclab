#!/usr/bin/env bash
# One-time setup: install Isaac Lab on a fresh Nebius GPU instance.
#
# Prerequisites: Nebius instance with NVIDIA GPU, Ubuntu 22.04, CUDA 12.x
#
# Usage:
#   ssh user@nebius-host 'bash -s' < scripts/setup-nebius-isaaclab.sh
#   OR
#   ./scripts/setup-nebius-isaaclab.sh  (run directly on the instance)

set -euo pipefail

echo "==> Isaac Lab Setup for Nebius GPU Instance"
echo "    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU not detected')"

ISAAC_LAB_DIR="$HOME/IsaacLab"
ISAAC_SIM_VERSION="4.5.0"  # Update as needed

# 1. System dependencies
echo "==> Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git git-lfs cmake build-essential

# 2. Clone Isaac Lab
if [ ! -d "$ISAAC_LAB_DIR" ]; then
    echo "==> Cloning Isaac Lab..."
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAAC_LAB_DIR"
else
    echo "==> Isaac Lab already exists at $ISAAC_LAB_DIR, pulling latest..."
    cd "$ISAAC_LAB_DIR" && git pull
fi

cd "$ISAAC_LAB_DIR"

# 3. Install Isaac Lab (pip installation — uses bundled Isaac Sim)
echo "==> Installing Isaac Lab (this may take a while)..."
if [ -f "isaaclab.sh" ]; then
    # Install Isaac Sim pip packages
    ./isaaclab.sh -i none  # Base install without RL libraries

    # Install RL frameworks
    echo "==> Installing RL frameworks..."
    ./isaaclab.sh -i skrl
    ./isaaclab.sh -i rsl_rl
    ./isaaclab.sh -i sb3

    echo "==> Verifying installation..."
    ./isaaclab.sh -p -c "import isaaclab; print(f'Isaac Lab {isaaclab.__version__} installed successfully')" || true
else
    echo "ERROR: isaaclab.sh not found. Check that IsaacLab cloned correctly."
    exit 1
fi

echo ""
echo "==> Isaac Lab setup complete!"
echo "    Install path: $ISAAC_LAB_DIR"
echo "    Next: deploy the remote agent with ./scripts/deploy-remote-agent.sh"
