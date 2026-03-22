#!/usr/bin/env bash
# Deploy the Isaac Lab remote agent to a Nebius GPU instance.
#
# Usage:
#   ./scripts/deploy-remote-agent.sh <HOST> [USER] [KEY_PATH]
#
# Example:
#   ./scripts/deploy-remote-agent.sh 203.0.113.42 ubuntu ~/.ssh/nebius_key

set -euo pipefail

HOST="${1:?Usage: $0 <HOST> [USER] [KEY_PATH]}"
USER="${2:-ubuntu}"
KEY="${3:-}"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
if [ -n "$KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $KEY"
fi

REMOTE_DIR="/home/$USER/mcp-server-isaaclab"

echo "==> Deploying remote agent to $USER@$HOST"

# 1. Copy the package
echo "==> Syncing files..."
rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
    -e "ssh $SSH_OPTS" \
    "$(dirname "$0")/../" \
    "$USER@$HOST:$REMOTE_DIR/"

# 2. Install dependencies on the remote
echo "==> Installing remote dependencies..."
ssh $SSH_OPTS "$USER@$HOST" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail
cd ~/mcp-server-isaaclab

# Create venv if it doesn't exist (use system Python to keep Isaac Lab accessible)
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
fi

source .venv/bin/activate
pip install -e ".[remote]" --quiet

echo "==> Remote agent installed successfully"
REMOTE_SCRIPT

# 3. Set up systemd service for auto-start
echo "==> Setting up systemd service..."
ssh $SSH_OPTS "$USER@$HOST" bash -s <<REMOTE_SERVICE
set -euo pipefail
sudo tee /etc/systemd/system/isaaclab-remote-agent.service > /dev/null <<EOF
[Unit]
Description=Isaac Lab Remote Agent for MCP
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/mcp-server-isaaclab
ExecStart=/home/$USER/mcp-server-isaaclab/.venv/bin/isaaclab-remote-agent
Restart=on-failure
RestartSec=5
Environment=ISAACLAB_PATH=/home/$USER/IsaacLab
Environment=ISAACLAB_AGENT_PORT=8421

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable isaaclab-remote-agent
sudo systemctl restart isaaclab-remote-agent

echo "==> Service started. Checking status..."
sleep 2
sudo systemctl status isaaclab-remote-agent --no-pager || true
REMOTE_SERVICE

echo ""
echo "==> Deployment complete!"
echo "    Remote agent running on $HOST:8421"
echo ""
echo "    To use with Claude, add to claude_desktop_config.json:"
echo "    {"
echo "      \"mcpServers\": {"
echo "        \"isaaclab\": {"
echo "          \"command\": \"mcp-server-isaaclab\""
echo "        }"
echo "      }"
echo "    }"
echo ""
echo "    Then in Claude: connect_instance(host=\"$HOST\")"
