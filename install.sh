#!/usr/bin/env bash
# Jarvis4 install script — sets up config, service, and dependencies.
#
# Usage:
#   ./install.sh              # Full install (config + service + deps)
#   ./install.sh --service    # Just install/update the systemd service
#   ./install.sh --deps       # Just install Python dependencies
#   ./install.sh --memory     # Add optional semantic memory deps (large)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${HOME}/.config/jarvis4"
SERVICE_NAME="jarvis4"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*"; }

# --- Config ---
install_config() {
    info "Setting up config directory: ${CONFIG_DIR}"
    mkdir -p "${CONFIG_DIR}"

    if [ ! -f "${SCRIPT_DIR}/config.json" ]; then
        warn "No config.json found — copying example"
        cp "${SCRIPT_DIR}/config.example.json" "${SCRIPT_DIR}/config.json"
        echo ""
        warn "Edit config.json with your API keys before starting:"
        warn "  - llm.api_key (Cerebras or OpenAI-compatible)"
        warn "  - integrations.homeassistant.token (if using HA)"
        warn "  - integrations.slack.bot_token (if using Slack)"
        echo ""
    fi

    # Create MCP servers config if not present
    if [ ! -f "${CONFIG_DIR}/mcp_servers.json" ]; then
        info "Creating empty MCP servers config"
        echo '{"mcpServers": {}}' > "${CONFIG_DIR}/mcp_servers.json"
    else
        info "MCP config already exists"
    fi
}

# --- Dependencies ---
install_deps() {
    info "Installing Python dependencies"
    pip install -r "${SCRIPT_DIR}/requirements.txt"
}

install_memory_deps() {
    info "Installing optional semantic memory dependencies (large: pulls in torch)"
    pip install -r "${SCRIPT_DIR}/requirements-memory.txt"
}

# --- Systemd service ---
install_service() {
    info "Installing systemd service: ${SERVICE_NAME}"

    # Detect Python path
    PYTHON_PATH="$(which python3)"
    PYTHON_DIR="$(dirname "${PYTHON_PATH}")"

    # Generate service file from template
    cat > /tmp/jarvis4.service << SVCEOF
[Unit]
Description=Jarvis4 Voice Assistant
After=network.target sound.target
Wants=network.target

[Service]
Type=simple
User=${USER}
Group=${USER}
WorkingDirectory=${SCRIPT_DIR}
Environment="PATH=${PYTHON_DIR}:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=${PYTHON_PATH} ${SCRIPT_DIR}/jarvis4.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Give access to audio and USB devices
SupplementaryGroups=audio plugdev

[Install]
WantedBy=multi-user.target
SVCEOF

    sudo cp /tmp/jarvis4.service "${SERVICE_FILE}"
    rm /tmp/jarvis4.service
    sudo systemctl daemon-reload
    info "Service installed at ${SERVICE_FILE}"

    # Enable but don't start
    sudo systemctl enable "${SERVICE_NAME}" 2>/dev/null || true
    info "Service enabled (use 'sudo systemctl start ${SERVICE_NAME}' to start)"
}

# --- Main ---
case "${1:-all}" in
    --service) install_service ;;
    --deps)    install_deps ;;
    --memory)  install_memory_deps ;;
    --config)  install_config ;;
    all)
        install_config
        install_deps
        install_service
        echo ""
        info "Jarvis4 installed!"
        info "  Config: ${SCRIPT_DIR}/config.json"
        info "  MCP:    ${CONFIG_DIR}/mcp_servers.json"
        info "  Service: sudo systemctl start ${SERVICE_NAME}"
        info "  Logs:    journalctl -u ${SERVICE_NAME} -f"
        ;;
    *)
        echo "Usage: $0 [--service|--deps|--memory|--config|all]"
        exit 1
        ;;
esac
