#!/usr/bin/env bash
# One-command local setup: run idCam from THIS folder instead of the systemd service.
#   bash scripts/setup_local.sh        # install deps, venv, config, then ./run.sh to start
# Safe to re-run. Does not touch /opt/idcam or the idcam service (it only stops it
# when you pass --stop-service).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STOP_SERVICE=0
for a in "$@"; do
    case "$a" in
        --stop-service) STOP_SERVICE=1 ;;
        -h|--help) sed -n 2,6p "$0"; exit 0 ;;
        *) echo "Unknown option: $a" >&2; exit 2 ;;
    esac
done

[[ $EUID -ne 0 ]] || { echo "Run as your normal user (not root); sudo is used only where needed." >&2; exit 1; }
command -v apt-get >/dev/null || { echo "This script targets Raspberry Pi OS / Debian (apt)." >&2; exit 1; }

echo "==> 1/5 System packages (sudo)"
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv python3-dev \
    python3-libcamera python3-picamera2 v4l-utils
# Optional extras: install when this OS has them, skip silently otherwise.
for pkg in python3-kms++ rpicam-apps libcamera-apps; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
        sudo apt-get install -y "$pkg" || echo "   (skipped optional $pkg)"
    fi
done

echo "==> 2/5 Camera access"
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx video; then
    sudo usermod -aG video "$USER"
    echo "   Added $USER to 'video'. Log out and back in if the camera is not accessible."
fi

echo "==> 3/5 Python venv (.venv, with system site-packages for Picamera2)"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

echo "==> 4/5 Config (./config.yaml, ./data, ./.env)"
mkdir -p data/tmp data/logs
if [[ ! -f config.yaml ]]; then
    if sudo test -f /etc/idcam/config.yaml; then
        sudo cat /etc/idcam/config.yaml > config.yaml
        echo "   Copied settings from /etc/idcam/config.yaml"
    else
        cp config.example.yaml config.yaml
        echo "   Created config.yaml from config.example.yaml - edit server.url and device_id"
    fi
    chmod 600 config.yaml
fi
# Point the service-owned directories at this folder so no root-owned path is needed.
sed -i -E "s|^(temp_dir:).*|\1 ${ROOT}/data/tmp|; s|^([[:space:]]*log_dir:).*|\1 ${ROOT}/data/logs|" config.yaml
if [[ ! -f .env ]]; then
    if sudo test -f /etc/idcam/environment; then
        sudo cat /etc/idcam/environment > .env
        echo "   Copied token/env from /etc/idcam/environment"
    else
        printf '# IDCAM_AUTH_TOKEN=your-token-here\n' > .env
        echo "   Created .env - put IDCAM_AUTH_TOKEN there"
    fi
    chmod 600 .env
fi

echo "==> 5/5 Launcher (./run.sh)"
cat > run.sh <<RUN
#!/usr/bin/env bash
cd "\$(dirname "\$0")"
set -a; [ -f .env ] && . ./.env; set +a
export IDCAM_CONFIG="\$PWD/config.yaml"
exec .venv/bin/idcam "\$@"
RUN
chmod +x run.sh

if [[ $STOP_SERVICE -eq 1 ]]; then
    sudo systemctl stop idcam 2>/dev/null || true
    sudo systemctl disable idcam 2>/dev/null || true
    echo "   idcam service stopped and disabled."
elif systemctl is-active --quiet idcam 2>/dev/null; then
    echo "   WARNING: the idcam service is running and holds the camera."
    echo "            Run: sudo systemctl stop idcam   (or re-run with --stop-service)"
fi

echo
echo "Done. Start with:  ./run.sh"
echo "Mock camera test:  IDCAM_MOCK_CAMERA=1 ./run.sh"
