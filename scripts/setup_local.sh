#!/usr/bin/env bash
# One-command local setup: run idCam from THIS folder instead of the systemd service.
#   bash scripts/setup_local.sh        # install deps, venv, config, then ./run.sh to start
# Safe to re-run. Does not touch /opt/idcam or the idcam service (it only stops it
# when you pass --stop-service).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STOP_SERVICE=0
WITH_LED=1
for a in "$@"; do
    case "$a" in
        --stop-service) STOP_SERVICE=1 ;;
        --no-led) WITH_LED=0 ;;  # skip the WS2812B status strip (GPIO18 by default; see led.gpio_pin)
        -h|--help) sed -n 2,6p "$0"; exit 0 ;;
        *) echo "Unknown option: $a" >&2; exit 2 ;;
    esac
done

[[ $EUID -ne 0 ]] || { echo "Run as your normal user (not root); sudo is used only where needed." >&2; exit 1; }
command -v apt-get >/dev/null || { echo "This script targets Raspberry Pi OS / Debian (apt)." >&2; exit 1; }

echo "==> 0/5 Camera stack guard"
# Arducam 64MP on Pi 5 needs Arducam's libcamera build (it ships arducam_64mp.json).
# The stock apt libcamera has no such file, so an upgrade silently kills the camera.
# If the Arducam tuning file is present, pin the libcamera core packages.
LIBCAMERA_PKGS=()
for pkg in libcamera0.7 libcamera-ipa libcamera-dev; do
    dpkg -s "$pkg" >/dev/null 2>&1 && LIBCAMERA_PKGS+=("$pkg")
done
if compgen -G "/usr/share/libcamera/ipa/rpi/*/arducam_64mp.json" >/dev/null; then
    if ((${#LIBCAMERA_PKGS[@]})); then
        sudo apt-mark hold "${LIBCAMERA_PKGS[@]}" >/dev/null
        echo "   Arducam libcamera detected; held: ${LIBCAMERA_PKGS[*]}"
    fi
else
    echo "   WARNING: arducam_64mp.json not found - the Arducam 64MP will not register."
    ARDUCAM_DEBS=(~/libcamera0.7_*-2_arm64.deb ~/libcamera-ipa_*-2_arm64.deb ~/libcamera-dev_*-2_arm64.deb)
    if [[ -e "${ARDUCAM_DEBS[0]}" && -e "${ARDUCAM_DEBS[1]}" && -e "${ARDUCAM_DEBS[2]}" ]]; then
        echo "   Arducam .deb files found in ~. Review and install them, then re-run this script:"
        echo "     sudo apt-get install --allow-downgrades ${ARDUCAM_DEBS[*]}"
    else
        echo "   Install Arducam's libcamera packages (install_pivariety_pkgs.sh), then re-run."
    fi
    echo "   (continuing: mock mode still works; real camera will fail until this is fixed)"
fi

echo "==> 1/5 System packages (sudo)"
# --no-upgrade: install only what is missing. Never upgrade libcamera/rpicam that
# already work with the camera (an upgrade can drop the arducam_64mp tuning file).
sudo apt-get update -y
sudo apt-get install -y --no-upgrade python3 python3-pip python3-venv python3-dev \
    python3-libcamera python3-picamera2 v4l-utils
# Optional extras: install when this OS has them, skip silently otherwise.
for pkg in python3-kms++ rpicam-apps libcamera-apps; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
        sudo apt-get install -y --no-upgrade "$pkg" || echo "   (skipped optional $pkg)"
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
if [[ $WITH_LED -eq 1 ]]; then
    # Status LED strip (WS2812B) libraries; the strip is optional and never blocks capture.
    .venv/bin/pip install -e ".[led]" || echo "   (LED libraries not installed; the strip stays off)"
    # /dev/pio0 (RP1 PIO, used to drive the strip) is group 'gpio' on Raspberry Pi OS.
    if getent group gpio >/dev/null && ! id -nG "$USER" | tr ' ' '\n' | grep -qx gpio; then
        sudo usermod -aG gpio "$USER"
        echo "   Added $USER to 'gpio'. Log out and back in if the LED strip stays dark."
    fi
fi

echo "==> 4/5 Config (./config.yaml, ./data, ./.env)"
mkdir -p data/tmp data/logs data/run
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
# Point the service-owned directories (tmp, logs, /run status + admin socket) at this folder so no root-owned path is needed.
sed -i -E "s|^(temp_dir:).*|\1 ${ROOT}/data/tmp|; s|^([[:space:]]*log_dir:).*|\1 ${ROOT}/data/logs|; s|^([[:space:]]*status_file:).*|\1 ${ROOT}/data/run/status.json|; s|^([[:space:]]*admin_socket:).*|\1 ${ROOT}/data/run/admin.sock|" config.yaml
# Turn the status strip on (pin, count and colours stay editable under 'led:' in config.yaml).
if [[ $WITH_LED -eq 1 ]]; then
    if grep -q '^led:' config.yaml; then
        awk 'BEGIN{inled=0}
             /^led:/{inled=1; print; next}
             inled && /^[^ \t#]/{inled=0}
             inled && /^[ \t]+enabled:/{sub(/enabled:.*/, "enabled: true")}
             {print}' config.yaml > config.yaml.tmp && mv config.yaml.tmp config.yaml && chmod 600 config.yaml
    else
        printf '\nled:\n  enabled: true\n  gpio_pin: 18   # BCM number of the data pin (GPIO18 = physical pin 12)\n  count: 30\n' >> config.yaml
    fi
    echo "   LED strip enabled on GPIO $(awk '/^led:/{f=1;next} f&&/gpio_pin:/{print $2; exit}' config.yaml)"
fi
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
