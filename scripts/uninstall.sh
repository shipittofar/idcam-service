#!/usr/bin/env bash
# Remove the idCam service and application bits.
# Never deletes /etc/idcam, /var/lib/idcam, or /var/log/idcam unless --purge
# is given AND the operator types DELETE.

set -euo pipefail

SERVICE_USER="idcam"
SERVICE_GROUP="idcam"
PURGE=0
STAGING_ROOT=""
YES_TOKEN=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
idCam uninstaller

Usage: sudo ./scripts/uninstall.sh [options]

Options:
  --purge           Also delete config, data, logs, and the idcam user
                    Requires typing DELETE when prompted (or --confirm DELETE)
  --confirm WORD    Non-interactive purge confirmation (must be DELETE)
  --staging DIR     Operate on a staging prefix created by install.sh --staging
  -h, --help        Show this help

Without --purge this removes:
  /opt/idcam
  /etc/systemd/system/idcam.service
  /etc/systemd/system/idcam.service.d

It keeps:
  /etc/idcam
  /var/lib/idcam
  /var/log/idcam
  the idcam user and group
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --purge) PURGE=1; shift ;;
        --confirm)
            YES_TOKEN="${2:-}"
            shift 2
            ;;
        --staging)
            STAGING_ROOT="${2:-}"
            [[ -n "$STAGING_ROOT" ]] || { echo "ERROR: --staging requires a directory" >&2; exit 2; }
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$STAGING_ROOT" ]]; then
    STAGING_ROOT="$(cd "$STAGING_ROOT" && pwd)"
fi

prefix_path() {
    local p="$1"
    if [[ -n "$STAGING_ROOT" ]]; then
        echo "${STAGING_ROOT}${p}"
    else
        echo "$p"
    fi
}

INSTALL_DIR="$(prefix_path /opt/idcam)"
CONFIG_DIR="$(prefix_path /etc/idcam)"
STATE_DIR="$(prefix_path /var/lib/idcam)"
LOG_DIR="$(prefix_path /var/log/idcam)"
UNIT_FILE="$(prefix_path /etc/systemd/system/idcam.service)"
DROPIN_DIR="$(prefix_path /etc/systemd/system/idcam.service.d)"

die() { echo "ERROR: $*" >&2; exit 1; }

if [[ -z "$STAGING_ROOT" && "${EUID}" -ne 0 ]]; then
    die "Run as root: sudo ./scripts/uninstall.sh"
fi

if [[ "$PURGE" -eq 1 ]]; then
    if [[ "$YES_TOKEN" != "DELETE" ]]; then
        echo "This will permanently delete:"
        echo "  ${CONFIG_DIR}"
        echo "  ${STATE_DIR}"
        echo "  ${LOG_DIR}"
        echo "  user/group ${SERVICE_USER}"
        printf "Type DELETE to confirm: "
        read -r YES_TOKEN
    fi
    if [[ "$YES_TOKEN" != "DELETE" ]]; then
        die "Purge aborted (confirmation was not DELETE)"
    fi
fi

if [[ -z "$STAGING_ROOT" ]] && command -v systemctl >/dev/null; then
    systemctl stop idcam.service 2>/dev/null || true
    systemctl disable idcam.service 2>/dev/null || true
fi

rm -rf "$INSTALL_DIR"
rm -f "$UNIT_FILE"
rm -rf "$DROPIN_DIR"

if [[ -z "$STAGING_ROOT" ]] && command -v systemctl >/dev/null; then
    systemctl daemon-reload || true
    systemctl reset-failed idcam.service 2>/dev/null || true
fi

echo "Removed application and systemd unit."
echo "Preserved (unless --purge): ${CONFIG_DIR} ${STATE_DIR} ${LOG_DIR}"

if [[ "$PURGE" -eq 1 ]]; then
    rm -rf "$CONFIG_DIR" "$STATE_DIR" "$LOG_DIR"
    if [[ -z "$STAGING_ROOT" ]]; then
        if id "$SERVICE_USER" >/dev/null 2>&1; then
            userdel "$SERVICE_USER" || true
        fi
        if getent group "$SERVICE_GROUP" >/dev/null; then
            groupdel "$SERVICE_GROUP" || true
        fi
    fi
    echo "Purge complete."
fi
