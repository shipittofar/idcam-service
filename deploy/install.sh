#!/usr/bin/env bash
# Backward-compatible wrapper. Prefer: sudo ./scripts/install.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../scripts/install.sh" "$@"
