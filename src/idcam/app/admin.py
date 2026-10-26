"""Admin diagnostics CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from idcam import __version__
from idcam.config.loader import load_config
from idcam.storage.disk_monitor import DiskMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="idCam admin diagnostics")
    parser.add_argument("--config", type=str, default=None)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Show version")
    sub.add_parser("status", help="Show current status file")
    sub.add_parser("config", help="Show loaded configuration (secrets redacted)")
    sub.add_parser("validate", help="Validate configuration file")
    sub.add_parser("health", help="Show system health snapshot")
    sub.add_parser("storage", help="Show storage status")

    args = parser.parse_args()

    if args.command == "version" or args.command is None:
        print(f"idcam {__version__}")
        return

    config_path = Path(args.config) if args.config else None

    if args.command == "validate":
        try:
            load_config(config_path)
            print("Configuration valid")
        except Exception as exc:
            print(f"Configuration invalid: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    config = load_config(config_path)

    if args.command == "status":
        status_file = config.ui.status_file
        if not status_file.exists():
            print(f"Status file not found: {status_file}", file=sys.stderr)
            sys.exit(1)
        print(status_file.read_text(encoding="utf-8"))
        return

    if args.command == "config":
        data = config.model_dump(mode="json")
        if data.get("auth", {}).get("token"):
            data["auth"]["token"] = "***REDACTED***"
        print(json.dumps(data, indent=2))
        return

    if args.command == "storage":
        monitor = DiskMonitor(config.storage, config.temp_dir)
        print(json.dumps(monitor.check().to_dict(), indent=2))
        return

    if args.command == "health":
        status_file = config.ui.status_file
        health = {
            "version": __version__,
            "config_valid": True,
            "status_file": str(status_file),
            "status_exists": status_file.exists(),
            "storage": DiskMonitor(config.storage, config.temp_dir).check().to_dict(),
            "timestamp": time.time(),
        }
        if status_file.exists():
            health["status"] = json.loads(status_file.read_text(encoding="utf-8"))
        print(json.dumps(health, indent=2))


if __name__ == "__main__":
    main()
