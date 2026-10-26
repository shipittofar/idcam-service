"""Application entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from idcam import __version__
from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.config.loader import load_config
from idcam.exceptions import ConfigurationError
from idcam.logging.setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="idCam document capture")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--mock-camera", action="store_true", help="Use mock camera (dev/test)")
    parser.add_argument("--version", action="version", version=f"idcam {__version__}")
    args = parser.parse_args()

    try:
        config_path = None
        if args.config:
            from pathlib import Path

            config_path = Path(args.config)
        config = load_config(config_path)
        setup_logging(config.logging)
        use_mock = args.mock_camera or os.environ.get("IDCAM_MOCK_CAMERA", "").lower() in ("1", "true")
        orchestrator = ApplicationOrchestrator(config, use_mock_camera=use_mock)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)

    exit_code = asyncio.run(orchestrator.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
