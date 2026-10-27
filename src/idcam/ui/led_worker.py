"""Owns the WS2812B strip in a process of its own (``python -m idcam.ui.led_worker``).

The capture service talks to it over a pipe: one ``R G B`` line per colour change. Keeping the
strip in a child process means:

* a failed import of Blinka (it checks platform dependencies on import and can fail at boot) is
  retried with a fresh interpreter, which an in-process retry cannot do;
* Blinka's interactive "install missing dependencies? [Y/n]" prompt can never block or consume
  the capture service's stdin (here stdin is /dev/null);
* a crash in the PIO driver cannot take the capture pipeline down.

No root is needed: ``/dev/pio0`` belongs to group ``gpio``, which the service user is in.
"""

from __future__ import annotations

import argparse
import os
import sys

EXIT_INIT_FAILED = 3


def _take_command_pipe():
    """Move the command pipe off fd 0 and point stdin at /dev/null (see module docstring)."""
    fd = os.dup(0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    return os.fdopen(fd, "r", encoding="ascii", errors="ignore", buffering=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="idCam WS2812B status strip worker")
    parser.add_argument("--pin", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--brightness", type=float, required=True)
    parser.add_argument("--byteorder", default="GRB")
    args = parser.parse_args(argv)

    commands = _take_command_pipe()
    try:
        from idcam.config.schema import LedConfig
        from idcam.ui.led import Pi5NeopixelBackend

        backend = Pi5NeopixelBackend(
            LedConfig(
                enabled=True,
                gpio_pin=args.pin,
                count=args.count,
                brightness=args.brightness,
                byteorder=args.byteorder,
            )
        )
    except (Exception, SystemExit) as exc:  # Blinka may sys.exit() from its dependency check
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return EXIT_INIT_FAILED

    try:
        for line in commands:
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                color = tuple(max(0, min(255, int(p))) for p in parts)
            except ValueError:
                continue
            backend.show(color)  # type: ignore[arg-type]
    finally:
        try:
            backend.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
