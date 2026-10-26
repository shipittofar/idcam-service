"""Long-run test mode for resource leak detection."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tracemalloc

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.config.loader import load_config
from idcam.logging.setup import setup_logging


async def run_long_test(iterations: int) -> int:
    config = load_config()
    setup_logging(config.logging)
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    orchestrator = ApplicationOrchestrator(config, use_mock_camera=True)

    # Mock rapid idle cycles without server
    with orchestrator._shutdown_event:
        pass

    for i in range(iterations):
        orchestrator.detector.reset_baseline()
        orchestrator._clear_transaction()
        orchestrator.sm.state = orchestrator.sm.state  # no-op keep alive
        if i % 10 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"iteration={i} memory_current={current / 1024:.1f}KB peak={peak / 1024:.1f}KB")

    snapshot_end = tracemalloc.take_snapshot()
    stats = snapshot_end.compare_to(snapshot_start, "lineno")
    print("Top memory growth:")
    for stat in stats[:5]:
        print(stat)
    tracemalloc.stop()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="idCam long-run resource test")
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("IDC_LONG_RUN_ITERATIONS", "50")))
    args = parser.parse_args()
    if os.environ.get("IDC_LONG_RUN_TEST", "").lower() not in ("1", "true"):
        print("Set IDC_LONG_RUN_TEST=1 to run", file=sys.stderr)
        sys.exit(1)
    sys.exit(asyncio.run(run_long_test(args.iterations)))


if __name__ == "__main__":
    main()
