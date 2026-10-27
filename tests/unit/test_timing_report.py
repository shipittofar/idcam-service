"""scripts/timing_report.py: before/after numbers straight from the application's own timing logs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "timing_report.py"


@pytest.fixture(scope="module")
def report_mod():
    spec = importlib.util.spec_from_file_location("timing_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_line(message: str) -> str:
    return json.dumps({"timestamp": "2026-10-27T00:00:00+00:00", "level": "INFO", "logger": "idcam.orchestrator", "message": message})


def _summary(outcome="success", focus=6450, total=12000):
    return _json_line(
        f"event=timing_summary outcome={outcome} detect_ms=700 lock_ms=49 stability_ms=1700 focus_ms={focus} "
        f"capture_ms=2400 validation_ms=600 upload_ms=800 total_ms={total} end_to_end_ms={total + 700} "
        f"detect_to_validated_ms=11300"
    )


def test_parse_reads_events_out_of_json_log_lines(report_mod):
    records = report_mod.parse([_summary(), "noise", _json_line("event=other x=1")])
    assert len(records["timing_summary"]) == 1
    assert records["timing_summary"][0]["focus_ms"] == "6450"


def test_percentiles_and_medians(report_mod):
    values = [10.0, 20.0, 30.0, 40.0, 100.0]
    s = report_mod.stats(values)
    assert s["median"] == 30.0 and s["max"] == 100.0
    assert report_mod.percentile(values, 95) == pytest.approx(88.0)
    assert report_mod.stats([]) is None


def test_report_counts_only_successful_documents_by_default(report_mod):
    lines = [_summary(focus=5000), _summary(focus=7000), _summary(outcome="failed:image_invalid", focus=30000)]
    text, data = report_mod.report(report_mod.parse(lines), include_failed=False)
    assert data["summary"]["focus_ms"]["n"] == 2
    assert data["summary"]["focus_ms"]["median"] == 6000
    assert "failed:image_invalid x1" in text

    _text, all_data = report_mod.report(report_mod.parse(lines), include_failed=True)
    assert all_data["summary"]["focus_ms"]["n"] == 3
    assert all_data["summary"]["focus_ms"]["max"] == 30000


def test_focus_still_and_loop_sections(report_mod):
    lines = [
        _json_line("event=focus_timing path=memory prep_ms=300 hw_af_ms=skipped refine_ms=2100 total_ms=2400 samples=5 settle=metadata confirm=parked_clean unstable_samples=0"),
        _json_line("event=focus_timing path=hardware prep_ms=300 hw_af_ms=900 refine_ms=4400 total_ms=5600 samples=8 settle=metadata confirm=measured unstable_samples=0"),
        _json_line("event=still_timing reconfigure_ms=900 frame_wait_ms=500 encode_ms=700 fallback_ms=None restore_preview_ms=800 total_ms=2900"),
        _json_line("event=loop_stats window_s=60 ticks=420 achieved_fps=7.00 target_fps=10.0 avg_tick_ms=80 max_tick_ms=310 state=IDLE"),
    ]
    text, data = report_mod.report(report_mod.parse(lines), include_failed=False)
    assert data["focus_timing"]["refine_ms"]["n"] == 2
    assert data["focus_timing"]["hw_af_ms"]["n"] == 1  # "skipped" is not a number
    assert "memory x1" in text and "hardware x1" in text
    assert data["still_timing"]["encode_ms"]["median"] == 700
    assert data["loop_stats"]["achieved_fps"]["median"] == 7.0


def test_compare_shows_the_change_per_stage(report_mod):
    before = report_mod.report(report_mod.parse([_summary(focus=6500, total=12000)] * 3), False)[1]
    after = report_mod.report(report_mod.parse([_summary(focus=4000, total=9000)] * 3), False)[1]
    text = report_mod.compare(before, after)
    assert "summary.focus_ms" in text and "(-2500)" in text


def test_empty_log_says_so_instead_of_inventing_numbers(report_mod, tmp_path, capsys):
    log = tmp_path / "empty.log"
    log.write_text("nothing relevant\n", encoding="utf-8")
    assert report_mod.main([str(log)]) == 0
    assert "No timing events found" in capsys.readouterr().out


def test_cli_end_to_end(report_mod, tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text("\n".join([_summary(), _summary(focus=5000)]), encoding="utf-8")
    assert report_mod.main([str(log)]) == 0
    out = capsys.readouterr().out
    assert "2 successful" in out and "focus_ms" in out
