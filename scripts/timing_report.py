#!/usr/bin/env python3
"""Summarise idCam timing logs: count, median, 95th percentile and max per stage.

Reads the ``event=timing_summary`` (one per document), ``event=focus_timing``,
``event=still_timing`` and ``event=loop_stats`` lines the application writes.

    python scripts/timing_report.py ~/idcam-run.log
    journalctl -u idcam-local -o cat --no-pager | python scripts/timing_report.py -
    python scripts/timing_report.py before.log --compare after.log
    python scripts/timing_report.py run.log --all        # include failed attempts

By default only successful documents are counted in the end-to-end table, so slow failures do
not hide behind a good median (use --all to see them too). Every number comes from the
application's own monotonic-clock measurements; nothing here is estimated.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from typing import Iterable

EVENTS = ("timing_summary", "focus_timing", "still_timing", "loop_stats")
_EVENT_RE = re.compile(r"event=(" + "|".join(EVENTS) + r")\s+([^\"\n]*)")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")

SUMMARY_FIELDS = (
    "detect_ms",
    "lock_ms",
    "stability_ms",
    "focus_ms",
    "capture_ms",
    "validation_ms",
    "upload_ms",
    "response_ms",
    "detect_to_validated_ms",
    "end_to_end_ms",
)
FOCUS_FIELDS = ("prep_ms", "hw_af_ms", "refine_ms", "total_ms", "samples")
STILL_FIELDS = ("reconfigure_ms", "frame_wait_ms", "encode_ms", "restore_preview_ms", "total_ms")
LOOP_FIELDS = ("achieved_fps", "avg_tick_ms", "max_tick_ms")


def parse(lines: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in lines:
        match = _EVENT_RE.search(line)
        if not match:
            continue
        record = {}
        for token in match.group(2).split():
            key, sep, value = token.partition("=")
            if sep:
                record[key] = value
        records[match.group(1)].append(record)
    return records


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (0-100)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def column(records: list[dict[str, str]], field: str) -> list[float]:
    return [float(r[field]) for r in records if field in r and _NUMBER.match(r[field])]


def stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def table(title: str, records: list[dict[str, str]], fields: tuple[str, ...]) -> tuple[str, dict[str, dict[str, float]]]:
    rows: dict[str, dict[str, float]] = {}
    out = [f"{title} ({len(records)} records)", f"  {'stage':<26}{'n':>5}{'median':>10}{'p95':>10}{'max':>10}"]
    for field in fields:
        s = stats(column(records, field))
        if s:
            rows[field] = s
            out.append(f"  {field:<26}{s['n']:>5.0f}{s['median']:>10.0f}{s['p95']:>10.0f}{s['max']:>10.0f}")
    return "\n".join(out), rows


def report(records: dict[str, list[dict[str, str]]], include_failed: bool) -> tuple[str, dict[str, dict[str, dict[str, float]]]]:
    sections: list[str] = []
    data: dict[str, dict[str, dict[str, float]]] = {}
    summaries = records.get("timing_summary", [])
    ok = [r for r in summaries if r.get("outcome") == "success"]
    chosen = summaries if include_failed else ok
    if summaries:
        failed = len(summaries) - len(ok)
        title = f"Documents: {len(ok)} successful, {failed} not completed (all values in ms)"
        text, data["summary"] = table(title, chosen, SUMMARY_FIELDS)
        sections.append(text)
        if failed:
            reasons: dict[str, int] = defaultdict(int)
            for r in summaries:
                if r.get("outcome") != "success":
                    reasons[r.get("outcome", "?")] += 1
            sections.append("  not completed: " + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())))
    for name, fields, title in (
        ("focus_timing", FOCUS_FIELDS, "Focus"),
        ("still_timing", STILL_FIELDS, "Still capture"),
        ("loop_stats", LOOP_FIELDS, "Main loop (per minute)"),
    ):
        recs = records.get(name, [])
        if recs:
            text, data[name] = table(title, recs, fields)
            sections.append(text)
            if name == "focus_timing":
                paths: dict[str, int] = defaultdict(int)
                for r in recs:
                    paths[r.get("path", "?")] += 1
                sections.append("  focus path: " + ", ".join(f"{k} x{v}" for k, v in sorted(paths.items())))
    if not sections:
        return "No timing events found. Run the current version and process a few documents first.", data
    return "\n\n".join(sections), data


def compare(before: dict[str, dict[str, dict[str, float]]], after: dict[str, dict[str, dict[str, float]]]) -> str:
    out = ["Before -> after (median / p95, ms; negative = faster)"]
    for group in ("summary", "focus_timing", "still_timing"):
        for field, b in before.get(group, {}).items():
            a = after.get(group, {}).get(field)
            if not a:
                continue
            out.append(
                f"  {group + '.' + field:<34}{b['median']:>8.0f} -> {a['median']:<8.0f}({a['median'] - b['median']:+.0f})"
                f"   p95 {b['p95']:>7.0f} -> {a['p95']:<7.0f}({a['p95'] - b['p95']:+.0f})"
            )
    return "\n".join(out)


def _read(path: str) -> list[str]:
    if path == "-":
        return sys.stdin.read().splitlines()
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="idCam timing report")
    parser.add_argument("log", help="log file, or - for stdin")
    parser.add_argument("--compare", metavar="AFTER_LOG", help="second log to compare against the first")
    parser.add_argument("--all", action="store_true", help="include failed attempts in the end-to-end table")
    args = parser.parse_args(argv)

    first_text, first = report(parse(_read(args.log)), args.all)
    print(first_text)
    if args.compare:
        second_text, second = report(parse(_read(args.compare)), args.all)
        print("\n--- " + args.compare + " ---\n" + second_text)
        print("\n" + compare(first, second))
    return 0


if __name__ == "__main__":
    sys.exit(main())
