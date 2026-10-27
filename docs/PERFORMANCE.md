# Performance: detection-to-validation latency

Goal: shorten the time from a document being placed to a validated capture **without** lowering
sharpness, OCR readability, capture reliability or detection accuracy. Quality gates are unchanged
(final image validation, the 64 MP still, the sharpness thresholds).

## Baseline (from Pi logs, before this work)

| Stage | Time | Source |
|---|---:|---|
| Detection (3 frames) | 0.6–1.0 s | estimated from frame pacing |
| Stability (6 frames) | 1.7 s | measured |
| Autofocus incl. lens refinement | 6.5 s (up to 12 s with the wide sweep) | measured |
| Pre-capture checks | 0.3 s | measured |
| 64 MP capture + camera reconfiguration | 2.4 s | measured (not broken down) |
| Validation | 0.6 s | measured |
| **Detection to validation** | **≈ 11.6 s** | measured |

Upload preparation and upload were not reliably measured.

## Where the time went (root causes in the code)

* **Autofocus 6.5 s.** Hardware AF (0.5–1 s, repeated when its result was not "accepted") followed
  by a lens search with a fixed 0.3 s settle per sample, two frames per sample, 8–9 samples and a
  confirming re-measurement. The search is what makes the image sharp (preview sharpness ≈ 1200
  after hardware AF, ≈ 2600 after the search) and is kept.
* **Detection/stability.** `preview_fps: 5` plus ≈ 0.08 s of work per tick gives ≈ 3.5 ticks/s;
  3 + 6 required frames is ≈ 2.6 s.
* **Capture 2.4 s.** Reconfigure to 9152×6944, wait for the lens, expose/read the frame, encode a
  190 MB image to JPEG, reconfigure back. Not measured per operation until this change.
* **Between documents.** ≥ 2 s SUCCESS hold plus 10 empty frames (≈ 2.8 s) before the next document.

## What changed

| Change | Effect | Quality safeguard |
|---|---|---|
| **Lens memory** (`af_lens_memory`) | The **last successful focus** (remembered as soon as a focus completes) is the starting point of the next document: a local search from that lens position, no hardware AF cycle and no wide sweep. Typically 3-4 samples instead of 8-9 plus AF | Small step (`af_local_step`, 0.25 dioptre) around the remembered position; the centre must be a bracketed peak *and* reach at least `af_local_min_ratio` (70 %) of the live preview's sharpness, otherwise the memory is dropped and the full procedure (hardware AF, refinement, wide sweep) runs. Cleared when the image taken with it is rejected, on focus failure or motion abort, and after `af_memory_max_age_sec` (6 h) |
| **Metadata settle** (`af_settle_by_metadata`) | Waits until the frame's `LensPosition` is within `af_settle_tolerance` of the command, at least `af_refine_min_settle_sec`, at most `af_refine_settle_sec` (the old fixed delay) | Frames are always *fresh* (exposed after the wait). Never waits longer than before; if the camera reports no metadata the old fixed wait applies. **Needs on-device validation** (see below) |
| **Adaptive refinement** (`af_adaptive`) | Skips the finest probe when the peak is already bracketed after the first fine probes, and skips the confirming re-measurement for a clean unimodal peak | Both skips require: a single peak, both neighbours ≥ 4 % lower, no motion hints. Noisy, bimodal or edge-of-range curves keep every stage and the confirmation |
| **One hardware AF pass** | When the lens search follows, a second hardware pass is not run just because the first result was not "accepted" | The search starts from the AF result either way; failure paths unchanged |
| **Motion detection** | Two fresh frames at the same lens position that differ by more than `af_motion_spread_min` mean motion; `af_motion_max_unstable` such samples abort focusing as `FocusMotionError` | The orchestrator returns to *waiting for stability* (up to `MAX_MOTION_RESTABILIZE` times) instead of continuing a 10 s sweep on a moving page |
| **Stability sensitivity** (`detection.stability_sensitivity`, default 0.6) | The motion thresholds of the stability wait and of the pre-capture check are divided by it: 0.6 tolerates ≈ 67 % more movement (a 40 % cut in sensitivity); 1.0 restores the old behaviour | A page still moving by more than the relaxed threshold (13.3 instead of 8 with the default `motion_threshold`) still resets stability; the still is validated independently |
| **Timing instrumentation** | `timing_summary`, `focus_timing`, `focus_sample`, `still_timing`, `loop_stats`, detection latency, removal wait | Log-only |

Not changed on purpose: capture resolution (64 MP), JPEG settings, sharpness thresholds, LED
brightness (0.2), detection defaults. See "Experimental profile".

## Autofocus decision logic

1. Prepare: fresh frame, set the AF window on the document (≈ 0.3 s).
2. If a successful focus is remembered (and fresh): local search from it (step 0.25, ≤ 5 samples,
   no fine probes). Centre bracketed by lower neighbours and ≥ 70 % of the live preview sharpness ⇒
   done (`path=memory`). Otherwise forget the memory and continue.
3. Hardware AF once → measure.
4. Lens search from that result: coarse climb → first fine probes → second fine probes **only if**
   the peak is not yet bracketed → if the curve has no prominent peak, the wide sweep (unchanged).
5. Confirmation: skipped (clean peak and the freshest reading is at the winner), *settled* (clean
   peak elsewhere: lens moved to it and allowed to settle with the metadata wait, no collapse
   check) or *measured* at the full settle time with the "reading collapsed ⇒ unstable" check
   (all other cases). The lens is always at rest before the pre-capture motion baseline is taken.
6. The lens is remembered at every successful focus; a rejected image, a focus failure or motion
   abort forgets it.

## Instrumentation (all monotonic clock)

* `event=timing_summary` — one per document: `detect_ms lock_ms stability_ms focus_ms capture_ms
  validation_ms upload_ms response_ms total_ms end_to_end_ms detect_to_validated_ms` (+ `outcome`).
  `upload_ms` includes preparing the smaller upload copy (`upload_image_prepared` has its own time).
* `event=focus_sample` — each lens sample: position, sharpness, spread of the two frames, lens
  metadata and whether it settled, exposure (`exposure_us`), `gain`.
* `event=focus_timing` — `path=memory|hardware`, `prep_ms hw_af_ms refine_ms total_ms samples`.
* `event=still_timing` — `reconfigure_ms frame_wait_ms encode_ms restore_preview_ms total_ms`.
* `event=still_capture_metadata` — exposure, gain, lens position and frame age of the saved still.
* `event=loop_stats` (once a minute) — achieved loop rate vs `preview_fps`, average/max tick time.
* `document_detected ... latency_ms=` and `document_removed ... waited_ms=`.
* Missing metadata is logged as `None` (never as 0).

Report: `python scripts/timing_report.py run.log` (median / p95 / max per stage), or
`journalctl -u idcam-local -o cat --no-pager | python scripts/timing_report.py -`, or
`--compare after.log`.

## Experimental profile (not the default — measure before adopting)

Candidate settings, same key hierarchy as `config.yaml`:

```yaml
camera:
  preview_fps: 10
  af_refine_settle_sec: 0.2
  af_refine_max_points: 5
detection:
  detect_frames_required: 2
  stable_frames_required: 4
  removal_frames_required: 6
led:
  brightness: 0.2        # unchanged: raise only after checking exposure_us / glare
```

Do not assume `preview_fps: 10` doubles throughput: compare `loop_stats achieved_fps` with the target.
Shorter stability/removal windows can capture a still-moving page or end the removal wait early:
check `moved_before_capture` / `moved_during_focus` counts and duplicate captures before adopting.
Change one group at a time (e.g. first only the detection group) so effects can be attributed.

## On-device validation procedure (required before trusting any number)

1. Baseline: check out `1eef471` (last commit before this work), run ≥ 20 documents of the same
   kind under normal light, save the log. Then the current version, same documents, same light.
2. `python scripts/timing_report.py before.log --compare after.log`. Look at **median and p95** of
   `detect_to_validated_ms`, the first-attempt success rate (`timing_summary outcome=success` vs
   `failed:*`) and `document_detected latency_ms`.
3. Metadata settle: in `focus_sample` lines `lens_ok=True` with small `waited_ms` means the lens
   metadata converged quickly. If `lens_ok=False` dominates, the metadata does not track the lens
   in this stack — set `af_settle_by_metadata: false` (falls back to the fixed wait).
4. Quality: keep `IDCAM_KEEP_REJECTED=1`, compare `image_validated sharpness_document` and the
   sharpness of saved originals before/after; OCR a sample of both sets. Only claim "unchanged
   quality" with those numbers.
5. Scenarios: still document; different heights; low light; glossy paper; page moved during
   detection and during focus; consecutive documents; camera init failure; upload failure.
6. Exposure: read `exposure_us` / `gain` in `focus_sample` and `still_capture_metadata` before
   touching LED brightness; check for glare and clipped highlights on real documents.

## Rollback

* Behaviour switches (no code change): `stability_sensitivity: 1.0` (old motion tolerance),
  `af_adaptive: false`, `af_lens_memory: false`,
  `af_settle_by_metadata: false`, `af_motion_max_unstable: 20` restore the previous focusing
  behaviour (all stages, hardware AF retries, fixed settle, no motion abort).
* Full: `git revert` the commit, or `git checkout 1eef471` in the project folder and restart
  `idcam-local`.

## Status

Verified here: unit tests with simulated cameras (settling lag, wrong/missing metadata, memory
hit/miss, clean/ambiguous curves, motion). **Not verified without the Pi:** real lens settling
time vs `LensPosition` metadata, the achieved speed-up, and image-quality equivalence.
