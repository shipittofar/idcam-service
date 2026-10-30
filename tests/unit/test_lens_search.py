"""Lens search: coarse sweep, fine refinement, honest failure."""

from __future__ import annotations

import pytest

from idcam.camera.lens_search import find_sharpest_lens
from idcam.exceptions import AutofocusError


class FakeLensCamera:
    """probe_lens returns the 'frame' (here: the sharpness itself) of a sharp peak at ``peak``."""

    def __init__(self, peak=9.0, width=0.35, follow=True, flat=False):
        self.peak, self.width, self.follow, self.flat = peak, width, follow, flat
        self.calls: list[float] = []
        self.settles: list[float] = []

    def probe_lens(self, position, settle_sec=0.7):
        self.calls.append(round(position, 3))
        self.settles.append(settle_sec)
        value = 1580.0 if self.flat else 15.0 + 2000.0 * max(0.0, 1.0 - abs(position - self.peak) / self.width)
        meta = {"LensPosition": position if self.follow else 11.18, "ExposureTime": 30000, "AnalogueGain": 2.0}
        return value, meta


def _fn(frame, roi=None):
    return frame


def search(camera, **kw):
    return find_sharpest_lens(camera, (0, 0, 10, 10), 0.0, 15.0, sharpness_fn=_fn, **kw)


def test_finds_a_narrow_peak_to_a_tenth_of_a_dioptre():
    cam = FakeLensCamera(peak=9.1)
    result = search(cam)
    assert result.lens == pytest.approx(9.1, abs=0.1)
    assert result.peak_ratio > 5 and result.meta_followed


def test_the_whole_range_is_scanned_then_refined_around_the_best_point():
    cam = FakeLensCamera(peak=6.0)
    search(cam)
    coarse = [c for c in cam.calls if abs(c * 2 - round(c * 2)) < 1e-6]
    assert min(coarse) == 1.0 and max(coarse) == 15.0  # coarse 0.5 steps over the range (from 1.0)
    fine = [c for c in cam.calls if abs(c * 2 - round(c * 2)) > 1e-6]
    assert fine and all(5.4 <= c <= 6.6 for c in fine)


def test_the_lens_gets_time_to_arrive_before_each_measurement():
    cam = FakeLensCamera()
    search(cam, settle_sec=0.9)
    assert set(cam.settles) == {0.9}


def test_a_scene_where_sharpness_does_not_depend_on_the_lens_is_reported_not_guessed():
    with pytest.raises(AutofocusError, match="no clear sharpness peak"):
        search(FakeLensCamera(flat=True))


def test_metadata_that_never_follows_the_lens_is_flagged_but_does_not_stop_the_search():
    result = search(FakeLensCamera(peak=9.0, follow=False))
    assert result.meta_followed is False
    assert result.lens == pytest.approx(9.0, abs=0.1)


def test_positions_are_clamped_to_the_advertised_range():
    cam = FakeLensCamera(peak=14.8)
    result = find_sharpest_lens(cam, None, 0.0, 15.0, sharpness_fn=_fn)
    assert max(cam.calls) <= 15.0 and result.lens <= 15.0


def test_gives_up_when_it_takes_too_long():
    cam = FakeLensCamera()
    with pytest.raises(AutofocusError, match="too long"):
        search(cam, max_seconds=-1.0)


def test_sharpness_does_not_depend_on_the_auto_exposure_level():
    import numpy as np

    from idcam.camera.lens_search import exposure_normalised_sharpness

    rng = np.random.default_rng(1)
    pattern = (rng.random((120, 160, 3)) > 0.5).astype(np.float32)
    dim = (pattern * 60 + 20).astype(np.uint8)
    bright = (pattern * 120 + 40).astype(np.uint8)  # the same scene, twice the exposure
    a = exposure_normalised_sharpness(dim, roi=(10, 10, 100, 80))
    b = exposure_normalised_sharpness(bright, roi=(10, 10, 100, 80))
    assert b == pytest.approx(a, rel=0.05)


def test_a_blurred_frame_scores_lower_even_when_it_is_brighter():
    import cv2
    import numpy as np

    from idcam.camera.lens_search import exposure_normalised_sharpness

    rng = np.random.default_rng(2)
    sharp = ((rng.random((120, 160, 3)) > 0.5) * 100 + 30).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 0)
    blurred = np.clip(blurred.astype(np.int16) + 60, 0, 255).astype(np.uint8)  # longer exposure
    assert exposure_normalised_sharpness(sharp, roi=(10, 10, 100, 80)) > exposure_normalised_sharpness(blurred, roi=(10, 10, 100, 80))
