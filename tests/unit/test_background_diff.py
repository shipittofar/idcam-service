"""What is NEW against the learned empty desk: dim corners, static clutter, held pages, refresh rules."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision import document_geometry as dg
from idcam.vision.document_detector import DocumentDetector

from test_hand_held import GREEN, H, NAVY, W, _card, _desk

_YY, _XX = np.mgrid[0:H, 0:W]
_R = np.hypot((_XX - W / 2) / (W / 2), (_YY - H / 2) / (H / 2))


def _dim(img, vignette=0.8, blur=14):
    """Vignetted corners and a lens that gets softer towards the edge."""
    out = np.clip(img.astype(np.float32) * (1 - vignette * np.clip(_R, 0, 1.4) ** 2 / 1.96)[..., None], 0, 255).astype(np.uint8)
    levels = [0, 2, 4, 7, 10, 14]
    blurred = [out if s == 0 else cv2.GaussianBlur(out, (0, 0), s) for s in levels]
    bins = np.digitize(np.clip(_R, 0, 1.4) / 1.4 * blur, [1, 3, 5.5, 8.5, 11.5])
    result = out.copy()
    for i in range(len(levels)):
        result = np.where((bins == i)[..., None], blurred[i], result)
    return result


def _detector(**kw):
    return DocumentDetector(DetectionConfig(background_check=True, background_warmup_frames=4, **kw))


def _learn(det, frame, n=10):
    for _ in range(n):
        det.detect(frame.copy())
    assert det._bg_work is not None


class TestDimCorners:
    @pytest.mark.parametrize("cx,cy", [(330, 230), (1590, 850), (1590, 230), (330, 850), (200, 540)])
    def test_a_page_in_a_vignetted_blurred_corner_is_detected_quickly(self, cx, cy):
        empty = _dim(_desk("dark"))
        det = _detector()
        _learn(det, empty)
        img, truth = _card("dark", NAVY, None, cx=cx, cy=cy, width=360, angle=4.0)
        results = [det.detect(_dim(img).copy()) for _ in range(6)]
        assert results.index(True) <= 3, results  # not the 9th frame, when the gate used to look again
        geom = det._last_geometry
        assert geom is not None and dg._bbox_iou(truth, geom.bbox) >= 0.85

    def test_the_gate_notices_a_dim_page_that_changes_a_few_thumbnail_pixels_by_a_few_levels(self):
        det = _detector()
        base = _dim(_desk("dark"))
        det._find_geometry(base)
        dim_page = base.copy()
        dim_page[820:1000, 1450:1760] += 9  # a faint patch in the dark corner
        assert det._scene_unchanged(dim_page) is False

    def test_sensor_noise_and_a_global_exposure_shift_do_not_wake_the_gate(self):
        rng = np.random.default_rng(1)
        det = _detector()
        base = _desk("dark")
        det._find_geometry(base)
        noisy = np.clip(base.astype(np.int16) + 6 + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
        assert det._scene_unchanged(noisy) is True


class TestStaticClutterCannotHideAPage:
    def test_a_high_scoring_static_object_does_not_outrank_the_real_page(self):
        desk = _desk("white")
        cv2.rectangle(desk, (150, 150), (850, 700), (60, 60, 60), -1)  # a large dark mat, always there
        cv2.rectangle(desk, (150, 150), (850, 700), (0, 0, 0), 8)
        det = _detector()
        _learn(det, desk)
        scene = desk.copy()
        img, truth = _card("white", GREEN, None, cx=1450, cy=600, width=420, angle=2.0)
        cv2.fillPoly(scene, [np.array([[1240, 480], [1660, 480], [1660, 720], [1240, 720]])], GREEN)
        results = [det.detect(scene.copy()) for _ in range(8)]
        assert results[-1] is True
        x, y, w, h = det._last_geometry.bbox
        assert x > 1000, det._last_geometry.bbox  # the card on the right, not the mat

    def test_the_mat_alone_is_never_a_document(self):
        desk = _desk("white")
        cv2.rectangle(desk, (150, 150), (850, 700), (60, 60, 60), -1)
        cv2.rectangle(desk, (150, 150), (850, 700), (0, 0, 0), 8)
        det = _detector()
        _learn(det, desk)
        assert not any(det.detect(desk.copy()) for _ in range(12))


class TestDifferenceMask:
    def test_nothing_new_gives_no_mask_and_a_page_gives_one(self):
        det = _detector()
        empty = _desk("dark")
        _learn(det, empty)
        assert det._difference_mask(empty.copy()) is None
        img, _ = _card("dark", NAVY, None)
        mask = det._difference_mask(img)
        assert mask is not None and 0.02 < np.count_nonzero(mask) / mask.size < 0.4

    def test_it_is_off_without_the_background_check(self):
        det = DocumentDetector(DetectionConfig(background_check=False))
        assert det._difference_mask(_desk("dark")) is None and det._extra_masks(_desk("dark")) is None

    def test_a_global_exposure_shift_is_not_new(self):
        det = _detector()
        empty = _desk("dark")
        _learn(det, empty)
        brighter = np.clip(empty.astype(np.int16) + 25, 0, 255).astype(np.uint8)
        assert det._difference_mask(brighter) is None


class TestBackgroundRefresh:
    def test_a_page_that_lies_there_undetected_is_never_learned_as_background(self):
        det = _detector(background_refresh_frames=10)
        empty = _desk("dark")
        _learn(det, empty)
        before = det._bg_work.copy()
        img, _ = _card("dark", NAVY, None)
        thumb = det._bg_thumbnail(img)
        det._calm_streak = 10  # a refresh moment, with the page lying on the desk
        det._update_background(thumb, img)
        assert np.array_equal(det._bg_work, before)

    def test_the_same_desk_with_slightly_different_light_is_refreshed(self):
        det = _detector(background_refresh_frames=10)
        empty = _desk("dark")
        _learn(det, empty)
        slightly = np.clip(empty.astype(np.int16) + 6, 0, 255).astype(np.uint8)
        det._calm_streak = 10
        det._update_background(det._bg_thumbnail(slightly), slightly)
        assert not np.array_equal(det._bg_work, det._work_image(empty))

    def test_a_long_lasting_different_picture_is_finally_taken_for_the_new_normal(self):
        det = _detector(background_refresh_frames=100)
        empty = _desk("dark")
        _learn(det, empty)
        other = _desk("white")  # the light was switched on
        for i in range(1, 8):
            det._calm_streak = 100 * i
            det._update_background(det._bg_thumbnail(other), other)
        assert np.allclose(det._bg_work, det._work_image(other))


class TestSmallDocumentIsNotTheEmptyDesk:
    def test_a_small_card_left_on_a_large_desk_keeps_the_scene_non_empty(self):
        det = _detector()
        empty = _desk("dark")
        _learn(det, empty)
        img, _ = _card("dark", NAVY, None, width=300)  # ~1.5 % of the frame
        assert det.is_scene_empty(img) is False
        assert det.is_scene_empty(empty.copy()) is True
