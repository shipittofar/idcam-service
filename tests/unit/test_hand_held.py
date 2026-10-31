"""A page held in a hand: the box must be the page, the confidence must pass, the stability wait must not stall."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision import document_geometry as dg
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.stability_detector import StabilityDetector

W, H = 1920, 1080
SKIN = (224, 172, 140)
NAVY = (30, 60, 140)
BURGUNDY = (120, 30, 50)
GREEN = (40, 120, 70)
WHITE = (235, 235, 232)


def _desk(kind: str) -> np.ndarray:
    rng = np.random.default_rng(1)
    base = {"dark": (45, 48, 50), "white": (215, 215, 210)}[kind]
    return np.clip(np.full((H, W, 3), base, np.float32) + rng.normal(0, 3, (H, W, 3)), 0, 255).astype(np.uint8)


def _card(kind, color, hand=None, cx=960, cy=480, width=420, angle=3.0):
    img = _desk(kind)
    height = width * 54 / 85.6
    r = np.deg2rad(angle)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]])
    quad = corners @ rot.T + [cx, cy]
    cv2.fillPoly(img, [quad.astype(np.int32)], color)
    x0, y0 = quad.min(0).astype(int)
    x1, y1 = quad.max(0).astype(int)
    w, h = x1 - x0, y1 - y0
    cv2.rectangle(img, (x0 + int(w * 0.08), y0 + int(h * 0.25)), (x0 + int(w * 0.3), y0 + int(h * 0.7)), tuple(int(v * 0.6) for v in color), -1)
    for i in range(4):
        y = y0 + int(h * (0.3 + i * 0.12))
        cv2.line(img, (x0 + int(w * 0.4), y), (x0 + int(w * 0.9), y), tuple(int(v * 0.5) for v in color), 2)
    if hand == "below":
        cv2.ellipse(img, (int((x0 + x1) / 2), int(y1 + h * 0.55)), (int(w * 0.55), int(h * 0.5)), 0, 0, 360, SKIN, -1)
    elif hand == "thumb_corner":
        cv2.ellipse(img, (int(x0 + w * 0.08), int(y1 - h * 0.10)), (int(w * 0.16), int(h * 0.20)), -30, 0, 360, SKIN, -1)
        cv2.ellipse(img, (int(x0 - w * 0.25), int(y1 + h * 0.35)), (int(w * 0.45), int(h * 0.45)), -20, 0, 360, SKIN, -1)
    return img, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _detect(img, **kw):
    cfg = DetectionConfig()
    return dg.detect_document_geometry(
        img,
        min_area_fraction=cfg.geometry_min_area_fraction,
        max_area_fraction=cfg.geometry_max_area_fraction,
        **kw,
    )


class TestHeldPage:
    @pytest.mark.parametrize("desk", ["dark", "white"])
    @pytest.mark.parametrize("color", [NAVY, BURGUNDY, GREEN])
    @pytest.mark.parametrize("hand", ["below", "thumb_corner"])
    def test_the_box_is_the_page_not_the_page_plus_the_hand(self, desk, color, hand):
        img, truth = _card(desk, color, hand)
        geom = _detect(img)
        assert geom is not None
        assert dg._bbox_iou(truth, geom.bbox) >= 0.85, (hand, geom.bbox, truth)
        assert geom.confidence >= DetectionConfig().min_geometry_confidence - DetectionConfig().hand_confidence_relax

    def test_a_white_page_on_a_dark_desk_under_a_thumb_passes_the_normal_threshold(self):
        img, truth = _card("dark", WHITE, "thumb_corner")
        geom = _detect(img)
        assert geom is not None and geom.confidence >= DetectionConfig().min_geometry_confidence
        assert dg._bbox_iou(truth, geom.bbox) >= 0.9

    def test_the_hand_free_variants_only_exist_when_a_hand_is_in_view(self):
        img, _ = _card("dark", NAVY, None)
        _, reports, _ = dg._detect_all(img)
        assert not any(r.source_method.endswith("_noskin") for r in reports)
        img2, _ = _card("dark", NAVY, "below")
        _, reports2, _ = dg._detect_all(img2)
        assert any(r.source_method.endswith("_noskin") for r in reports2)

    def test_a_skin_coloured_page_without_a_hand_is_still_found(self):
        img, truth = _card("dark", (190, 150, 90), None)  # tan: inside the skin colour range
        geom = _detect(img)
        assert geom is not None and geom.confidence >= 0.5
        assert dg._bbox_iou(truth, geom.bbox) >= 0.9

    def test_a_desk_that_is_skin_coloured_switches_the_cue_off(self):
        img = np.full((H, W, 3), SKIN, np.uint8)
        mask, fraction, raw = dg._skin_mask(cv2.resize(img, (640, 360)))
        assert mask is None and fraction > 0.9 and raw > 0.9  # the whole frame is "skin": useless


class TestRanking:
    def test_a_candidate_that_swallowed_a_hand_ranks_lower_but_keeps_its_confidence(self):
        base = dict(
            corners=((0, 0), (10, 0), (10, 10), (0, 10)), bbox=(0, 0, 10, 10), center=(5, 5), width=10, height=10,
            rotation_deg=0.0, aspect_ratio=1.0, area_fraction=0.1, edge_strength=1.0, rectangularity=1.0,
        )
        clean = dg.DocumentGeometry(confidence=0.6, skin_fraction=0.0, **base)
        with_hand = dg.DocumentGeometry(confidence=0.7, skin_fraction=0.3, **base)
        skin_page = dg.DocumentGeometry(confidence=0.7, skin_fraction=0.9, **base)
        assert with_hand.confidence == 0.7
        assert with_hand.rank < clean.rank
        assert skin_page.rank == skin_page.confidence  # a skin-coloured page is not penalised


class TestDetectorThresholds:
    def test_a_hand_free_candidate_needs_a_little_less_confidence(self):
        det = DocumentDetector(DetectionConfig(min_geometry_confidence=0.5, hand_confidence_relax=0.05))

        class G:
            def __init__(self, method):
                self.detection_method = method

        assert det._present_threshold(G("otsu")) == pytest.approx(0.5)
        assert det._present_threshold(G("bg_otsu_noskin")) == pytest.approx(0.45)
        assert det._present_threshold(G("bg_otsu_noskin"), 0.9) == pytest.approx(0.40)


class TestStabilityWithAHand:
    """A held page is judged by how far its corners move; the picture-difference measure cannot tell a
    3 px tremor from a real move (it saturates)."""

    @staticmethod
    def _held(jitter, frames=10):
        rng = np.random.default_rng(3)
        img, (x, y, w, h) = _card("dark", NAVY, "thumb_corner")
        base = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        sd = StabilityDetector(DetectionConfig())
        sd.set_roi((x, y, w, h))
        for _ in range(frames):
            moved = base + (rng.integers(-jitter, jitter + 1, (4, 2)) if jitter else 0)
            sd.update(img, corners=tuple((int(a), int(b)) for a, b in moved))
        return sd

    def test_a_page_held_with_a_small_tremor_becomes_stable(self):
        sd = self._held(jitter=3)
        assert sd.hand_present and sd.stable_count >= DetectionConfig().stable_frames_required

    def test_a_page_that_is_really_moving_does_not(self):
        sd = self._held(jitter=30)
        assert sd.hand_present and sd.stable_count < DetectionConfig().stable_frames_required

    def test_a_still_held_page_is_stable_too(self):
        assert self._held(jitter=0).stable_count >= DetectionConfig().stable_frames_required

    def test_without_a_hand_the_corners_are_ignored_and_the_picture_decides(self):
        rng = np.random.default_rng(3)
        img, box = _card("dark", WHITE, None)
        sd = StabilityDetector(DetectionConfig())
        sd.set_roi(box)
        for _ in range(10):
            dx, dy = rng.integers(-5, 6, 2)
            frame = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
            sd.update(frame, corners=((0, 0), (1, 0), (1, 1), (0, 1)))  # identical corners must not make it "stable"
        assert not sd.hand_present and sd.stable_count < DetectionConfig().stable_frames_required

    def test_a_page_on_the_desk_keeps_the_strict_threshold(self):
        cfg = DetectionConfig()
        assert StabilityDetector(cfg).effective_motion_threshold == pytest.approx(cfg.motion_threshold / cfg.stability_sensitivity)


class TestTopAndBottomEdges:
    @pytest.mark.parametrize("side", ["top", "bottom"])
    def test_a_page_a_few_pixels_from_the_top_or_bottom_passes_with_the_vertical_penalty_eased(self, side):
        w, h = 360, 227
        cy = 2 + h // 2 if side == "top" else H - 2 - h // 2
        img, truth = _card("dark", NAVY, None, cx=960, cy=cy, width=w, angle=0.0)
        strict = _detect(img, touch_penalty_vertical=0.7)
        eased = _detect(img, touch_penalty_vertical=0.9)
        assert eased is not None and eased.confidence >= 0.5
        assert strict is None or strict.confidence < eased.confidence

    def test_a_page_cut_off_by_the_top_is_still_below_the_threshold_by_default(self):
        w = 360
        img, _ = _card("dark", NAVY, None, cx=960, cy=40, width=w, angle=0.0)  # a third of it is outside
        geom = _detect(img, touch_penalty_vertical=DetectionConfig().geometry_touch_penalty_vertical)
        assert geom is None or geom.confidence < DetectionConfig().min_geometry_confidence


class TestSwallowedHandNeedsPatience:
    def test_a_winner_that_includes_the_hand_needs_more_frames_than_a_clean_one(self):
        from types import SimpleNamespace

        det = DocumentDetector(DetectionConfig(detect_frames_required=3, hand_detect_patience=4))

        def feed(swallows):
            geom = SimpleNamespace(
                confidence=0.7, bbox=(100, 100, 400, 300), corners=((0, 0),) * 4, edge_strength=0.0,
                area_fraction=0.1, swallows_hand=swallows, detection_method="otsu",
            )
            det._find_geometry = lambda frame: geom  # type: ignore[method-assign]
            return det.detect(np.zeros((120, 160, 3), np.uint8))

        assert [feed(True) for _ in range(11)][-1] is False  # 3 x 4 = 12 frames needed
        assert feed(True) is True
        det2 = DocumentDetector(DetectionConfig(detect_frames_required=3, hand_detect_patience=4))
        det = det2
        assert [feed(False) for _ in range(3)][-1] is True

    def test_the_reason_is_visible_while_waiting(self):
        from types import SimpleNamespace

        det = DocumentDetector(DetectionConfig(detect_frames_required=3, hand_detect_patience=4))
        geom = SimpleNamespace(
            confidence=0.7, bbox=(100, 100, 400, 300), corners=((0, 0),) * 4, edge_strength=0.0,
            area_fraction=0.1, swallows_hand=True, detection_method="otsu",
        )
        det._find_geometry = lambda frame: geom  # type: ignore[method-assign]
        det.detect(np.zeros((120, 160, 3), np.uint8))
        assert det._last_reject_reason == "waiting_for_hand_free_outline"


class TestHandShape:
    def _g(self, skin, fill, conf=0.7):
        return dg.DocumentGeometry(
            corners=((0, 0), (10, 0), (10, 10), (0, 10)), bbox=(0, 0, 10, 10), center=(5, 5), width=10, height=10,
            rotation_deg=0.0, aspect_ratio=1.0, area_fraction=0.1, confidence=conf, edge_strength=1.0,
            rectangularity=1.0, skin_fraction=skin, quad_fill=fill,
        )

    def test_a_mostly_skin_coloured_box_that_is_not_a_clean_rectangle_is_a_hand(self):
        assert self._g(skin=0.8, fill=0.7).swallows_hand  # a big hand and arm around a small card
        assert self._g(skin=0.3, fill=0.95).swallows_hand
        assert not self._g(skin=0.8, fill=0.95).swallows_hand  # a skin-coloured page
        assert not self._g(skin=0.02, fill=0.7).swallows_hand

    def test_the_hsv_branch_catches_skin_that_the_chroma_range_misses(self):
        warm = np.zeros((20, 20, 3), np.uint8)
        warm[:] = (200, 130, 95)  # a tanned, warm-lit hand: orange hue (not more saturated than skin gets)
        assert np.count_nonzero(dg.skin_pixels(warm)) > 0
        blue = np.zeros((20, 20, 3), np.uint8)
        blue[:] = (30, 60, 140)
        assert np.count_nonzero(dg.skin_pixels(blue)) == 0


class TestSkinColouredDesk:
    """On a tan/wood desk everything is skin-coloured; only pixels that are NEW (vs the learned empty desk) count."""

    DESK = (196, 150, 118)

    def _scene(self, hand):
        rng = np.random.default_rng(1)
        desk = np.clip(np.full((H, W, 3), self.DESK, np.float32) + rng.normal(0, 3, (H, W, 3)), 0, 255).astype(np.uint8)
        img = desk.copy()
        w = 420
        h = int(w * 54 / 85.6)
        x0, y0 = 960 - w // 2, 470 - h // 2
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), NAVY, -1)
        if hand:
            s = (228, 176, 142)
            cv2.ellipse(img, (x0 + int(w * 0.08), y0 + h - int(h * 0.1)), (int(w * 0.16), int(h * 0.2)), -30, 0, 360, s, -1)
            cv2.ellipse(img, (x0 - int(w * 0.2), y0 + h + int(h * 0.4)), (int(w * 0.5), int(h * 0.5)), -20, 0, 360, s, -1)
        return desk, img, (x0, y0, w, h)

    def test_without_the_change_mask_the_cue_is_off_on_such_a_desk(self):
        _, img, _ = self._scene(True)
        geom = _detect(img)
        assert geom is not None and geom.frame_skin == 0.0

    def test_with_the_change_mask_only_the_hand_counts_as_skin(self):
        desk, img, truth = self._scene(True)
        change = (np.abs(img.astype(np.int16) - desk.astype(np.int16)).max(axis=2) > 22).astype(np.uint8) * 255
        geom = _detect(img, change_mask=change)
        assert geom is not None and 0.02 < geom.frame_skin < 0.2
        assert dg._bbox_iou(truth, geom.bbox) >= 0.9

    def test_the_detector_builds_the_change_mask_from_its_learned_background(self):
        desk, img, _ = self._scene(True)
        det = DocumentDetector(DetectionConfig(background_check=True, background_warmup_frames=3))
        assert det._change_mask(desk) is None  # nothing learned yet
        for _ in range(8):
            det.detect(desk.copy())
        mask = det._change_mask(img)
        assert mask is not None and mask.dtype == np.uint8 and np.count_nonzero(mask) > 0
        assert np.count_nonzero(det._change_mask(desk)) == 0  # the empty desk itself has no changed pixels


class TestDiagnosticsAndFill:
    def test_the_best_candidates_are_listed_for_the_log(self):
        img, _ = _card("dark", NAVY, "below")
        geom = _detect(img)
        assert geom is not None and 1 <= len(geom.alternatives) <= 4
        method, bbox, conf, skin, fill = geom.alternatives[0]
        assert isinstance(method, str) and len(bbox) == 4 and 0 <= conf <= 1

    def test_an_outline_with_bites_out_of_it_ranks_below_a_clean_one_of_equal_confidence(self):
        base = dict(
            corners=((0, 0), (10, 0), (10, 10), (0, 10)), bbox=(0, 0, 10, 10), center=(5, 5), width=10, height=10,
            rotation_deg=0.0, aspect_ratio=1.0, area_fraction=0.1, confidence=0.6, edge_strength=1.0, rectangularity=1.0,
        )
        clean = dg.DocumentGeometry(quad_fill=0.98, **base)
        bitten = dg.DocumentGeometry(quad_fill=0.7, **base)
        assert bitten.confidence == clean.confidence and bitten.rank < clean.rank


class TestAsciiSnapshot:
    def test_two_readable_maps_with_the_outline_and_the_warm_colours(self):
        from idcam.vision.document_detector import ascii_snapshot

        img, _ = _card("dark", NAVY, "thumb_corner")
        geom = _detect(img)
        gray, cr = ascii_snapshot(img, geom.corners)
        gray_rows, cr_rows = gray.split("|"), cr.split("|")
        assert len(gray_rows) == len(cr_rows) >= 8 and all(len(r) == 64 for r in gray_rows + cr_rows)
        assert "Q" in gray  # the detected outline
        assert any(ch in "789" for ch in cr)  # the hand is warm: high red chroma
        assert set(cr) <= set("0123456789|")

    def test_a_frame_without_an_outline_still_renders(self):
        from idcam.vision.document_detector import ascii_snapshot

        gray, cr = ascii_snapshot(np.full((120, 160, 3), 128, np.uint8))
        assert "Q" not in gray and len(gray.split("|")) == len(cr.split("|"))


class TestDarkOrShadedHand:
    """In the Pi's ASCII snapshots the hand was Y 25-75 with red chroma 132-156: a luma floor made the cue blind."""

    @staticmethod
    def _scene(card_color, hand_color, kind):
        img, (x0, y0, w, h) = _card("dark", card_color, None, cx=960, cy=380, width=520, angle=2.0)
        x1, y1 = x0 + w, y0 + h
        if kind == "below":
            cv2.ellipse(img, (int((x0 + x1) / 2), int(y1 + h * 0.35)), (int(w * 0.5), int(h * 0.45)), 0, 0, 360, hand_color, -1)
            cv2.rectangle(img, (int((x0 + x1) / 2 - w * 0.3), int(y1 + h * 0.4)), (int((x0 + x1) / 2 + w * 0.3), H), hand_color, -1)
        else:
            cv2.ellipse(img, (int(x0 + w * 0.1), int(y1 - h * 0.08)), (int(w * 0.14), int(h * 0.2)), -30, 0, 360, hand_color, -1)
            cv2.ellipse(img, (int(x0 - w * 0.15), int(y1 + h * 0.4)), (int(w * 0.45), int(h * 0.45)), -20, 0, 360, hand_color, -1)
        return img, (x0, y0, w, h)

    @pytest.mark.parametrize("hand", [(85, 55, 45), (110, 75, 60), (224, 172, 140)])
    @pytest.mark.parametrize("card", [NAVY, (150, 160, 190), BURGUNDY])
    @pytest.mark.parametrize("kind", ["below", "thumb"])
    def test_the_box_is_the_card_whatever_the_tone_of_the_hand(self, hand, card, kind):
        img, truth = self._scene(card, hand, kind)
        geom = _detect(img)
        assert geom is not None
        assert dg._bbox_iou(truth, geom.bbox) >= 0.9, (hand, card, kind, geom.bbox)
        assert geom.frame_skin > 0.02  # the cue saw the hand

    def test_the_skin_classifier_accepts_dark_skin_and_refuses_dark_red_covers(self):
        def is_skin(rgb):
            px = np.zeros((4, 4, 3), np.uint8)
            px[:] = rgb
            return bool(np.count_nonzero(dg.skin_pixels(px)))

        assert is_skin((85, 55, 45)) and is_skin((110, 75, 60)) and is_skin((224, 172, 140))
        assert not is_skin(BURGUNDY) and not is_skin((72, 18, 30)) and not is_skin(NAVY) and not is_skin((130, 130, 130))


class TestContainerBeatsInnerBlock:
    def test_a_photo_block_inside_the_card_does_not_win_over_the_card(self):
        img, truth = _card("dark", BURGUNDY, None, cx=960, cy=380, width=520, angle=2.0)
        geom = _detect(img)
        assert geom is not None and dg._bbox_iou(truth, geom.bbox) >= 0.9

    def test_containment_helper(self):
        assert dg._bbox_contained((120, 120, 50, 50), (100, 100, 200, 200))
        assert not dg._bbox_contained((90, 120, 50, 50), (100, 100, 100, 100))  # sticks out by more than 5 %


class TestDarkHandOnADarkDesk:
    """The 'new pixels only' restriction must not blind the cue when the desk is not skin-coloured."""

    def test_a_hand_only_slightly_brighter_than_the_dark_desk_is_still_seen(self):
        desk = np.full((360, 640, 3), (30, 30, 32), np.uint8)
        frame = desk.copy()
        cv2.ellipse(frame, (320, 250), (110, 70), 0, 0, 360, (62, 40, 33), -1)  # a hand: barely above the desk
        change = (np.abs(frame.astype(np.int16) - desk.astype(np.int16)).max(axis=2) > 40).astype(np.uint8) * 255
        mask, fraction, raw = dg._skin_mask(frame, change)  # a strict change mask would erase the whole hand
        assert mask is not None and 0.05 < fraction < 0.3 and abs(fraction - raw) < 1e-6

    def test_the_restriction_still_applies_when_everything_is_skin_coloured(self):
        desk = np.full((360, 640, 3), (196, 150, 118), np.uint8)
        frame = desk.copy()
        cv2.ellipse(frame, (320, 250), (110, 70), 0, 0, 360, (228, 176, 142), -1)
        change = (np.abs(frame.astype(np.int16) - desk.astype(np.int16)).max(axis=2) > 14).astype(np.uint8) * 255
        mask, fraction, raw = dg._skin_mask(frame, change)
        assert raw > 0.35 and mask is not None and 0.05 < fraction < 0.3  # only the hand is new AND skin-coloured

    def test_the_log_shows_the_raw_fraction_even_when_the_cue_is_off(self):
        desk = np.full((360, 640, 3), (30, 30, 32), np.uint8)
        mask, fraction, raw = dg._skin_mask(desk)
        assert mask is None and raw == 0.0


def test_the_running_build_can_be_logged():
    from idcam.app.orchestrator import _git_commit

    commit = _git_commit()
    assert isinstance(commit, str) and commit


class TestEdgeZone:
    """Pages near (not touching) a border need a little less confidence; messy or cut-off ones do not."""

    @staticmethod
    def _det(**kw):
        det = DocumentDetector(DetectionConfig(min_geometry_confidence=0.5, edge_zone_fraction=0.18, edge_zone_relax=0.08, **kw))
        det._last_frame_size = (1920, 1080)
        return det

    @staticmethod
    def _g(bbox, rect=1.0, fill=1.0, method="otsu"):
        from types import SimpleNamespace

        return SimpleNamespace(bbox=bbox, rectangularity=rect, quad_fill=fill, detection_method=method, confidence=0.45)

    def test_a_clean_page_near_the_left_edge_gets_the_relaxed_threshold(self):
        det = self._det()
        assert det._present_threshold(self._g((120, 400, 600, 380))) == pytest.approx(0.42)
        assert det._present_threshold(self._g((600, 400, 600, 380))) == pytest.approx(0.5)  # the middle: unchanged

    def test_a_page_near_the_bottom_or_a_corner_is_covered_too(self):
        det = self._det()
        assert det._present_threshold(self._g((600, 600, 600, 440))) == pytest.approx(0.42)  # 40 px from the bottom
        assert det._present_threshold(self._g((60, 50, 500, 320))) == pytest.approx(0.42)  # a corner

    def test_a_page_touching_the_border_is_not_relaxed(self):
        det = self._det()
        assert det._present_threshold(self._g((0, 400, 600, 380))) == pytest.approx(0.5)
        assert det._present_threshold(self._g((600, 700, 600, 380))) == pytest.approx(0.5)  # bottom edge at 1080

    def test_a_ragged_outline_near_the_edge_is_not_relaxed(self):
        det = self._det()
        assert det._present_threshold(self._g((120, 400, 600, 380), rect=0.7)) == pytest.approx(0.5)
        assert det._present_threshold(self._g((120, 400, 600, 380), fill=0.8)) == pytest.approx(0.5)

    def test_it_stacks_with_the_hand_relaxation_and_can_be_switched_off(self):
        det = self._det()
        assert det._present_threshold(self._g((120, 400, 600, 380), method="bg_otsu_noskin")) == pytest.approx(0.37)
        off = DocumentDetector(DetectionConfig(min_geometry_confidence=0.5, edge_zone_relax=0.0))
        off._last_frame_size = (1920, 1080)
        assert off._present_threshold(self._g((120, 400, 600, 380))) == pytest.approx(0.5)

    def test_a_page_near_the_edge_that_scores_below_the_normal_threshold_is_detected(self):
        det = DocumentDetector(DetectionConfig(min_geometry_confidence=0.5, edge_zone_relax=0.08, detect_frames_required=3))
        from types import SimpleNamespace

        geom = SimpleNamespace(
            confidence=0.45, bbox=(120, 400, 600, 380), corners=((0, 0),) * 4, edge_strength=0.0, area_fraction=0.1,
            rectangularity=1.0, quad_fill=1.0, detection_method="otsu", swallows_hand=False,
        )
        det._find_geometry = lambda frame: geom  # type: ignore[method-assign]
        det._last_frame_size = (1920, 1080)
        results = [det.detect(np.zeros((120, 160, 3), np.uint8)) for _ in range(3)]
        assert results[-1] is True
