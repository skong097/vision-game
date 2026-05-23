"""
test_ar_overlay.py — 중앙 ROI 판정 + 헬퍼 단위 테스트
"""

from dataclasses import dataclass

import pytest

from ar_overlay import (
    CenterROI,
    DEFAULT_ROI_RATIO,
    bbox_center,
    collect_targets,
    compute_center_roi,
    find_target_in_roi,
)


@dataclass
class _Det:
    class_name: str
    confidence: float
    bbox: tuple


# ============================================================
# 1. CenterROI API
# ============================================================
class TestCenterROI:
    def test_center(self):
        r = CenterROI(100, 200, 300, 500)
        assert r.center == (200, 350)

    def test_width_height(self):
        r = CenterROI(100, 200, 300, 500)
        assert r.width == 200
        assert r.height == 300

    def test_contains_point_inside(self):
        r = CenterROI(100, 200, 300, 500)
        assert r.contains_point(200, 350)

    def test_contains_point_outside(self):
        r = CenterROI(100, 200, 300, 500)
        assert not r.contains_point(50, 350)
        assert not r.contains_point(200, 600)

    def test_contains_edge_inclusive(self):
        r = CenterROI(100, 200, 300, 500)
        assert r.contains_point(100, 200)
        assert r.contains_point(300, 500)

    def test_to_tuple(self):
        r = CenterROI(10, 20, 30, 40)
        assert r.to_tuple() == (10, 20, 30, 40)


# ============================================================
# 2. compute_center_roi
# ============================================================
class TestComputeCenterROI:
    def test_default_ratio(self):
        r = compute_center_roi(640, 480)
        # 30% of min(640, 480) = 144
        assert r.width == 144
        assert r.height == 144

    def test_center_at_middle(self):
        r = compute_center_roi(640, 480)
        cx, cy = r.center
        assert abs(cx - 320) <= 1
        assert abs(cy - 240) <= 1

    def test_smaller_ratio(self):
        small = compute_center_roi(640, 480, ratio=0.1)
        big = compute_center_roi(640, 480, ratio=0.5)
        assert small.width < big.width

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            compute_center_roi(640, 480, ratio=0)
        with pytest.raises(ValueError):
            compute_center_roi(640, 480, ratio=1.5)


# ============================================================
# 3. bbox_center
# ============================================================
class TestBboxCenter:
    def test_basic(self):
        assert bbox_center((10, 20, 30, 40)) == (20, 30)

    def test_zero_size(self):
        assert bbox_center((100, 200, 100, 200)) == (100, 200)


# ============================================================
# 4. find_target_in_roi
# ============================================================
class TestFindTarget:
    def setup_method(self):
        self.roi = CenterROI(100, 100, 300, 300)

    def test_returns_none_when_no_match(self):
        dets = [_Det("dog", 0.9, (150, 150, 250, 250))]
        r = find_target_in_roi(dets, target_class="cup", roi=self.roi)
        assert r is None

    def test_finds_target_in_roi(self):
        dets = [
            _Det("cup", 0.8, (150, 150, 250, 250)),  # 중심 (200, 200) ∈ ROI
            _Det("cup", 0.5, (10, 10, 50, 50)),       # 중심 (30, 30) ROI 밖
        ]
        r = find_target_in_roi(dets, target_class="cup", roi=self.roi)
        assert r is not None
        assert r.confidence == 0.8

    def test_returns_highest_conf_in_roi(self):
        dets = [
            _Det("cup", 0.6, (150, 150, 250, 250)),
            _Det("cup", 0.9, (180, 180, 220, 220)),
        ]
        r = find_target_in_roi(dets, target_class="cup", roi=self.roi)
        assert r.confidence == 0.9

    def test_min_confidence_filters(self):
        dets = [_Det("cup", 0.5, (200, 200, 220, 220))]
        r = find_target_in_roi(dets, target_class="cup", roi=self.roi,
                                min_confidence=0.7)
        assert r is None

    def test_outside_roi_not_picked(self):
        dets = [_Det("cup", 0.95, (10, 10, 50, 50))]
        r = find_target_in_roi(dets, target_class="cup", roi=self.roi)
        assert r is None

    def test_ignores_other_class(self):
        dets = [
            _Det("dog", 0.9, (200, 200, 220, 220)),
            _Det("cup", 0.5, (200, 200, 220, 220)),
        ]
        r = find_target_in_roi(dets, target_class="cup", roi=self.roi)
        assert r is not None
        assert r.class_name == "cup"


# ============================================================
# 5. collect_targets
# ============================================================
class TestCollectTargets:
    def test_collects_class(self):
        dets = [
            _Det("cup", 0.7, (0, 0, 10, 10)),
            _Det("cup", 0.4, (100, 100, 110, 110)),
            _Det("dog", 0.9, (0, 0, 10, 10)),
        ]
        out = collect_targets(dets, "cup")
        assert len(out) == 2

    def test_min_conf_filter(self):
        dets = [
            _Det("cup", 0.7, (0, 0, 10, 10)),
            _Det("cup", 0.4, (100, 100, 110, 110)),
        ]
        out = collect_targets(dets, "cup", min_confidence=0.5)
        assert len(out) == 1
        assert out[0].confidence == 0.7

    def test_empty_list(self):
        assert collect_targets([], "cup") == []
