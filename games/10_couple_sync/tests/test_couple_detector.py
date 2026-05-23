"""
test_couple_detector.py — 좌표 변환 단위 테스트
======================================================

검증:
- remap_landmark_to_full_frame: 좌/우 변환 정확
- 0과 1 경계 처리
- 잘못된 side 예외
- remap_landmarks: 전체 리스트 변환
- None 처리
- 다양한 landmark 객체 호환 (.x/.y or 튜플)
"""

from collections import namedtuple

import pytest

from couple_detector import (
    RemappedLandmark,
    SIDE_LEFT,
    SIDE_RIGHT,
    remap_landmark_to_full_frame,
    remap_landmarks,
    split_frame_x,
)


LM = namedtuple("LM", ["x", "y"])


# ============================================================
# 1. remap_landmark_to_full_frame
# ============================================================
class TestRemapX:
    def test_left_zero_maps_to_zero(self):
        assert remap_landmark_to_full_frame(0.0, SIDE_LEFT) == 0.0

    def test_left_half_maps_to_quarter(self):
        # ROI 중앙 (x=0.5) → 전체 frame 25%
        assert remap_landmark_to_full_frame(0.5, SIDE_LEFT) == 0.25

    def test_left_one_maps_to_half(self):
        # ROI 우측 끝 → 전체 frame 50%
        assert remap_landmark_to_full_frame(1.0, SIDE_LEFT) == 0.5

    def test_right_zero_maps_to_half(self):
        # 우측 ROI 좌측 끝 → 전체 frame 50%
        assert remap_landmark_to_full_frame(0.0, SIDE_RIGHT) == 0.5

    def test_right_half_maps_to_three_quarter(self):
        assert remap_landmark_to_full_frame(0.5, SIDE_RIGHT) == 0.75

    def test_right_one_maps_to_full(self):
        assert remap_landmark_to_full_frame(1.0, SIDE_RIGHT) == 1.0

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            remap_landmark_to_full_frame(0.5, "middle")


# ============================================================
# 2. split_frame_x
# ============================================================
class TestSplitFrame:
    def test_even_width(self):
        left_end, right_start = split_frame_x(640)
        assert left_end == 320
        assert right_start == 320

    def test_odd_width(self):
        # 정수 분할
        left_end, right_start = split_frame_x(641)
        assert left_end == 320
        assert right_start == 320


# ============================================================
# 3. remap_landmarks 리스트 변환
# ============================================================
class TestRemapLandmarks:
    def test_left_landmarks(self):
        lms = [LM(0.5, 0.5), LM(0.8, 0.3)]
        out = remap_landmarks(lms, SIDE_LEFT)
        assert len(out) == 2
        assert out[0].x == 0.25
        assert out[0].y == 0.5
        assert out[1].x == 0.4
        assert out[1].y == 0.3

    def test_right_landmarks(self):
        lms = [LM(0.5, 0.5), LM(0.8, 0.3)]
        out = remap_landmarks(lms, SIDE_RIGHT)
        assert out[0].x == 0.75
        assert out[0].y == 0.5
        assert out[1].x == pytest.approx(0.9)

    def test_returns_remapped_landmark_objects(self):
        lms = [LM(0.5, 0.5)]
        out = remap_landmarks(lms, SIDE_LEFT)
        assert isinstance(out[0], RemappedLandmark)
        assert hasattr(out[0], "x")
        assert hasattr(out[0], "y")

    def test_none_input_returns_none(self):
        assert remap_landmarks(None, SIDE_LEFT) is None

    def test_tuple_input_compatible(self):
        # (x, y) 튜플도 지원
        lms = [(0.5, 0.5), (0.2, 0.7)]
        out = remap_landmarks(lms, SIDE_LEFT)
        assert out[0].x == 0.25
        assert out[1].x == 0.1

    def test_empty_list(self):
        out = remap_landmarks([], SIDE_LEFT)
        assert out == []

    def test_invalid_side(self):
        with pytest.raises(ValueError):
            remap_landmarks([LM(0.5, 0.5)], "center")

    def test_y_unchanged(self):
        # y는 분할 영향 X
        lms = [LM(0.5, 0.3), LM(0.5, 0.7)]
        out_left = remap_landmarks(lms, SIDE_LEFT)
        out_right = remap_landmarks(lms, SIDE_RIGHT)
        for o_l, o_r in zip(out_left, out_right):
            assert o_l.y == o_r.y
