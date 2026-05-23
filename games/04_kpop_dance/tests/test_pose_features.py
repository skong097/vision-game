"""
test_pose_features.py — Pose feature 추출 단위 테스트
========================================================

검증:
- 어깨 너비 정규화 (scale-invariance)
- 팔 각도: 펴짐 ≈ 180°, 굽힘 < 90°
- 손 y 오프셋 부호 (위 = 음수)
- 양손 거리 (T자에서 크고, 박수에서 작음)
- 평균 y (Y자에서 음수 큰 값)
- edge: 어깨 너비 0, landmark 좌표 동일
"""

import math
from collections import namedtuple

import pytest

from pose_features import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    _angle_deg,
    _dist,
    extract_pose_features,
)


# ============================================================
# 픽스처: Pose-like landmark 시뮬레이션
# ============================================================
LM = namedtuple("LM", ["x", "y"])


def make_pose(coords):
    """coords: dict {index: (x, y)} → 33-element list (사용 안 하는 인덱스는 dummy)."""
    arr = [LM(0.5, 0.5)] * 33
    for i, (x, y) in coords.items():
        arr[i] = LM(x, y)
    return arr


# 기준: 어깨 너비 0.20, 어깨 y = 0.40, 손목 y는 포즈마다
SX_LEFT, SX_RIGHT = 0.40, 0.60   # 어깨 x (왼쪽이 화면 좌측이라 더 작은 값)
SHOULDER_Y = 0.40
SHOULDER_WIDTH = 0.20


def t_pose_landmarks(y_offset=0.0):
    """T자: 양팔 수평. 손목은 어깨 y와 같음.
    y_offset != 0이면 손이 살짝 위/아래."""
    sy = SHOULDER_Y
    wy = sy + y_offset
    return make_pose({
        LEFT_SHOULDER:  (SX_LEFT, sy),
        RIGHT_SHOULDER: (SX_RIGHT, sy),
        LEFT_ELBOW:     (SX_LEFT - 0.10, sy),
        RIGHT_ELBOW:    (SX_RIGHT + 0.10, sy),
        LEFT_WRIST:     (SX_LEFT - 0.20, wy),
        RIGHT_WRIST:    (SX_RIGHT + 0.20, wy),
    })


def y_pose_landmarks():
    """Y자: 양팔 위로 V자. 손목 y는 어깨보다 위(작음)."""
    sy = SHOULDER_Y
    return make_pose({
        LEFT_SHOULDER:  (SX_LEFT, sy),
        RIGHT_SHOULDER: (SX_RIGHT, sy),
        LEFT_ELBOW:     (SX_LEFT - 0.05, sy - 0.10),
        RIGHT_ELBOW:    (SX_RIGHT + 0.05, sy - 0.10),
        LEFT_WRIST:     (SX_LEFT - 0.10, sy - 0.25),
        RIGHT_WRIST:    (SX_RIGHT + 0.10, sy - 0.25),
    })


def clap_landmarks():
    """박수: 양 손이 가슴 앞 중앙에 모임."""
    sy = SHOULDER_Y
    cx = 0.50
    cy = sy + 0.05
    return make_pose({
        LEFT_SHOULDER:  (SX_LEFT, sy),
        RIGHT_SHOULDER: (SX_RIGHT, sy),
        LEFT_ELBOW:     (cx - 0.05, sy + 0.03),
        RIGHT_ELBOW:    (cx + 0.05, sy + 0.03),
        LEFT_WRIST:     (cx - 0.01, cy),
        RIGHT_WRIST:    (cx + 0.01, cy),
    })


# ============================================================
# 1. 헬퍼 함수
# ============================================================
class TestHelpers:
    def test_dist_basic(self):
        assert _dist((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_dist_zero(self):
        assert _dist((1, 1), (1, 1)) == 0.0

    def test_angle_straight(self):
        # 일직선 (A--B--C)
        assert _angle_deg((0, 0), (1, 0), (2, 0)) == pytest.approx(180.0,
                                                                    abs=1e-3)

    def test_angle_right(self):
        # 직각
        assert _angle_deg((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0,
                                                                    abs=1e-3)

    def test_angle_acute(self):
        # 예각 ≈ 45도
        assert _angle_deg((1, 1), (0, 0), (1, 0)) == pytest.approx(45.0,
                                                                    abs=1e-3)

    def test_angle_zero_length(self):
        # 한 벡터가 0 → 0 반환
        assert _angle_deg((0, 0), (0, 0), (1, 0)) == 0.0


# ============================================================
# 2. T자 — 팔 펴짐 + 손 어깨 높이
# ============================================================
class TestTPose:
    def test_arm_angles_close_to_180(self):
        lm = t_pose_landmarks()
        f = extract_pose_features(lm)
        assert f["left_arm_angle"] == pytest.approx(180.0, abs=2)
        assert f["right_arm_angle"] == pytest.approx(180.0, abs=2)

    def test_hand_y_offsets_near_zero(self):
        lm = t_pose_landmarks()
        f = extract_pose_features(lm)
        assert abs(f["left_hand_y_off"]) < 0.05
        assert abs(f["right_hand_y_off"]) < 0.05
        assert abs(f["hand_y_avg"]) < 0.05

    def test_hand_distance_large(self):
        # T자에서 양손 거리는 어깨너비의 ~3배 (sw 0.20, 손 사이 0.60)
        lm = t_pose_landmarks()
        f = extract_pose_features(lm)
        assert f["hand_distance"] > 2.5


# ============================================================
# 3. Y자 — 양손 위로
# ============================================================
class TestYPose:
    def test_hand_y_offsets_negative(self):
        lm = y_pose_landmarks()
        f = extract_pose_features(lm)
        # 손이 어깨보다 위 → y_off 음수
        assert f["left_hand_y_off"] < -0.5
        assert f["right_hand_y_off"] < -0.5
        assert f["hand_y_avg"] < -0.5


# ============================================================
# 4. 박수 — 손 모음
# ============================================================
class TestClap:
    def test_hand_distance_small(self):
        lm = clap_landmarks()
        f = extract_pose_features(lm)
        # 손이 서로 가까움 → 어깨너비의 0.x 정도
        assert f["hand_distance"] < 0.5

    def test_hand_y_near_shoulder(self):
        lm = clap_landmarks()
        f = extract_pose_features(lm)
        # 손이 어깨 약간 아래
        assert -0.1 < f["hand_y_avg"] < 0.5


# ============================================================
# 5. Scale-invariance
# ============================================================
class TestScaleInvariance:
    def test_t_pose_features_independent_of_scale(self):
        """어깨 너비가 달라도 T자 feature는 동일 (정규화)."""
        # 기본 (어깨 너비 0.20)
        lm1 = t_pose_landmarks()
        # 축소판 (어깨 너비 0.10, 모든 거리 절반)
        lm2 = make_pose({
            LEFT_SHOULDER:  (0.45, 0.40),
            RIGHT_SHOULDER: (0.55, 0.40),
            LEFT_ELBOW:     (0.40, 0.40),
            RIGHT_ELBOW:    (0.60, 0.40),
            LEFT_WRIST:     (0.35, 0.40),
            RIGHT_WRIST:    (0.65, 0.40),
        })
        f1 = extract_pose_features(lm1)
        f2 = extract_pose_features(lm2)
        # 각도는 동일, 거리는 어깨 너비 비율로 동일
        assert f1["left_arm_angle"] == pytest.approx(f2["left_arm_angle"],
                                                      abs=2)
        assert f1["hand_distance"] == pytest.approx(f2["hand_distance"],
                                                     abs=0.05)


# ============================================================
# 6. Edge cases
# ============================================================
class TestEdgeCases:
    def test_zero_shoulder_width(self):
        """어깨 양쪽이 같은 점에 있을 때 → 0 division 보호."""
        lm = make_pose({
            LEFT_SHOULDER:  (0.50, 0.50),
            RIGHT_SHOULDER: (0.50, 0.50),
            LEFT_ELBOW:     (0.40, 0.50),
            RIGHT_ELBOW:    (0.60, 0.50),
            LEFT_WRIST:     (0.30, 0.50),
            RIGHT_WRIST:    (0.70, 0.50),
        })
        # 예외 없이 결과 반환 (값은 매우 큼)
        f = extract_pose_features(lm)
        assert isinstance(f["hand_distance"], float)

    def test_returns_all_keys(self):
        lm = t_pose_landmarks()
        f = extract_pose_features(lm)
        for k in ("left_arm_angle", "right_arm_angle",
                  "left_hand_y_off", "right_hand_y_off",
                  "hand_distance", "hand_y_avg"):
            assert k in f

    def test_indexing_landmark_object_works(self):
        # .x/.y 없는 인덱싱 객체 (튜플)도 지원
        coords = {
            LEFT_SHOULDER:  (SX_LEFT, SHOULDER_Y),
            RIGHT_SHOULDER: (SX_RIGHT, SHOULDER_Y),
            LEFT_ELBOW:     (SX_LEFT - 0.10, SHOULDER_Y),
            RIGHT_ELBOW:    (SX_RIGHT + 0.10, SHOULDER_Y),
            LEFT_WRIST:     (SX_LEFT - 0.20, SHOULDER_Y),
            RIGHT_WRIST:    (SX_RIGHT + 0.20, SHOULDER_Y),
        }
        arr = [(0.5, 0.5)] * 33
        for i, c in coords.items():
            arr[i] = c
        f = extract_pose_features(arr)
        assert f["left_arm_angle"] == pytest.approx(180.0, abs=2)
