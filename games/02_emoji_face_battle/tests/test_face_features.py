"""
test_face_features.py - landmark → feature 추출 단위 테스트
"""

import math
from collections import namedtuple

import pytest

from face_features import (
    LEFT_BROW_INNER, LEFT_EYE_BOTTOM, LEFT_EYE_INNER,
    LEFT_EYE_OUTER, LEFT_EYE_TOP, LOWER_LIP_CENTER,
    MOUTH_LEFT, MOUTH_RIGHT, NOSE_TIP, RIGHT_BROW_INNER,
    RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
    RIGHT_EYE_TOP, UPPER_LIP_CENTER,
    extract_face_features,
)


# ============================================================
# 합성 landmark 헬퍼 (MediaPipe NormalizedLandmark 모방)
# ============================================================
LM = namedtuple("LM", ["x", "y", "z"])
LM.__new__.__defaults__ = (0.0,)  # z는 기본 0


def make_face_landmarks(
    *,
    eye_outer_left=(0.30, 0.40),   # 왼눈 바깥
    eye_outer_right=(0.70, 0.40),  # 오른눈 바깥
    eye_inner_left=(0.42, 0.40),
    eye_inner_right=(0.58, 0.40),
    eye_top_left=(0.36, 0.38),
    eye_bottom_left=(0.36, 0.42),
    eye_top_right=(0.64, 0.38),
    eye_bottom_right=(0.64, 0.42),
    mouth_left=(0.38, 0.65),
    mouth_right=(0.62, 0.65),
    upper_lip=(0.50, 0.62),
    lower_lip=(0.50, 0.68),
    brow_inner_left=(0.42, 0.34),
    brow_inner_right=(0.58, 0.34),
    nose_tip=(0.50, 0.50),
    n=478,
):
    """478개 landmark를 채운 리스트. 핵심 인덱스만 위 인자로 덮어씀.

    나머지는 (0.5, 0.5)로 채워짐 — 사용하지 않으므로 무관.
    """
    arr = [LM(0.5, 0.5) for _ in range(n)]

    arr[LEFT_EYE_OUTER]   = LM(*eye_outer_left)
    arr[RIGHT_EYE_OUTER]  = LM(*eye_outer_right)
    arr[LEFT_EYE_INNER]   = LM(*eye_inner_left)
    arr[RIGHT_EYE_INNER]  = LM(*eye_inner_right)
    arr[LEFT_EYE_TOP]     = LM(*eye_top_left)
    arr[LEFT_EYE_BOTTOM]  = LM(*eye_bottom_left)
    arr[RIGHT_EYE_TOP]    = LM(*eye_top_right)
    arr[RIGHT_EYE_BOTTOM] = LM(*eye_bottom_right)
    arr[MOUTH_LEFT]       = LM(*mouth_left)
    arr[MOUTH_RIGHT]      = LM(*mouth_right)
    arr[UPPER_LIP_CENTER] = LM(*upper_lip)
    arr[LOWER_LIP_CENTER] = LM(*lower_lip)
    arr[LEFT_BROW_INNER]  = LM(*brow_inner_left)
    arr[RIGHT_BROW_INNER] = LM(*brow_inner_right)
    arr[NOSE_TIP]         = LM(*nose_tip)
    return arr


# ============================================================
# Test: 기본 feature 추출
# ============================================================
class TestBasicExtraction:
    def test_returns_all_expected_keys(self):
        f = extract_face_features(make_face_landmarks())
        assert set(f.keys()) == {
            "mouth_aspect_ratio", "mouth_open", "mouth_corner_lift",
            "eye_open_avg", "brow_height_avg", "brow_furrow",
        }

    def test_all_values_are_float(self):
        f = extract_face_features(make_face_landmarks())
        for k, v in f.items():
            assert isinstance(v, float), f"{k} is {type(v)}"

    def test_default_face_neutral_features(self):
        # 기본값 얼굴은 무표정에 가까워야 함
        f = extract_face_features(make_face_landmarks())
        # 입꼬리는 아래입술과 거의 같은 높이 (수평선상 0.65 vs 0.68)
        # → corner_lift = (0.65 - 0.68) / interocular = 음수 (살짝 위)
        # 단, 이는 기본값일 뿐. 실기에선 사람마다 다름.
        # 절댓값이 너무 크지 않은지만 확인.
        assert abs(f["mouth_corner_lift"]) < 0.20


class TestScaleInvariance:
    def test_different_face_sizes_give_similar_features(self):
        """얼굴 크기가 달라져도 feature 비율은 거의 동일해야 한다."""
        small_face = make_face_landmarks(
            eye_outer_left=(0.40, 0.45),
            eye_outer_right=(0.60, 0.45),
            eye_inner_left=(0.46, 0.45),
            eye_inner_right=(0.54, 0.45),
            eye_top_left=(0.43, 0.44),
            eye_bottom_left=(0.43, 0.46),
            eye_top_right=(0.57, 0.44),
            eye_bottom_right=(0.57, 0.46),
            mouth_left=(0.44, 0.57),
            mouth_right=(0.56, 0.57),
            upper_lip=(0.50, 0.55),
            lower_lip=(0.50, 0.59),
            brow_inner_left=(0.46, 0.42),
            brow_inner_right=(0.54, 0.42),
        )
        large_face = make_face_landmarks(
            eye_outer_left=(0.20, 0.35),
            eye_outer_right=(0.80, 0.35),
            eye_inner_left=(0.38, 0.35),
            eye_inner_right=(0.62, 0.35),
            eye_top_left=(0.29, 0.32),
            eye_bottom_left=(0.29, 0.38),
            eye_top_right=(0.71, 0.32),
            eye_bottom_right=(0.71, 0.38),
            mouth_left=(0.32, 0.71),
            mouth_right=(0.68, 0.71),
            upper_lip=(0.50, 0.65),
            lower_lip=(0.50, 0.77),
            brow_inner_left=(0.38, 0.27),
            brow_inner_right=(0.62, 0.27),
        )
        f_small = extract_face_features(small_face)
        f_large = extract_face_features(large_face)
        # 같은 비율로 키운 얼굴은 거의 동일한 feature를 내야 함
        for key in f_small:
            assert abs(f_small[key] - f_large[key]) < 0.05, (
                f"{key}: small={f_small[key]:.3f}, large={f_large[key]:.3f}"
            )


class TestExpressionSensitivity:
    def test_smile_corner_lift_is_negative(self):
        # 입꼬리를 위로 (작은 y) — y가 작을수록 위
        smile = make_face_landmarks(
            mouth_left=(0.38, 0.62),    # 위
            mouth_right=(0.62, 0.62),   # 위
            upper_lip=(0.50, 0.62),
            lower_lip=(0.50, 0.66),     # 아래입술이 입꼬리보다 아래
        )
        f = extract_face_features(smile)
        assert f["mouth_corner_lift"] < -0.02, "웃음은 입꼬리가 위로 (음수)"

    def test_sad_corner_lift_is_positive(self):
        sad = make_face_landmarks(
            mouth_left=(0.38, 0.70),    # 입꼬리 아래
            mouth_right=(0.62, 0.70),
            upper_lip=(0.50, 0.62),
            lower_lip=(0.50, 0.68),     # 아래입술이 입꼬리보다 위
        )
        f = extract_face_features(sad)
        assert f["mouth_corner_lift"] > 0.02, "슬픔은 입꼬리가 아래 (양수)"

    def test_surprised_mouth_open_is_high(self):
        surprised = make_face_landmarks(
            upper_lip=(0.50, 0.58),
            lower_lip=(0.50, 0.74),     # 입을 크게 벌림
        )
        f = extract_face_features(surprised)
        assert f["mouth_open"] > 0.15, "놀람은 입이 크게 벌어짐"

    def test_surprised_eye_open_is_high(self):
        surprised = make_face_landmarks(
            eye_top_left=(0.36, 0.36),    # 눈 크게 뜸
            eye_bottom_left=(0.36, 0.44),
            eye_top_right=(0.64, 0.36),
            eye_bottom_right=(0.64, 0.44),
        )
        f = extract_face_features(surprised)
        assert f["eye_open_avg"] > 0.30, "놀람은 눈이 크게 떠짐"

    def test_angry_brow_furrow_is_low(self):
        # 눈썹 안쪽 사이가 좁아짐
        angry = make_face_landmarks(
            brow_inner_left=(0.47, 0.34),
            brow_inner_right=(0.53, 0.34),
        )
        f = extract_face_features(angry)
        assert f["brow_furrow"] < 0.20, "화남은 눈썹 사이가 좁아짐"


class TestEdgeCases:
    def test_zero_distance_no_crash(self):
        # 모든 landmark가 같은 점 — 0 division 위험
        landmarks = [LM(0.5, 0.5) for _ in range(478)]
        f = extract_face_features(landmarks)
        # crash 없이 dict 반환되는 것만 검증 (값은 의미 없음)
        assert "mouth_aspect_ratio" in f

    def test_accepts_tuple_landmarks(self):
        # (x, y) 튜플 리스트도 동작
        n = 478
        arr = [(0.5, 0.5) for _ in range(n)]
        arr[LEFT_EYE_OUTER] = (0.30, 0.40)
        arr[RIGHT_EYE_OUTER] = (0.70, 0.40)
        arr[LEFT_EYE_INNER] = (0.42, 0.40)
        arr[RIGHT_EYE_INNER] = (0.58, 0.40)
        arr[LEFT_EYE_TOP] = (0.36, 0.38)
        arr[LEFT_EYE_BOTTOM] = (0.36, 0.42)
        arr[RIGHT_EYE_TOP] = (0.64, 0.38)
        arr[RIGHT_EYE_BOTTOM] = (0.64, 0.42)
        arr[MOUTH_LEFT] = (0.38, 0.65)
        arr[MOUTH_RIGHT] = (0.62, 0.65)
        arr[UPPER_LIP_CENTER] = (0.50, 0.62)
        arr[LOWER_LIP_CENTER] = (0.50, 0.68)
        arr[LEFT_BROW_INNER] = (0.42, 0.34)
        arr[RIGHT_BROW_INNER] = (0.58, 0.34)
        f = extract_face_features(arr)
        assert isinstance(f["mouth_aspect_ratio"], float)
