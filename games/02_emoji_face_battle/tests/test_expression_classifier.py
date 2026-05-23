"""
test_expression_classifier.py - 표정 분류 + 유사도 단위 테스트
"""

import pytest

from expression_classifier import (
    ALL_EXPRESSIONS, EXPR_ANGRY, EXPR_NEUTRAL, EXPR_SAD,
    EXPR_SMILE, EXPR_SURPRISED, EXPRESSION_PROFILES,
    NEUTRAL_THRESHOLD, _feature_score,
    classify_expression, expression_similarity,
)


# ============================================================
# 합성 feature dict 헬퍼 — 각 표정의 "이상적" feature
# ============================================================
def smile_features():
    """완벽한 웃음 feature (profile 중앙값)"""
    return {
        "mouth_corner_lift": -0.10,    # 입꼬리 위로
        "mouth_aspect_ratio": 6.0,     # 가로로 늘어남
        "mouth_open": 0.08,
        "eye_open_avg": 0.20,          # 웃을 때 눈 살짝 좁아짐
        "brow_height_avg": 0.0,
        "brow_furrow": 0.40,           # 일반 (angry 아님)
    }


def sad_features():
    return {
        "mouth_corner_lift": 0.10,     # 입꼬리 아래 (sweet middle)
        "mouth_aspect_ratio": 4.0,
        "mouth_open": 0.06,
        "eye_open_avg": 0.30,
        "brow_height_avg": 0.05,
        "brow_furrow": 0.40,
    }


def surprised_features():
    return {
        "mouth_corner_lift": 0.0,
        "mouth_aspect_ratio": 1.5,
        "mouth_open": 0.25,             # 입 크게 벌림
        "eye_open_avg": 0.45,           # 눈 크게 뜸
        "brow_height_avg": -0.02,
        "brow_furrow": 0.40,
    }


def angry_features():
    return {
        "mouth_corner_lift": 0.0,
        "mouth_aspect_ratio": 3.0,      # smile 영역(4.5+) 밖
        "mouth_open": 0.06,
        "eye_open_avg": 0.30,
        "brow_height_avg": 0.05,
        "brow_furrow": 0.20,            # 눈썹 좁아짐 (핵심)
    }


def neutral_features():
    """어느 표정에도 부합하지 않는 평소 얼굴

    의도: 모든 표정과 0.70 미만 유사도 → NEUTRAL로 분류되어야 함.
    """
    return {
        "mouth_corner_lift": 0.0,       # 정확히 0 (smile/sad 모두 가까움)
        "mouth_aspect_ratio": 3.5,      # 어느 표정 sweet spot도 벗어남
        "mouth_open": 0.06,             # 일반 닫힌 입
        "eye_open_avg": 0.27,           # 보통
        "brow_height_avg": 0.05,        # 일반
        "brow_furrow": 0.35,            # 정상 너비 (angry 아님)
    }


# ============================================================
# Test: _feature_score (내부 헬퍼)
# ============================================================
class TestFeatureScore:
    def test_inside_range_returns_one(self):
        assert _feature_score(5.0, 4.0, 6.0) == 1.0
        assert _feature_score(4.0, 4.0, 6.0) == 1.0   # 경계
        assert _feature_score(6.0, 4.0, 6.0) == 1.0

    def test_below_range_decays(self):
        # 범위 [4, 6], width=2. value=3 (1만큼 아래) → 1 - 1/2 = 0.5 (1*width fall-off)
        assert _feature_score(3.0, 4.0, 6.0) == pytest.approx(0.5)

    def test_above_range_decays(self):
        assert _feature_score(7.0, 4.0, 6.0) == pytest.approx(0.5)

    def test_far_outside_returns_zero(self):
        # 1*width 거리에서 정확히 0
        assert _feature_score(2.0, 4.0, 6.0) == 0.0   # width=2, 2 거리
        assert _feature_score(8.0, 4.0, 6.0) == 0.0
        assert _feature_score(20.0, 4.0, 6.0) == 0.0

    def test_zero_width_no_crash(self):
        # high == low (실수로 만들었을 때)
        score = _feature_score(5.0, 5.0, 5.0)
        assert 0.0 <= score <= 1.0


# ============================================================
# Test: 유사도 (ideal feature는 1.0에 가까워야)
# ============================================================
class TestSimilarityIdeal:
    def test_smile_features_score_high_for_smile(self):
        s = expression_similarity(EXPR_SMILE, smile_features())
        assert s >= 0.9, f"이상적 웃음의 유사도 {s:.2f}가 너무 낮음"

    def test_sad_features_score_high_for_sad(self):
        s = expression_similarity(EXPR_SAD, sad_features())
        assert s >= 0.9

    def test_surprised_features_score_high(self):
        s = expression_similarity(EXPR_SURPRISED, surprised_features())
        assert s >= 0.9

    def test_angry_features_score_high(self):
        s = expression_similarity(EXPR_ANGRY, angry_features())
        assert s >= 0.9


class TestSimilarityCrossLow:
    """이상적 X 표정 features는 다른 표정 Y에 대해 낮은 유사도여야 함"""

    @pytest.mark.parametrize("right,wrong_features", [
        (EXPR_SMILE, sad_features()),
        (EXPR_SMILE, surprised_features()),
        (EXPR_SAD, smile_features()),
        (EXPR_SURPRISED, smile_features()),
        (EXPR_SURPRISED, sad_features()),
        (EXPR_ANGRY, smile_features()),
    ])
    def test_wrong_features_low_score(self, right, wrong_features):
        s = expression_similarity(right, wrong_features)
        # 다른 표정 features는 유사도 0.7 미만이어야 분류 가능
        assert s < 0.7, (
            f"{right}에 대해 잘못된 features가 유사도 {s:.2f} (너무 높음)"
        )


class TestSimilarityNeutralLow:
    """중립 features는 어느 표정에도 NEUTRAL_THRESHOLD 미만이어야"""

    @pytest.mark.parametrize("expr", ALL_EXPRESSIONS)
    def test_neutral_below_neutral_threshold(self, expr):
        s = expression_similarity(expr, neutral_features())
        assert s < NEUTRAL_THRESHOLD, (
            f"중립 features가 {expr}에 대해 유사도 {s:.2f}로 NEUTRAL 분류 안 됨"
        )


class TestSimilarityErrors:
    def test_unknown_expression_raises(self):
        with pytest.raises(ValueError):
            expression_similarity("happy_dance", smile_features())

    def test_missing_feature_treated_as_zero(self):
        # mouth_corner_lift 누락
        bad = {"mouth_aspect_ratio": 6.0}
        s = expression_similarity(EXPR_SMILE, bad)
        # 한 feature는 1.0, 다른 한 feature는 0 → min == 0
        assert s == 0.0


# ============================================================
# Test: 분류 (가장 가까운 표정)
# ============================================================
class TestClassify:
    @pytest.mark.parametrize("expected,features", [
        (EXPR_SMILE, smile_features()),
        (EXPR_SAD, sad_features()),
        (EXPR_SURPRISED, surprised_features()),
        (EXPR_ANGRY, angry_features()),
    ])
    def test_ideal_features_classified_correctly(self, expected, features):
        expr, conf = classify_expression(features)
        assert expr == expected, f"{expected} features → {expr} 분류"
        assert conf >= 0.8

    def test_neutral_features_classified_as_neutral(self):
        expr, conf = classify_expression(neutral_features())
        # 중립 features는 모든 표정과 NEUTRAL_THRESHOLD 미만이라 NEUTRAL이어야
        assert expr == EXPR_NEUTRAL, (
            f"중립 features가 {expr}로 분류됨 (conf {conf:.2f})"
        )

    def test_classification_returns_tuple(self):
        result = classify_expression(smile_features())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_neutral_threshold_constant(self):
        assert 0.0 < NEUTRAL_THRESHOLD < 1.0


# ============================================================
# Test: profile sanity
# ============================================================
class TestProfiles:
    def test_all_expressions_have_profile(self):
        for expr in ALL_EXPRESSIONS:
            assert expr in EXPRESSION_PROFILES
            assert len(EXPRESSION_PROFILES[expr]) >= 1

    def test_profile_ranges_are_valid(self):
        for expr, profile in EXPRESSION_PROFILES.items():
            for feature, (low, high) in profile.items():
                assert low <= high, f"{expr}.{feature}: low {low} > high {high}"
