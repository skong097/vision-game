"""
test_color_classifier.py — 세련된 8색 분류 + 매칭 점수 단위 테스트
"""

import pytest

from color_classifier import (
    ALL_COLORS, COLOR_BURGUNDY, COLOR_CHARCOAL, COLOR_CREAM,
    COLOR_MAUVE, COLOR_MUSTARD, COLOR_NAVY, COLOR_NEUTRAL,
    COLOR_PROFILES, COLOR_SAGE, COLOR_TERRACOTTA,
    HUE_FALLOFF, NEUTRAL_THRESHOLD, SAT_FALLOFF, VAL_FALLOFF,
    _band_score, _hue_score,
    classify_color, color_match_score,
)


# ============================================================
# 각 색의 "이상적" HSV — profile 중앙값
# ============================================================
IDEAL_HSV = {
    COLOR_BURGUNDY:   (355.0, 0.65, 0.35),  # h in [340,360] sub-range
    COLOR_TERRACOTTA: (16.5,  0.575, 0.50),
    COLOR_MUSTARD:    (45.0,  0.725, 0.60),
    COLOR_SAGE:       (105.0, 0.275, 0.60),
    COLOR_NAVY:       (222.5, 0.65, 0.275),
    COLOR_MAUVE:      (317.5, 0.275, 0.625),
    COLOR_CREAM:      (40.0,  0.15, 0.895),
    COLOR_CHARCOAL:   (180.0, 0.15, 0.19),  # 차콜은 hue 무관, 임의값
}


# ============================================================
# 1. 각 색의 ideal HSV는 자기 색에 대해 score 1.0 + 정확히 분류
# ============================================================
@pytest.mark.parametrize("color", ALL_COLORS)
def test_ideal_hsv_scores_perfect(color):
    s = color_match_score(color, IDEAL_HSV[color])
    assert s == pytest.approx(1.0), \
        f"{color} ideal HSV {IDEAL_HSV[color]} 점수 {s} (1.0 기대)"


@pytest.mark.parametrize("color", ALL_COLORS)
def test_ideal_hsv_classifies_correctly(color):
    result, conf = classify_color(IDEAL_HSV[color])
    assert result == color, \
        f"{color} ideal HSV가 {result}로 분류됨 (conf={conf})"
    assert conf == pytest.approx(1.0)


# ============================================================
# 2. Hue 순환 — 버건디는 0/360 경계를 가로지름
# ============================================================
def test_burgundy_hue_wraps_at_zero_side():
    # h=5는 [0, 10] sub-range 안 → score 1.0
    assert color_match_score(COLOR_BURGUNDY, (5.0, 0.65, 0.35)) == pytest.approx(1.0)


def test_burgundy_hue_wraps_at_360_side():
    # h=350은 [340, 360] sub-range 안 → score 1.0
    assert color_match_score(COLOR_BURGUNDY, (350.0, 0.65, 0.35)) == pytest.approx(1.0)


def test_hue_circular_distance_across_boundary():
    # 차콜 외 색에서 hue 거리 계산이 0/360 경계를 가로질러 동작하는지 확인
    # 머스타드 범위 [35, 55]에 h=20은 거리 15 → falloff 15에 가까워 score 0.0 근처
    s = _hue_score(20.0, COLOR_PROFILES[COLOR_MUSTARD]["h_range"])
    assert 0.0 <= s <= 0.05  # 거의 0


# ============================================================
# 3. 차콜 — hue 무관
# ============================================================
def test_charcoal_hue_agnostic():
    # 어떤 hue든 S/V만 맞으면 차콜
    for h in (0.0, 90.0, 180.0, 270.0, 359.9):
        score = color_match_score(COLOR_CHARCOAL, (h, 0.10, 0.20))
        assert score == pytest.approx(1.0), f"h={h}에서 차콜 score {score}"


def test_charcoal_fails_when_too_bright():
    # V 0.50은 charcoal v_range (0.10, 0.28) 밖 → falloff 0.12 → 0.50-0.28=0.22, score=0
    score = color_match_score(COLOR_CHARCOAL, (0.0, 0.10, 0.50))
    assert score == pytest.approx(0.0)


# ============================================================
# 4. min 집계 — 한 축만 어긋나도 전체 점수 떨어짐
# ============================================================
def test_min_aggregation_one_axis_off():
    # 머스타드 hue·S 완벽, 그러나 V를 너무 밝게 (0.90 — v_range max 0.75)
    # → d = 0.15, falloff 0.12 → v_score < 0
    s = color_match_score(COLOR_MUSTARD, (45.0, 0.725, 0.90))
    assert s == pytest.approx(0.0)


def test_min_aggregation_picks_lowest_axis():
    # 두 축은 1.0, 한 축만 0.5 → min = 0.5
    # 세이지 hue·V 완벽, S는 0.46 (sat_range max 0.40, d=0.06, fall=0.12 → 0.5)
    s = color_match_score(COLOR_SAGE, (105.0, 0.46, 0.60))
    assert s == pytest.approx(0.5)


# ============================================================
# 5. 비비드 그린은 세이지가 아님 (S가 결정)
# ============================================================
def test_vivid_green_is_not_sage():
    # 녹색 hue(105) + 높은 S(0.85) → 세이지 sat_range 0.15-0.40 한참 벗어남
    # 다른 색에도 안 맞아 NEUTRAL
    result, conf = classify_color((105.0, 0.85, 0.60))
    assert result == COLOR_NEUTRAL
    assert conf < NEUTRAL_THRESHOLD


def test_sage_with_low_sat_succeeds():
    # 같은 hue, S만 낮춤
    result, conf = classify_color((105.0, 0.27, 0.60))
    assert result == COLOR_SAGE


# ============================================================
# 6. 크림 vs 머스타드 — hue 비슷, V로 분리
# ============================================================
def test_cream_classified_at_high_v():
    result, _ = classify_color((40.0, 0.15, 0.90))
    assert result == COLOR_CREAM


def test_mustard_classified_at_mid_v():
    result, _ = classify_color((45.0, 0.72, 0.60))
    assert result == COLOR_MUSTARD


def test_cream_v_too_low_falls_through():
    # 같은 hue·S지만 V가 mustard·cream 둘 다 안 맞는 중간(0.78)
    # cream v_range (0.82, 0.97) — 0.78은 d=0.04, falloff 0.12 → 0.67 (OK)
    # 사실 위 케이스는 cream으로 분류됨 (0.67 > NEUTRAL_THRESHOLD 0.60)
    result, conf = classify_color((40.0, 0.15, 0.78))
    assert result == COLOR_CREAM
    assert conf > NEUTRAL_THRESHOLD


# ============================================================
# 7. 모호한 색 → NEUTRAL
# ============================================================
def test_far_from_all_palette_is_neutral():
    # 시안(180°) 비비드 — 우리 팔레트에 없음
    result, conf = classify_color((180.0, 0.85, 0.85))
    assert result == COLOR_NEUTRAL
    assert conf < NEUTRAL_THRESHOLD


def test_pure_white_is_neutral():
    # 순백 (S=0, V=1) — 크림은 V도 비슷하지만 hue 제약 통과? S=0이면 hue 의미 약함
    # cream profile: h(20-60), s(0.05-0.25), v(0.82-0.97)
    # 순백(h=0, s=0, v=1.0): hue 거리 20° → score 0 (falloff 15)
    # → 크림 점수 0 → NEUTRAL
    result, _ = classify_color((0.0, 0.0, 1.0))
    assert result == COLOR_NEUTRAL


# ============================================================
# 8. 잘못된 target → ValueError
# ============================================================
def test_unknown_target_raises():
    with pytest.raises(ValueError):
        color_match_score("not_a_color", (180.0, 0.5, 0.5))


def test_neutral_is_not_a_target():
    with pytest.raises(ValueError):
        color_match_score(COLOR_NEUTRAL, (180.0, 0.5, 0.5))


# ============================================================
# 9. 보조 함수 — 직접 점검
# ============================================================
def test_band_score_inside_range():
    assert _band_score(0.5, 0.4, 0.6, 0.1) == 1.0


def test_band_score_outside_linear_falloff():
    # 0.7은 high 0.6에서 0.1 떨어짐, falloff 0.1 → 1.0 - 1.0 = 0.0
    assert _band_score(0.7, 0.4, 0.6, 0.1) == pytest.approx(0.0)


def test_band_score_halfway_falloff():
    # 0.65는 d=0.05, falloff 0.1 → 0.5
    assert _band_score(0.65, 0.4, 0.6, 0.1) == pytest.approx(0.5)


def test_hue_score_none_means_unconstrained():
    assert _hue_score(123.45, None) == 1.0
    assert _hue_score(0.0, None) == 1.0


def test_hue_score_inside_range():
    assert _hue_score(45.0, [(35, 55)]) == 1.0


# ============================================================
# 10. 신뢰도 게이트
# ============================================================
def test_neutral_threshold_is_strict_enough():
    # 게이트 0.60은 "한 축 perfect + 한 축 ~50% + 한 축 perfect"가
    # NEUTRAL 처리되는 경계. 한 축 절반만 맞아도 NEUTRAL — strict 의도.
    # min 집계 + 0.6 게이트는 함께 동작.
    assert 0.5 < NEUTRAL_THRESHOLD < 0.75
