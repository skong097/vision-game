"""
expression_classifier.py — 표정 4종 분류 + 유사도 점수
========================================================

face_features.extract_face_features() 결과를 입력으로 받아:
- classify_expression(): 가장 가까운 표정 + 신뢰도
- expression_similarity(): 특정 표정 대비 유사도 0~1

순수 함수 모듈. MediaPipe/카메라 의존성 없음 → 풍부한 단위 테스트.

설계 방식:
- 각 표정에 **profile** = {feature_name: (low, high) 기대 범위}
- 한 feature가 범위 안에 들면 score 1.0
- 범위 바깥이면 거리에 따라 선형 감소 (2*range_width 거리에서 0)
- 표정 유사도 = 모든 feature score의 평균

W4 Step 2 시점은 일반적인 비율 가정으로 시작.
실기 테스트 후 본인 얼굴로 임계값 튜닝 (W4 Step 5).

Author: Stephen (gjkong)
Date: 2026-05-05 (W4 Step 2)
"""


# ============================================================
# 1. 표정 종류
# ============================================================
EXPR_SMILE = "smile"        # 웃음
EXPR_SAD = "sad"            # 슬픔
EXPR_SURPRISED = "surprised"  # 놀람
EXPR_ANGRY = "angry"        # 화남
EXPR_NEUTRAL = "neutral"    # 무표정 (분류 결과로만 등장, 출제 X)

ALL_EXPRESSIONS = (EXPR_SMILE, EXPR_SAD, EXPR_SURPRISED, EXPR_ANGRY)
GAME_EXPRESSIONS = ALL_EXPRESSIONS  # 게임 출제 대상


# ============================================================
# 2. 표정 profile (초기값 — 실기 튜닝 전)
# ============================================================
# 각 feature: (low, high) 기대 범위
EXPRESSION_PROFILES = {
    EXPR_SMILE: {
        # 입꼬리가 아래입술보다 위 (음수)
        "mouth_corner_lift": (-0.20, -0.05),
        # 입이 가로로 늘어남
        "mouth_aspect_ratio": (4.5, 8.0),
    },
    EXPR_SAD: {
        # 입꼬리가 아래입술보다 아래 (양수, 0.05 이상으로 strict)
        "mouth_corner_lift": (0.05, 0.20),
        # 입은 닫혀있거나 살짝
        "mouth_open": (0.02, 0.10),
    },
    EXPR_SURPRISED: {
        # 입을 위아래로 크게 벌림 (low 0.18 strict — 일반 무표정과 차별)
        "mouth_open": (0.18, 0.50),
        # 눈 크게 뜸
        "eye_open_avg": (0.32, 0.60),
    },
    EXPR_ANGRY: {
        # 눈썹 안쪽이 좁아짐 (찌푸림, 0.27 미만)
        "brow_furrow": (0.10, 0.27),
        # 눈썹 처짐 또는 평소보다 낮음
        "brow_height_avg": (-0.05, 0.10),
    },
}


# ============================================================
# 3. 신뢰 임계 — 어느 표정에도 0.70 미만이면 NEUTRAL
# ============================================================
# 0.70은 "한 feature는 완벽 매치 + 다른 feature는 ~50% 매치" 수준이 NEUTRAL의
# 상한이 되도록 결정. min 집계와 함께 분류기를 strict하게 유지.
NEUTRAL_THRESHOLD = 0.70


# ============================================================
# 4. Feature score (범위 비교 + 선형 감소)
# ============================================================
def _feature_score(value: float, low: float, high: float) -> float:
    """value가 [low, high] 범위에 있으면 1.0,
    바깥이면 거리에 따라 선형 감소 (1*width 거리에서 0).

    fall-off 단위가 width 기준이라 feature 단위가 작은 값(0.05) ~ 큰 값(7.0)
    뭐든 일관된 비례 감쇠.
    """
    if low <= value <= high:
        return 1.0
    width = max(high - low, 1e-6)
    if value < low:
        d = (low - value) / width
    else:
        d = (value - high) / width
    # 1*width 거리에서 0 (가파른 fall-off로 cross-talk 줄임)
    return max(0.0, 1.0 - d)


# ============================================================
# 5. 유사도 (target 표정 대비)
# ============================================================
def expression_similarity(target: str, features: dict) -> float:
    """target 표정의 profile에 features가 얼마나 부합하는지 0~1.

    집계: **min(per-feature score)** — 한 feature라도 크게 벗어나면 전체 0에 가까움.
    평균 대신 min을 쓰는 이유: 표정은 핵심 특징이 모두 충족되어야 명확.
    예) 웃음은 입꼬리 위 + 입 가로 늘어남 둘 다 있어야 웃음.

    Args:
        target: ALL_EXPRESSIONS 중 하나
        features: face_features.extract_face_features() 결과
    Returns:
        0.0~1.0
    Raises:
        ValueError: 알 수 없는 target
    """
    if target not in EXPRESSION_PROFILES:
        raise ValueError(f"알 수 없는 표정: {target}")

    profile = EXPRESSION_PROFILES[target]
    if not profile:
        return 0.0

    scores = []
    for feature_name, (low, high) in profile.items():
        if feature_name not in features:
            # feature 누락 시 0점 (min과 일관 — 누락된 feature는 모름)
            scores.append(0.0)
        else:
            scores.append(_feature_score(features[feature_name], low, high))
    return min(scores) if scores else 0.0


# ============================================================
# 6. 분류 (모든 표정 중 가장 유사한 것)
# ============================================================
def classify_expression(features: dict) -> tuple:
    """features를 가장 잘 설명하는 표정 + 신뢰도 반환

    Returns:
        (expression, confidence): expression은 ALL_EXPRESSIONS 중 하나
                                   또는 EXPR_NEUTRAL (모두 임계 미만 시),
                                   confidence는 0~1 (해당 표정 유사도)
    """
    best_expr = EXPR_NEUTRAL
    best_score = 0.0
    for expr in ALL_EXPRESSIONS:
        s = expression_similarity(expr, features)
        if s > best_score:
            best_score = s
            best_expr = expr

    if best_score < NEUTRAL_THRESHOLD:
        return (EXPR_NEUTRAL, best_score)
    return (best_expr, best_score)
