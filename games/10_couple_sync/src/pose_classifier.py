"""
pose_classifier.py — 5종 포즈 분류 + 유사도
==============================================

pose_features.extract_pose_features() 결과를 입력으로 받아:
- classify_pose(): 가장 가까운 포즈 + 신뢰도
- pose_similarity(): 특정 포즈 대비 유사도 0~1

순수 함수 모듈. MediaPipe/카메라 의존성 없음 → 풍부한 단위 테스트.

설계 (W4 expression_classifier와 동일 패턴)
-------------------------------------------
- 각 포즈에 **profile** = {feature_name: (low, high)} 기대 범위
- 한 feature 안 들면 거리에 따라 선형 감소
- 집계: **min(per-feature score)** — 모든 핵심 feature가 동시 충족되어야 함
- 임계 미만이면 NEUTRAL (무자세)

Author: Stephen (gjkong)
Date: 2026-05-12 (W6 Step 2)
"""


# ============================================================
# 1. 포즈 종류
# ============================================================
POSE_T = "t_pose"
POSE_Y = "y_pose"
POSE_LEFT_UP = "left_up"
POSE_RIGHT_UP = "right_up"
POSE_CLAP = "clap"
POSE_NEUTRAL = "neutral"

ALL_POSES = (POSE_T, POSE_Y, POSE_LEFT_UP, POSE_RIGHT_UP, POSE_CLAP)
GAME_POSES = ALL_POSES


# ============================================================
# 2. Pose profile
# ============================================================
# y_off 음수 = 위. shoulder 너비 단위.
# 거리 단위는 어깨 너비.
POSE_PROFILES = {
    # T자: 양팔 펴짐 + 손 어깨 높이
    POSE_T: {
        "left_arm_angle":   (155.0, 180.0),
        "right_arm_angle":  (155.0, 180.0),
        "left_hand_y_off":  (-0.20, 0.20),
        "right_hand_y_off": (-0.20, 0.20),
        "hand_distance":    (2.4, 4.0),
    },
    # Y자: 양손 위로 + 팔 펴짐
    POSE_Y: {
        "left_hand_y_off":  (-2.0, -0.55),
        "right_hand_y_off": (-2.0, -0.55),
        "left_arm_angle":   (130.0, 180.0),
        "right_arm_angle":  (130.0, 180.0),
    },
    # 왼손 위 + 오른팔 옆 (T자처럼)
    POSE_LEFT_UP: {
        "left_hand_y_off":  (-2.0, -0.55),
        "right_hand_y_off": (-0.20, 0.20),
        "right_arm_angle":  (140.0, 180.0),
    },
    # 오른손 위 + 왼팔 옆
    POSE_RIGHT_UP: {
        "right_hand_y_off": (-2.0, -0.55),
        "left_hand_y_off":  (-0.20, 0.20),
        "left_arm_angle":   (140.0, 180.0),
    },
    # 박수: 양손 모음
    POSE_CLAP: {
        "hand_distance":  (0.0, 0.45),
        "hand_y_avg":     (-0.2, 0.5),
    },
}


# ============================================================
# 3. 임계
# ============================================================
NEUTRAL_THRESHOLD = 0.70


# ============================================================
# 4. Feature score (W4 expression_classifier와 동일)
# ============================================================
def _feature_score(value: float, low: float, high: float) -> float:
    """value가 [low, high]에 있으면 1.0, 바깥이면 width 단위로 선형 감소."""
    if low <= value <= high:
        return 1.0
    width = max(high - low, 1e-6)
    if value < low:
        d = (low - value) / width
    else:
        d = (value - high) / width
    return max(0.0, 1.0 - d)


# ============================================================
# 5. 유사도
# ============================================================
def pose_similarity(target: str, features: dict) -> float:
    """target 포즈 profile 대비 features의 유사도 0~1.

    집계: min(per-feature score) — 핵심 feature 모두 동시 충족 필요.

    Raises:
        ValueError: 알 수 없는 target
    """
    if target not in POSE_PROFILES:
        raise ValueError(f"알 수 없는 포즈: {target}")

    profile = POSE_PROFILES[target]
    if not profile:
        return 0.0

    scores = []
    for fname, (low, high) in profile.items():
        if fname not in features:
            scores.append(0.0)
        else:
            scores.append(_feature_score(features[fname], low, high))
    return min(scores) if scores else 0.0


# ============================================================
# 6. 분류
# ============================================================
def classify_pose(features: dict) -> tuple:
    """가장 잘 설명하는 포즈 + 신뢰도.

    Returns:
        (pose, confidence). 모두 임계 미만 시 (NEUTRAL, best_score).
    """
    best = POSE_NEUTRAL
    best_score = 0.0
    for pose in ALL_POSES:
        s = pose_similarity(pose, features)
        if s > best_score:
            best_score = s
            best = pose

    if best_score < NEUTRAL_THRESHOLD:
        return (POSE_NEUTRAL, best_score)
    return (best, best_score)
