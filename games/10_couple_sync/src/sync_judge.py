"""
sync_judge.py — 두 사람 features → 싱크 판정
=================================================

순수 함수. cv2/MediaPipe 의존성 X.

판정 로직
---------
- 각 사람의 features dict로 pose_similarity(mission_pose, features) 계산
- 두 점수의 **min** = pair_score (둘 다 잘 해야 매칭)
- pair_score ≥ threshold → 매칭(true)
- 둘 다 0.80 이상이면 **싱크 보너스** 플래그

결과는 SyncResult dataclass:
- both_match: bool
- pair_score: float (min)
- left_score: float
- right_score: float
- bonus: bool (둘 다 0.80 이상)

Author: Stephen (gjkong)
Date: 2026-05-12 (W8 Step 2)
"""

from dataclasses import dataclass

try:
    from .pose_classifier import POSE_NEUTRAL, pose_similarity
except ImportError:
    from pose_classifier import POSE_NEUTRAL, pose_similarity


# ============================================================
# 1. 싱크 보너스 임계
# ============================================================
SYNC_BONUS_THRESHOLD = 0.80


# ============================================================
# 2. SyncResult
# ============================================================
@dataclass(frozen=True)
class SyncResult:
    """싱크 판정 결과.

    Attributes:
        both_match: 둘 다 매칭 (pair_score ≥ threshold)
        pair_score: min(left, right)
        left_score: 왼쪽 사람 similarity
        right_score: 오른쪽 사람 similarity
        bonus: 둘 다 0.80 이상 (보너스 점수)
    """
    both_match: bool
    pair_score: float
    left_score: float
    right_score: float
    bonus: bool


# ============================================================
# 3. 싱크 판정
# ============================================================
def judge_sync(
    target_pose: str,
    left_features,
    right_features,
    threshold: float,
) -> SyncResult:
    """두 사람의 features → target_pose 대비 싱크 판정.

    Args:
        target_pose: 미션 포즈 (POSE_T 등)
        left_features: 왼쪽 사람 features dict (또는 None — 미추적)
        right_features: 오른쪽 사람 features dict (또는 None)
        threshold: 매칭 임계 (0~1)

    Returns:
        SyncResult. features가 None이면 해당 점수는 0.0.
    """
    left_score = 0.0
    right_score = 0.0

    if left_features is not None:
        try:
            left_score = pose_similarity(target_pose, left_features)
        except (ValueError, KeyError):
            left_score = 0.0

    if right_features is not None:
        try:
            right_score = pose_similarity(target_pose, right_features)
        except (ValueError, KeyError):
            right_score = 0.0

    pair_score = min(left_score, right_score)
    both_match = pair_score >= threshold
    bonus = (
        left_score >= SYNC_BONUS_THRESHOLD
        and right_score >= SYNC_BONUS_THRESHOLD
    )

    return SyncResult(
        both_match=both_match,
        pair_score=pair_score,
        left_score=left_score,
        right_score=right_score,
        bonus=bonus,
    )
