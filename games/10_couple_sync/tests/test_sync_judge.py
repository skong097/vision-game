"""
test_sync_judge.py — 싱크 판정 단위 테스트
================================================

검증:
- 둘 다 매칭 / 한쪽만 / 둘 다 미매칭
- pair_score = min
- bonus 임계 (둘 다 0.80 이상)
- None features 처리
- 잘못된 target_pose 처리
"""

import pytest

from pose_classifier import (
    POSE_CLAP,
    POSE_PROFILES,
    POSE_T,
    POSE_Y,
)
from sync_judge import (
    SYNC_BONUS_THRESHOLD,
    SyncResult,
    judge_sync,
)


# ============================================================
# 헬퍼 — 특정 포즈의 ideal feature dict 생성
# ============================================================
def ideal_features(pose: str) -> dict:
    base = {
        "left_arm_angle":   90.0,
        "right_arm_angle":  90.0,
        "left_hand_y_off":  0.0,
        "right_hand_y_off": 0.0,
        "hand_distance":    1.0,
        "hand_y_avg":       0.0,
    }
    profile = POSE_PROFILES[pose]
    for fname, (low, high) in profile.items():
        base[fname] = (low + high) / 2.0
    return base


# ============================================================
# 1. 둘 다 매칭
# ============================================================
class TestBothMatch:
    def test_both_perfect_match(self):
        l = ideal_features(POSE_T)
        r = ideal_features(POSE_T)
        result = judge_sync(POSE_T, l, r, threshold=0.7)
        assert result.both_match is True
        assert result.pair_score == pytest.approx(1.0, abs=1e-6)
        assert result.left_score == pytest.approx(1.0)
        assert result.right_score == pytest.approx(1.0)
        assert result.bonus is True

    def test_both_at_threshold(self):
        """정확히 임계점에서 매칭."""
        # T자 left_arm_angle (155, 180), 정확히 155 → 1.0
        # 다른 feature는 중심이라 1.0
        # 모두 1.0 → match O
        l = ideal_features(POSE_T)
        r = ideal_features(POSE_T)
        result = judge_sync(POSE_T, l, r, threshold=1.0)
        assert result.both_match is True


# ============================================================
# 2. 한쪽만 매칭
# ============================================================
class TestOneMismatch:
    def test_left_only(self):
        """왼쪽만 T자, 오른쪽은 Y자 → match X."""
        l = ideal_features(POSE_T)
        r = ideal_features(POSE_Y)
        result = judge_sync(POSE_T, l, r, threshold=0.7)
        assert result.both_match is False
        assert result.left_score > 0.7
        # Y자의 features가 T자 profile에서 낮은 점수
        assert result.right_score < 0.7
        assert result.pair_score == result.right_score
        assert result.bonus is False

    def test_right_only(self):
        l = ideal_features(POSE_Y)
        r = ideal_features(POSE_T)
        result = judge_sync(POSE_T, l, r, threshold=0.7)
        assert result.both_match is False
        assert result.right_score > 0.7
        assert result.left_score < 0.7


# ============================================================
# 3. 둘 다 미매칭
# ============================================================
class TestBothMismatch:
    def test_both_wrong_pose(self):
        l = ideal_features(POSE_Y)
        r = ideal_features(POSE_CLAP)
        result = judge_sync(POSE_T, l, r, threshold=0.7)
        assert result.both_match is False
        assert result.pair_score < 0.7

    def test_pair_score_is_min(self):
        """pair_score는 항상 두 점수의 최소값."""
        # T자 ideal vs Y자 ideal — 한쪽이 다른 쪽보다 낮음
        l = ideal_features(POSE_T)
        r = ideal_features(POSE_Y)
        result = judge_sync(POSE_T, l, r, threshold=0.5)
        assert result.pair_score == min(result.left_score, result.right_score)


# ============================================================
# 4. None features (한쪽 미추적)
# ============================================================
class TestNone:
    def test_left_none(self):
        r = ideal_features(POSE_T)
        result = judge_sync(POSE_T, None, r, threshold=0.7)
        assert result.left_score == 0.0
        assert result.right_score > 0.7
        assert result.both_match is False
        assert result.pair_score == 0.0

    def test_both_none(self):
        result = judge_sync(POSE_T, None, None, threshold=0.7)
        assert result.left_score == 0.0
        assert result.right_score == 0.0
        assert result.both_match is False
        assert result.bonus is False


# ============================================================
# 5. 보너스 (둘 다 0.80 이상)
# ============================================================
class TestBonus:
    def test_bonus_when_both_high(self):
        l = ideal_features(POSE_T)  # 1.0
        r = ideal_features(POSE_T)  # 1.0
        result = judge_sync(POSE_T, l, r, threshold=0.5)
        assert result.bonus is True

    def test_no_bonus_when_one_lower(self):
        # 한쪽이 0.80 미만이면 보너스 X
        # T자 ideal로 시작, left_arm_angle을 살짝 벗어남
        l = ideal_features(POSE_T)
        l["left_arm_angle"] = 130.0  # T자 (155, 180) 밖
        # width 25 → d=25, score = 1 - 25/25 = 0 (이 feature) → min 0
        r = ideal_features(POSE_T)
        result = judge_sync(POSE_T, l, r, threshold=0.5)
        assert result.left_score < SYNC_BONUS_THRESHOLD
        assert result.bonus is False

    def test_bonus_threshold_value(self):
        assert SYNC_BONUS_THRESHOLD == 0.80


# ============================================================
# 6. 결과 객체 / API
# ============================================================
class TestApi:
    def test_returns_sync_result(self):
        l = ideal_features(POSE_T)
        r = ideal_features(POSE_T)
        result = judge_sync(POSE_T, l, r, threshold=0.7)
        assert isinstance(result, SyncResult)

    def test_invalid_pose_returns_zero_scores(self):
        # 알 수 없는 포즈 — pose_similarity ValueError → 점수 0
        l = ideal_features(POSE_T)
        r = ideal_features(POSE_T)
        result = judge_sync("unknown_pose", l, r, threshold=0.5)
        assert result.left_score == 0.0
        assert result.right_score == 0.0
