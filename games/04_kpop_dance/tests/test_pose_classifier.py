"""
test_pose_classifier.py — 5종 포즈 분류 + 유사도 단위 테스트
==============================================================

검증:
- 각 포즈 ideal feature dict → similarity ≈ 1.0
- 다른 포즈는 낮은 점수
- NEUTRAL fallback
- _feature_score 경계
- 잘못된 target 예외
"""

import pytest

from pose_classifier import (
    ALL_POSES,
    GAME_POSES,
    NEUTRAL_THRESHOLD,
    POSE_CLAP,
    POSE_LEFT_UP,
    POSE_NEUTRAL,
    POSE_PROFILES,
    POSE_RIGHT_UP,
    POSE_T,
    POSE_Y,
    _feature_score,
    classify_pose,
    pose_similarity,
)


# ============================================================
# 이상적 feature dict (각 포즈의 profile 중심값)
# ============================================================
def features_for(pose: str, override=None) -> dict:
    """포즈의 모든 feature를 profile 중심값으로 채운 dict."""
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
    if override:
        base.update(override)
    return base


# ============================================================
# 1. _feature_score 경계
# ============================================================
class TestFeatureScore:
    def test_inside_range(self):
        assert _feature_score(150.0, 100.0, 180.0) == 1.0

    def test_at_low_edge(self):
        assert _feature_score(100.0, 100.0, 180.0) == 1.0

    def test_at_high_edge(self):
        assert _feature_score(180.0, 100.0, 180.0) == 1.0

    def test_below_falls_off(self):
        s = _feature_score(60.0, 100.0, 180.0)
        # 40/80 = 0.5 → 1 - 0.5 = 0.5
        assert s == pytest.approx(0.5)

    def test_far_below_clamps_zero(self):
        assert _feature_score(0.0, 100.0, 180.0) == 0.0

    def test_above_falls_off(self):
        s = _feature_score(220.0, 100.0, 180.0)
        # 40/80 = 0.5
        assert s == pytest.approx(0.5)


# ============================================================
# 2. 각 포즈 ideal features → similarity ≈ 1.0
# ============================================================
class TestPerfectMatch:
    @pytest.mark.parametrize("pose", ALL_POSES)
    def test_perfect_match_returns_one(self, pose):
        f = features_for(pose)
        s = pose_similarity(pose, f)
        assert s == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("pose", ALL_POSES)
    def test_classify_perfect_returns_pose(self, pose):
        f = features_for(pose)
        result_pose, conf = classify_pose(f)
        assert result_pose == pose
        assert conf >= NEUTRAL_THRESHOLD


# ============================================================
# 3. 다른 포즈에 대해 낮은 점수
# ============================================================
class TestCrossPoseScore:
    def test_t_features_score_y_pose_below_neutral(self):
        # T자 features는 Y자 NEUTRAL 임계 미만이어야 잘못 분류 X
        f = features_for(POSE_T)
        s = pose_similarity(POSE_Y, f)
        assert s < NEUTRAL_THRESHOLD

    def test_clap_features_score_t_pose_below_neutral(self):
        f = features_for(POSE_CLAP)
        s = pose_similarity(POSE_T, f)
        assert s < NEUTRAL_THRESHOLD

    def test_left_up_features_score_right_up_below_neutral(self):
        # 왼손위와 오른손위는 거울 대칭 — 서로의 점수가 NEUTRAL 미만
        f = features_for(POSE_LEFT_UP)
        s = pose_similarity(POSE_RIGHT_UP, f)
        assert s < NEUTRAL_THRESHOLD

    def test_t_features_classified_as_t_not_y(self):
        # 실전 의미: classify_pose에서 T로 분류되어야 함 (Y나 NEUTRAL X)
        f = features_for(POSE_T)
        pose, _conf = classify_pose(f)
        assert pose == POSE_T


# ============================================================
# 4. classify_pose 동작
# ============================================================
class TestClassify:
    def test_neutral_when_no_match(self):
        # 모든 feature가 어느 profile에도 안 맞는 값
        f = {
            "left_arm_angle":   60.0,
            "right_arm_angle":  60.0,
            "left_hand_y_off":  1.5,    # 손이 아래
            "right_hand_y_off": 1.5,
            "hand_distance":    1.5,    # 어중간
            "hand_y_avg":       1.5,
        }
        pose, conf = classify_pose(f)
        assert pose == POSE_NEUTRAL

    def test_returns_tuple(self):
        f = features_for(POSE_T)
        result = classify_pose(f)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_confidence_above_threshold_for_match(self):
        f = features_for(POSE_T)
        pose, conf = classify_pose(f)
        assert conf >= NEUTRAL_THRESHOLD
        assert pose == POSE_T


# ============================================================
# 5. 예외 / API
# ============================================================
class TestApi:
    def test_invalid_target_raises(self):
        with pytest.raises(ValueError):
            pose_similarity("knee_drop", {})

    def test_missing_feature_treated_as_zero(self):
        # T자 profile에 left_arm_angle만 주면 다른 feature 누락 → 0
        s = pose_similarity(POSE_T, {"left_arm_angle": 175.0})
        assert s == 0.0

    def test_all_poses_in_profiles(self):
        for p in ALL_POSES:
            assert p in POSE_PROFILES

    def test_game_poses_equals_all(self):
        assert GAME_POSES == ALL_POSES


# ============================================================
# 6. Boundary — 임계점에서 정확히
# ============================================================
class TestBoundary:
    def test_at_profile_edge_score_one(self):
        # T자 left_arm_angle 범위 (155, 180), 정확히 155에서 1.0
        f = features_for(POSE_T, {"left_arm_angle": 155.0})
        s = pose_similarity(POSE_T, f)
        # 다른 feature는 중심값이므로 1.0, 최소도 1.0
        assert s == pytest.approx(1.0)

    def test_just_outside_drops_below_one(self):
        # T자 left_arm_angle 범위에서 5도 벗어남 → 작은 감소
        f = features_for(POSE_T, {"left_arm_angle": 150.0})
        s = pose_similarity(POSE_T, f)
        # width 25, d 5 → 1 - 5/25 = 0.8
        assert 0.7 < s < 0.85
