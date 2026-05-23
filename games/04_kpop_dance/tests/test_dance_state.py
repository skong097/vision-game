"""
test_dance_state.py — W6 댄스 라운드 단위 테스트
=======================================================

검증 (W4 round_state 패턴 + W6 pose 적용):
- 5포즈 풀, 직전 회피
- 시드 재현성 + 인스턴스 격리
- correct/partial/fail 경계
- 종료 사유 우선순위, end_reason 캐시
- 난이도별 임계·측정 시간
- 잘못된 입력 예외
"""

import pytest

from dance_state import (
    DIFFICULTIES,
    DIFFICULTY_CONFIG,
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_NORMAL,
    DanceState,
    END_FINISHED,
    END_NONE,
    END_WIN,
    SCORE_CORRECT,
    SCORE_FAIL,
    SCORE_PARTIAL,
    TOTAL_ROUNDS,
    WIN_THRESHOLD,
    make_outcome,
)
from pose_classifier import GAME_POSES


# ============================================================
# 1. make_outcome
# ============================================================
class TestMakeOutcome:
    def test_correct_at_threshold(self):
        o = make_outcome("t_pose", 0.70, 0.70, 0.40)
        assert o["verdict"] == "correct"
        assert o["score"] == SCORE_CORRECT

    def test_partial_at_threshold(self):
        o = make_outcome("y_pose", 0.40, 0.70, 0.40)
        assert o["verdict"] == "partial"
        assert o["score"] == SCORE_PARTIAL

    def test_fail_below_partial(self):
        o = make_outcome("clap", 0.10, 0.70, 0.40)
        assert o["verdict"] == "fail"
        assert o["score"] == SCORE_FAIL

    def test_clipping(self):
        o_above = make_outcome("t_pose", 1.5, 0.70, 0.40)
        assert o_above["accuracy"] == 1.0
        o_below = make_outcome("t_pose", -0.3, 0.70, 0.40)
        assert o_below["accuracy"] == 0.0


# ============================================================
# 2. next_target
# ============================================================
class TestNextTarget:
    def test_target_in_game_poses(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(20):
            assert s.next_target() in GAME_POSES

    def test_avoids_immediate_repeat(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=42)
        targets = [s.next_target() for _ in range(50)]
        for i in range(len(targets) - 1):
            assert targets[i] != targets[i + 1]

    def test_seed_reproducibility(self):
        a = DanceState(DIFFICULTY_NORMAL, seed=777)
        b = DanceState(DIFFICULTY_NORMAL, seed=777)
        seq_a = [a.next_target() for _ in range(30)]
        seq_b = [b.next_target() for _ in range(30)]
        assert seq_a == seq_b

    def test_seed_isolation(self):
        a = DanceState(DIFFICULTY_NORMAL, seed=42)
        b = DanceState(DIFFICULTY_NORMAL, seed=42)
        a.next_target()
        a.next_target()
        seq_b = [b.next_target() for _ in range(3)]
        c = DanceState(DIFFICULTY_NORMAL, seed=42)
        seq_c = [c.next_target() for _ in range(3)]
        assert seq_b == seq_c

    def test_history_recorded(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=1)
        for _ in range(5):
            s.next_target()
        assert len(s.history) == 5


# ============================================================
# 3. apply_round
# ============================================================
class TestApplyRound:
    def test_perfect_game_wins(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            t = s.next_target()
            s.apply_round(t, 0.95)
        assert s.total_score == 500
        assert s.correct_count == 5
        assert s.is_win()
        assert s.check_end() == END_WIN

    def test_four_correct_wins(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        for acc in [0.95, 0.95, 0.95, 0.95, 0.10]:
            t = s.next_target()
            s.apply_round(t, acc)
        assert s.correct_count == 4
        assert s.check_end() == END_WIN

    def test_three_correct_finished(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        for acc in [0.95, 0.95, 0.95, 0.10, 0.10]:
            t = s.next_target()
            s.apply_round(t, acc)
        assert s.correct_count == 3
        assert s.check_end() == END_FINISHED
        assert not s.is_win()

    def test_partial_doesnt_count_for_win(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        for acc in [0.95, 0.95, 0.95, 0.55, 0.55]:
            t = s.next_target()
            s.apply_round(t, acc)
        assert s.correct_count == 3
        assert s.partial_count == 2
        assert s.total_score == 400
        assert s.check_end() == END_FINISHED

    def test_in_progress_no_end(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(3):
            t = s.next_target()
            s.apply_round(t, 0.9)
        assert s.check_end() == END_NONE

    def test_raises_on_extra_round(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            t = s.next_target()
            s.apply_round(t, 0.9)
        with pytest.raises(RuntimeError):
            s.apply_round("t_pose", 0.5)

    def test_end_reason_cached(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            t = s.next_target()
            s.apply_round(t, 0.95)
        first = s.check_end()
        for _ in range(5):
            assert s.check_end() == first


# ============================================================
# 4. 난이도
# ============================================================
class TestDifficulty:
    def test_easy(self):
        s = DanceState(DIFFICULTY_EASY, seed=0)
        assert s.get_measure_time() == 3.5
        assert s.get_correct_threshold() == 0.60
        assert s.get_partial_threshold() == 0.30

    def test_normal(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        assert s.get_measure_time() == 2.5
        assert s.get_correct_threshold() == 0.70

    def test_hard(self):
        s = DanceState(DIFFICULTY_HARD, seed=0)
        assert s.get_measure_time() == 1.5
        assert s.get_correct_threshold() == 0.80

    def test_monotonic_thresholds(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["correct"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["correct"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["correct"]
        assert e < n < h

    def test_monotonic_measure_time(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["measure_time"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["measure_time"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["measure_time"]
        assert e > n > h

    def test_invalid_difficulty(self):
        with pytest.raises(ValueError):
            DanceState("god_tier", seed=0)


# ============================================================
# 5. 동일 정확도가 난이도별로 다른 결과
# ============================================================
class TestDifficultyBoundary:
    def test_065_easy_correct(self):
        s = DanceState(DIFFICULTY_EASY, seed=0)
        s.next_target()
        o = s.apply_round("t_pose", 0.65)
        assert o["verdict"] == "correct"

    def test_065_normal_partial(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        s.next_target()
        o = s.apply_round("t_pose", 0.65)
        assert o["verdict"] == "partial"

    def test_065_hard_partial(self):
        s = DanceState(DIFFICULTY_HARD, seed=0)
        s.next_target()
        o = s.apply_round("t_pose", 0.65)
        assert o["verdict"] == "partial"


# ============================================================
# 6. Summary / reset / constants
# ============================================================
class TestMisc:
    def test_summary_keys(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=0)
        summary = s.get_summary()
        for k in ("difficulty", "round_index", "total_rounds", "score",
                  "max_score", "correct", "partial", "fail",
                  "win_threshold", "end_reason", "history"):
            assert k in summary
        assert summary["max_score"] == 500

    def test_reset(self):
        s = DanceState(DIFFICULTY_NORMAL, seed=42)
        s.next_target()
        s.apply_round("t_pose", 0.95)
        s.reset()
        assert s.round_index == 0
        assert s.total_score == 0
        assert s.outcomes == []
        assert s.history == []

    def test_constants(self):
        assert TOTAL_ROUNDS == 5
        assert WIN_THRESHOLD == 4
        assert SCORE_CORRECT == 100
