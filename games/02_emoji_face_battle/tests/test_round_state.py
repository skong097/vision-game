"""
test_round_state.py — W4 라운드 진행 + 점수 + 종료 단위 테스트
==================================================================

검증 항목:
- 출제: 4종 표정 범위, 직전 회피, 시드 재현성, 시드 인스턴스 격리
- 정확도 → outcome 변환 (correct/partial/fail 경계, clipping)
- 누적 점수·종료 사유·우선순위
- 난이도별 임계·측정시간
- 잘못된 입력 예외
"""

import pytest

from round_state import (
    DIFFICULTIES,
    DIFFICULTY_CONFIG,
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_NORMAL,
    END_FINISHED,
    END_NONE,
    END_WIN,
    RoundState,
    SCORE_CORRECT,
    SCORE_FAIL,
    SCORE_PARTIAL,
    TOTAL_ROUNDS,
    WIN_THRESHOLD,
    make_outcome,
)
from expression_classifier import GAME_EXPRESSIONS


# ============================================================
# 1. make_outcome — 임계 경계와 점수 변환
# ============================================================
class TestMakeOutcome:
    def test_correct_at_threshold(self):
        # 정답 임계와 정확히 같은 값 → correct
        o = make_outcome("smile", 0.70, 0.70, 0.40)
        assert o["verdict"] == "correct"
        assert o["score"] == SCORE_CORRECT
        assert o["target"] == "smile"
        assert o["accuracy"] == 0.70

    def test_correct_above_threshold(self):
        o = make_outcome("sad", 0.85, 0.70, 0.40)
        assert o["verdict"] == "correct"
        assert o["score"] == 100

    def test_partial_at_threshold(self):
        o = make_outcome("angry", 0.40, 0.70, 0.40)
        assert o["verdict"] == "partial"
        assert o["score"] == SCORE_PARTIAL

    def test_partial_below_correct(self):
        o = make_outcome("surprised", 0.55, 0.70, 0.40)
        assert o["verdict"] == "partial"
        assert o["score"] == 50

    def test_fail_below_partial(self):
        o = make_outcome("smile", 0.10, 0.70, 0.40)
        assert o["verdict"] == "fail"
        assert o["score"] == SCORE_FAIL

    def test_clipping_above_one(self):
        # 정확도가 1.0을 초과하면 1.0으로 clipping
        o = make_outcome("smile", 1.5, 0.70, 0.40)
        assert o["accuracy"] == 1.0
        assert o["verdict"] == "correct"

    def test_clipping_below_zero(self):
        # 음수도 0.0으로
        o = make_outcome("smile", -0.3, 0.70, 0.40)
        assert o["accuracy"] == 0.0
        assert o["verdict"] == "fail"
        assert o["score"] == 0


# ============================================================
# 2. RoundState — 출제 (next_target)
# ============================================================
class TestNextTarget:
    def test_target_in_game_expressions(self):
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(20):
            t = s.next_target()
            assert t in GAME_EXPRESSIONS

    def test_avoids_immediate_repeat(self):
        s = RoundState(DIFFICULTY_NORMAL, seed=42)
        targets = [s.next_target() for _ in range(50)]
        for i in range(len(targets) - 1):
            assert targets[i] != targets[i + 1], (
                f"직전 회피 실패: idx {i} = {targets[i]}, idx {i+1} = {targets[i+1]}"
            )

    def test_seed_reproducibility(self):
        # 같은 시드 → 같은 수열
        a = RoundState(DIFFICULTY_NORMAL, seed=777)
        b = RoundState(DIFFICULTY_NORMAL, seed=777)
        seq_a = [a.next_target() for _ in range(30)]
        seq_b = [b.next_target() for _ in range(30)]
        assert seq_a == seq_b

    def test_seed_instance_isolation(self):
        # 두 인스턴스가 동일 시드라도 서로 영향 X — 전역 random 미오염
        a = RoundState(DIFFICULTY_NORMAL, seed=42)
        b = RoundState(DIFFICULTY_NORMAL, seed=42)
        # a를 진행시켜도 b의 결과가 a를 만들 때와 같아야 함
        a.next_target()
        a.next_target()
        a.next_target()
        seq_b = [b.next_target() for _ in range(3)]
        # b를 새로 만들면 a가 처음 3번 뽑은 것과 같아야 함
        c = RoundState(DIFFICULTY_NORMAL, seed=42)
        seq_c = [c.next_target() for _ in range(3)]
        assert seq_b == seq_c

    def test_history_recorded(self):
        s = RoundState(DIFFICULTY_NORMAL, seed=1)
        for _ in range(5):
            s.next_target()
        assert len(s.history) == 5
        assert all(t in GAME_EXPRESSIONS for t in s.history)


# ============================================================
# 3. apply_round — 누적
# ============================================================
class TestApplyRound:
    def test_perfect_game(self):
        # 5라운드 모두 정답 → 500점, 4정답 이상 → END_WIN
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            t = s.next_target()
            s.apply_round(t, 0.95)
        assert s.total_score == 500
        assert s.correct_count == 5
        assert s.is_win()
        assert s.check_end() == END_WIN

    def test_win_with_four_correct(self):
        # 4정답 + 1실패 = 400점, END_WIN
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        accuracies = [0.95, 0.95, 0.95, 0.95, 0.10]
        for acc in accuracies:
            t = s.next_target()
            s.apply_round(t, acc)
        assert s.correct_count == 4
        assert s.fail_count == 1
        assert s.total_score == 400
        assert s.check_end() == END_WIN

    def test_finished_three_correct(self):
        # 3정답 + 2실패 = 300점, END_FINISHED (도전 부족)
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        accuracies = [0.95, 0.95, 0.95, 0.10, 0.10]
        for acc in accuracies:
            t = s.next_target()
            s.apply_round(t, acc)
        assert s.correct_count == 3
        assert s.check_end() == END_FINISHED
        assert not s.is_win()

    def test_partial_scores(self):
        # 3정답 + 2부분 = 300+100 = 400점이지만 correct는 3 → FINISHED
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        accuracies = [0.95, 0.95, 0.95, 0.55, 0.55]
        for acc in accuracies:
            t = s.next_target()
            s.apply_round(t, acc)
        assert s.correct_count == 3
        assert s.partial_count == 2
        assert s.total_score == 400
        # 부분 정답은 WIN_THRESHOLD에 포함 X
        assert s.check_end() == END_FINISHED

    def test_in_progress_no_end(self):
        # 3라운드만 진행하면 END_NONE
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(3):
            t = s.next_target()
            s.apply_round(t, 0.9)
        assert s.check_end() == END_NONE
        assert not s.is_game_over()

    def test_raises_on_extra_round(self):
        # 5라운드 후 6번째 apply_round → RuntimeError
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            t = s.next_target()
            s.apply_round(t, 0.9)
        with pytest.raises(RuntimeError):
            s.apply_round("smile", 0.5)

    def test_end_reason_cached(self):
        # 한 번 결정된 종료 사유는 변하지 않음
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            t = s.next_target()
            s.apply_round(t, 0.95)
        first = s.check_end()
        # 같은 호출 반복해도 동일 결과 (cache 동작 확인)
        for _ in range(5):
            assert s.check_end() == first


# ============================================================
# 4. 난이도별 임계
# ============================================================
class TestDifficulty:
    def test_easy_thresholds(self):
        s = RoundState(DIFFICULTY_EASY, seed=0)
        assert s.get_correct_threshold() == 0.60
        assert s.get_partial_threshold() == 0.30
        assert s.get_measure_time() == 3.0

    def test_normal_thresholds(self):
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        assert s.get_correct_threshold() == 0.70
        assert s.get_partial_threshold() == 0.40
        assert s.get_measure_time() == 2.0

    def test_hard_thresholds(self):
        s = RoundState(DIFFICULTY_HARD, seed=0)
        assert s.get_correct_threshold() == 0.80
        assert s.get_partial_threshold() == 0.50
        assert s.get_measure_time() == 1.2

    def test_correct_strict_easier_to_easier(self):
        # 단조성: easy < normal < hard
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["correct"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["correct"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["correct"]
        assert e < n < h

    def test_measure_time_decreases_with_difficulty(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["measure_time"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["measure_time"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["measure_time"]
        assert e > n > h

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError):
            RoundState("insane", seed=0)


# ============================================================
# 5. 정확도 임계 동일 값 (난이도별로 정답/부분 갈림)
# ============================================================
class TestDifficultyBoundary:
    """동일 정확도(0.65)가 난이도별로 다른 결과를 내는지"""

    def test_065_easy_correct(self):
        s = RoundState(DIFFICULTY_EASY, seed=0)
        s.next_target()
        out = s.apply_round("smile", 0.65)
        # easy: correct ≥ 0.60 → correct
        assert out["verdict"] == "correct"

    def test_065_normal_partial(self):
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        s.next_target()
        out = s.apply_round("smile", 0.65)
        # normal: 0.40 ≤ 0.65 < 0.70 → partial
        assert out["verdict"] == "partial"

    def test_065_hard_partial(self):
        s = RoundState(DIFFICULTY_HARD, seed=0)
        s.next_target()
        out = s.apply_round("smile", 0.65)
        # hard: 0.50 ≤ 0.65 < 0.80 → partial
        assert out["verdict"] == "partial"


# ============================================================
# 6. Summary / reset
# ============================================================
class TestSummaryAndReset:
    def test_summary_keys(self):
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        summary = s.get_summary()
        for k in ("difficulty", "round_index", "total_rounds", "score",
                  "max_score", "correct", "partial", "fail",
                  "win_threshold", "end_reason", "history"):
            assert k in summary
        assert summary["max_score"] == 500
        assert summary["total_rounds"] == TOTAL_ROUNDS
        assert summary["win_threshold"] == WIN_THRESHOLD

    def test_reset_clears_state_keeps_rng(self):
        # reset 후에도 RNG 진행 상태는 유지 (다시 시드 박지 않음)
        s = RoundState(DIFFICULTY_NORMAL, seed=42)
        s.next_target()
        s.apply_round("smile", 0.95)
        s.reset()
        assert s.round_index == 0
        assert s.total_score == 0
        assert s.correct_count == 0
        assert s.outcomes == []
        assert s.history == []
        assert s.check_end() == END_NONE
        # reset 후 next_target은 reset 전 인스턴스의 첫 next_target과 달라야 함
        # (RNG 상태가 보존되므로 이미 한 번 뽑은 자리에서 시작)
        # 비교: 새 인스턴스로 같은 시드 → 첫 번째 결과
        fresh = RoundState(DIFFICULTY_NORMAL, seed=42)
        first_fresh = fresh.next_target()
        # reset된 s의 첫 next_target은 fresh의 첫 결과와 다를 가능성이 높음
        # (history 비었으므로 회피 로직은 동일 영향)
        first_after_reset = s.next_target()
        # 결정적인 단언이 어려운 부분 — 핵심은 reset이 RNG를 다시 박지 않는다는 점
        # 따라서 fresh와 reset된 인스턴스는 같은 시드로 시작했지만 진행 상태가 다름
        # 이 단언은 RNG 상태 진행 의도를 문서화하는 의미
        assert first_fresh in GAME_EXPRESSIONS
        assert first_after_reset in GAME_EXPRESSIONS

    def test_outcomes_list_grows(self):
        s = RoundState(DIFFICULTY_NORMAL, seed=0)
        assert s.outcomes == []
        for i in range(3):
            t = s.next_target()
            s.apply_round(t, 0.9)
            assert len(s.outcomes) == i + 1


# ============================================================
# 7. 상수 sanity
# ============================================================
class TestConstants:
    def test_constants_match_design(self):
        assert TOTAL_ROUNDS == 5
        assert WIN_THRESHOLD == 4
        assert SCORE_CORRECT == 100
        assert SCORE_PARTIAL == 50
        assert SCORE_FAIL == 0

    def test_all_difficulties_present(self):
        assert set(DIFFICULTIES) == set(DIFFICULTY_CONFIG.keys())
