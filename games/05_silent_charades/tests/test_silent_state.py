"""
test_silent_state.py — W9 라운드 상태 단위 테스트
========================================================

검증 (W4·W6 패턴):
- make_outcome (correct/partial/fail 경계)
- next_word 출제
- apply_round 누적, current_word 필요
- 5라운드 후 추가 호출 거부
- 종료 사유 (WIN > FINISHED), cache
- 난이도별 임계·시간
- api_used_count 누적
"""

import pytest

from silent_words import WORD_CATALOG
from silent_state import (
    DIFFICULTIES,
    DIFFICULTY_CONFIG,
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_NORMAL,
    END_FINISHED,
    END_NONE,
    END_WIN,
    SCORE_CORRECT,
    SCORE_FAIL,
    SCORE_PARTIAL,
    SilentState,
    TOTAL_ROUNDS,
    WIN_THRESHOLD,
    make_outcome,
)


# ============================================================
# 1. make_outcome
# ============================================================
class TestMakeOutcome:
    def test_correct_at_threshold(self):
        o = make_outcome("강아지", 70, "good", 70, 40, True)
        assert o["verdict"] == "correct"
        assert o["round_score"] == SCORE_CORRECT

    def test_partial_at_threshold(self):
        o = make_outcome("비행기", 40, "ok", 70, 40, True)
        assert o["verdict"] == "partial"
        assert o["round_score"] == SCORE_PARTIAL

    def test_fail_below_partial(self):
        o = make_outcome("박수", 10, "no", 70, 40, True)
        assert o["verdict"] == "fail"
        assert o["round_score"] == SCORE_FAIL

    def test_clipping_above_100(self):
        o = make_outcome("강아지", 150, "x", 70, 40, True)
        assert o["llm_score"] == 100
        assert o["verdict"] == "correct"

    def test_clipping_below_zero(self):
        o = make_outcome("강아지", -10, "x", 70, 40, True)
        assert o["llm_score"] == 0
        assert o["verdict"] == "fail"

    def test_carries_word_and_comment(self):
        o = make_outcome("노래", 75, "marvelous", 70, 40, True)
        assert o["word"] == "노래"
        assert o["comment"] == "marvelous"
        assert o["used_api"] is True

    def test_keeps_used_api_false(self):
        o = make_outcome("박수", 60, "fallback", 70, 40, False)
        assert o["used_api"] is False


# ============================================================
# 2. next_word
# ============================================================
class TestNextWord:
    def test_sets_current_word(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        w = s.next_word()
        assert w is not None
        assert s.current_word is w

    def test_avoids_immediate_repeat(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=42)
        slugs = [s.next_word().slug for _ in range(40)]
        for i in range(len(slugs) - 1):
            assert slugs[i] != slugs[i + 1]


# ============================================================
# 3. apply_round
# ============================================================
class TestApplyRound:
    def test_requires_current_word(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        with pytest.raises(RuntimeError):
            s.apply_round(80, "good")

    def test_perfect_game(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            s.next_word()
            s.apply_round(95, "great")
        assert s.total_score == 500
        assert s.correct_count == 5
        assert s.is_win()

    def test_four_correct_wins(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        scores = [95, 95, 95, 95, 10]
        for sc in scores:
            s.next_word()
            s.apply_round(sc, "x")
        assert s.correct_count == 4
        assert s.check_end() == END_WIN

    def test_three_correct_finished(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        scores = [95, 95, 95, 10, 10]
        for sc in scores:
            s.next_word()
            s.apply_round(sc, "x")
        assert s.correct_count == 3
        assert s.check_end() == END_FINISHED

    def test_partial_count(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        scores = [95, 95, 95, 55, 55]
        for sc in scores:
            s.next_word()
            s.apply_round(sc, "x")
        assert s.correct_count == 3
        assert s.partial_count == 2
        assert s.total_score == 400
        assert s.check_end() == END_FINISHED

    def test_raises_on_extra_round(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            s.next_word()
            s.apply_round(90, "x")
        s.next_word()  # 6번째 word는 가져올 수 있어도
        with pytest.raises(RuntimeError):
            s.apply_round(90, "x")

    def test_end_reason_cached(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            s.next_word()
            s.apply_round(95, "x")
        first = s.check_end()
        for _ in range(5):
            assert s.check_end() == first

    def test_api_used_count(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        s.next_word()
        s.apply_round(80, "x", used_api=True)
        s.next_word()
        s.apply_round(80, "x", used_api=False)
        s.next_word()
        s.apply_round(80, "x", used_api=True)
        assert s.api_used_count == 2


# ============================================================
# 4. 난이도
# ============================================================
class TestDifficulty:
    def test_easy(self):
        s = SilentState(DIFFICULTY_EASY, seed=0)
        assert s.get_express_time() == 20.0
        assert s.get_correct_threshold() == 60
        assert s.get_partial_threshold() == 30

    def test_normal(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        assert s.get_express_time() == 15.0
        assert s.get_correct_threshold() == 70

    def test_hard(self):
        s = SilentState(DIFFICULTY_HARD, seed=0)
        assert s.get_express_time() == 10.0
        assert s.get_correct_threshold() == 80

    def test_monotonic_thresholds(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["correct"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["correct"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["correct"]
        assert e < n < h

    def test_monotonic_time(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["express_time"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["express_time"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["express_time"]
        assert e > n > h

    def test_invalid_difficulty(self):
        with pytest.raises(ValueError):
            SilentState("expert", seed=0)


# ============================================================
# 5. 동일 점수 난이도별 결과
# ============================================================
class TestDifficultyBoundary:
    def test_65_easy_correct(self):
        s = SilentState(DIFFICULTY_EASY, seed=0)
        s.next_word()
        o = s.apply_round(65, "x")
        assert o["verdict"] == "correct"

    def test_65_normal_partial(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        s.next_word()
        o = s.apply_round(65, "x")
        assert o["verdict"] == "partial"

    def test_65_hard_partial(self):
        s = SilentState(DIFFICULTY_HARD, seed=0)
        s.next_word()
        o = s.apply_round(65, "x")
        assert o["verdict"] == "partial"


# ============================================================
# 6. Summary / Reset
# ============================================================
class TestSummaryReset:
    def test_summary_keys(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        for k in ("difficulty", "round_index", "total_rounds", "score",
                  "max_score", "correct", "partial", "fail",
                  "win_threshold", "api_used", "end_reason",
                  "current_word_ko"):
            assert k in s.get_summary()

    def test_max_score(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=0)
        assert s.get_max_score() == 500

    def test_reset(self):
        s = SilentState(DIFFICULTY_NORMAL, seed=42)
        s.next_word()
        s.apply_round(80, "x")
        s.reset()
        assert s.round_index == 0
        assert s.total_score == 0
        assert s.outcomes == []
        assert s.current_word is None
        assert s.check_end() == END_NONE


# ============================================================
# 7. 상수
# ============================================================
class TestConstants:
    def test_constants(self):
        assert TOTAL_ROUNDS == 5
        assert WIN_THRESHOLD == 4
        assert SCORE_CORRECT == 100
