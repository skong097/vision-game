"""
test_treasure_state.py — 60s / 발견 카운트 / 유지 시간 단위 테스트
"""

import pytest

from treasure_state import (
    DIFFICULTIES,
    DIFFICULTY_CONFIG,
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_NORMAL,
    END_NONE,
    END_TIMEOUT,
    END_WIN,
    SCORE_PER_FIND,
    TIME_BONUS_PER_SEC,
    TIME_LIMIT,
    TreasureState,
)


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


# ============================================================
# 1. start / next_target
# ============================================================
class TestStartTarget:
    def test_start_creates_target(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        assert s.current_target is not None
        assert s.current_target.yolo_class in DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["pool"]

    def test_no_target_before_start(self):
        s = TreasureState(DIFFICULTY_NORMAL, seed=0,
                          time_provider=FakeClock(0))
        assert s.current_target is None


# ============================================================
# 2. apply_detection — hold time 누적
# ============================================================
class TestApplyDetection:
    def test_no_apply_before_start(self):
        s = TreasureState(DIFFICULTY_NORMAL, seed=0,
                          time_provider=FakeClock(0))
        found = s.apply_detection(in_roi=True, dt=0.5)
        assert found is False

    def test_hold_accumulates_in_roi(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        # NORMAL hold_required = 1.0s
        assert s.apply_detection(True, 0.4) is False
        assert s.hold_time == pytest.approx(0.4)
        assert s.apply_detection(True, 0.5) is False
        assert s.hold_time == pytest.approx(0.9)

    def test_find_at_hold_threshold(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        found = s.apply_detection(True, 1.0)
        assert found is True
        assert s.found_count == 1
        assert s.score == SCORE_PER_FIND

    def test_out_of_roi_resets_hold(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_detection(True, 0.5)
        assert s.hold_time > 0
        s.apply_detection(False, 0.1)
        assert s.hold_time == 0.0

    def test_next_target_after_find(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        first = s.current_target.yolo_class
        s.apply_detection(True, 1.0)
        # 발견 후 새 보물 (직전 회피)
        assert s.current_target.yolo_class != first

    def test_history_recorded(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_EASY, seed=0, time_provider=clock)
        s.start_game()
        # easy goal=3, hold=0.7s
        for _ in range(3):
            s.apply_detection(True, 0.7)
        assert len(s.found_history) == 3

    def test_attempts_counts_in_roi(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_detection(True, 0.3)
        s.apply_detection(True, 0.3)
        s.apply_detection(False, 0.3)
        # in_roi 호출 2번
        assert s.attempts == 2


# ============================================================
# 3. 종료 조건
# ============================================================
class TestEndConditions:
    def test_win_at_goal(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_EASY, seed=0, time_provider=clock)
        s.start_game()
        # easy goal=3, hold=0.7
        for _ in range(3):
            s.apply_detection(True, 0.7)
        assert s.check_end() == END_WIN
        assert s.is_win()

    def test_time_bonus_added(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_EASY, seed=0, time_provider=clock)
        s.start_game()
        # 빠르게 발견 → 잔여 시간 많음
        for _ in range(3):
            s.apply_detection(True, 0.7)
        # 3 × 20 = 60점 + 시간 보너스
        # 경과 시간 ~2.1s, 잔여 ~57.9s → 보너스 ~57
        summary = s.get_summary()
        assert summary["score"] > 60  # 보너스 포함
        assert summary["time_bonus"] > 0

    def test_timeout(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        clock.t = TIME_LIMIT + 1
        assert s.check_end() == END_TIMEOUT
        assert not s.is_win()

    def test_no_end_in_progress(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_detection(True, 1.0)
        clock.t = 20
        assert s.check_end() == END_NONE

    def test_end_reason_cached(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        clock.t = TIME_LIMIT + 1
        first = s.check_end()
        for _ in range(5):
            assert s.check_end() == first

    def test_no_apply_after_end(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_EASY, seed=0, time_provider=clock)
        s.start_game()
        for _ in range(3):
            s.apply_detection(True, 0.7)
        assert s.is_win()
        before = s.score
        s.apply_detection(True, 0.7)
        assert s.score == before  # 무시됨


# ============================================================
# 4. 시간 / hold progress
# ============================================================
class TestTime:
    def test_remaining_clamped(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        clock.t = TIME_LIMIT + 10
        assert s.get_remaining() == 0.0

    def test_hold_progress(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_detection(True, 0.5)
        # NORMAL hold=1.0, 0.5/1.0=0.5
        assert s.get_hold_progress() == pytest.approx(0.5)


# ============================================================
# 5. 난이도
# ============================================================
class TestDifficulty:
    def test_easy(self):
        s = TreasureState(DIFFICULTY_EASY, seed=0, time_provider=FakeClock(0))
        assert s.get_goal() == 3
        assert s.get_hold_required() == 0.7

    def test_normal(self):
        s = TreasureState(DIFFICULTY_NORMAL, seed=0,
                          time_provider=FakeClock(0))
        assert s.get_goal() == 5
        assert s.get_hold_required() == 1.0

    def test_hard(self):
        s = TreasureState(DIFFICULTY_HARD, seed=0, time_provider=FakeClock(0))
        assert s.get_goal() == 7
        assert s.get_hold_required() == 1.5

    def test_monotonic_goal(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["goal"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["goal"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["goal"]
        assert e < n < h

    def test_monotonic_hold(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["hold_required"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["hold_required"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["hold_required"]
        assert e < n < h

    def test_monotonic_confidence(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["min_confidence"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["min_confidence"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["min_confidence"]
        assert e < n < h

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            TreasureState("godmode", time_provider=FakeClock(0))


# ============================================================
# 6. Summary / Reset
# ============================================================
class TestSummaryReset:
    def test_summary_keys(self):
        s = TreasureState(DIFFICULTY_NORMAL, seed=0,
                          time_provider=FakeClock(0))
        for k in ("difficulty", "score", "found_count", "goal",
                  "elapsed", "remaining", "attempts", "end_reason",
                  "time_bonus", "current_target_ko",
                  "current_target_class", "found_history"):
            assert k in s.get_summary()

    def test_reset(self):
        clock = FakeClock(0)
        s = TreasureState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_detection(True, 0.5)
        s.reset()
        assert s.score == 0
        assert s.found_count == 0
        assert s.hold_time == 0.0
        assert s.start_time is None
        assert s.current_target is None
        assert s.check_end() == END_NONE
