"""
test_sync_state.py — 미션 회전·유지 시간·점수·종료 단위 테스트
======================================================================

검증:
- 미션 회전 (직전 회피, 시드 격리)
- apply_sync: 누적 → 임계 시 완료 → 다음 미션
- 매칭 끊김 시 hold_time 리셋
- 보너스 점수
- 종료 사유 우선순위 (WIN > TIMEOUT)
- time_provider 주입
- 잘못된 입력
"""

import pytest

from pose_classifier import GAME_POSES, POSE_T
from sync_judge import SyncResult
from sync_state import (
    DIFFICULTIES,
    DIFFICULTY_CONFIG,
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_NORMAL,
    END_NONE,
    END_TIMEOUT,
    END_WIN,
    SCORE_BONUS,
    SCORE_PER_COMPLETION,
    SyncState,
    TIME_LIMIT,
)


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def match_result(both=True, bonus=False, pair=1.0):
    """SyncResult 만들기 헬퍼."""
    return SyncResult(
        both_match=both,
        pair_score=pair,
        left_score=pair,
        right_score=pair,
        bonus=bonus,
    )


# ============================================================
# 1. 미션 회전
# ============================================================
class TestNextMission:
    def test_first_mission_in_game_poses(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        m = s.next_mission()
        assert m in GAME_POSES
        assert s.current_mission == m

    def test_avoids_immediate_repeat(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=42,
                      time_provider=FakeClock(0))
        targets = [s.next_mission() for _ in range(50)]
        for i in range(len(targets) - 1):
            assert targets[i] != targets[i + 1]

    def test_seed_reproducibility(self):
        a = SyncState(DIFFICULTY_NORMAL, seed=777,
                      time_provider=FakeClock(0))
        b = SyncState(DIFFICULTY_NORMAL, seed=777,
                      time_provider=FakeClock(0))
        seq_a = [a.next_mission() for _ in range(20)]
        seq_b = [b.next_mission() for _ in range(20)]
        assert seq_a == seq_b

    def test_history_recorded(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        for _ in range(5):
            s.next_mission()
        assert len(s.history) == 5

    def test_next_mission_resets_hold(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        s.next_mission()
        s.hold_time = 0.5
        s.next_mission()
        assert s.hold_time == 0.0


# ============================================================
# 2. start_game / 첫 미션 자동 출제
# ============================================================
class TestStart:
    def test_start_creates_mission(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        s.start_game()
        assert s.current_mission is not None
        assert s.current_mission in GAME_POSES

    def test_start_no_double_mission(self):
        # 이미 next_mission 호출된 후 start_game → 새 미션 생성 X
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        m = s.next_mission()
        s.start_game()
        assert s.current_mission == m


# ============================================================
# 3. apply_sync — 유지·완료
# ============================================================
class TestApplySync:
    def test_hold_accumulates_when_match(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        # match=True 누적
        s.apply_sync(0.3, match_result(both=True))
        assert s.hold_time == pytest.approx(0.3)
        s.apply_sync(0.3, match_result(both=True))
        assert s.hold_time == pytest.approx(0.6)

    def test_completion_at_threshold(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        # NORMAL hold_required = 0.7s
        completed = s.apply_sync(0.7, match_result(both=True))
        assert completed is True
        assert s.completions == 1
        assert s.score == SCORE_PER_COMPLETION
        # hold reset, 다음 미션
        assert s.hold_time == 0.0

    def test_no_match_resets_hold(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_sync(0.5, match_result(both=True))
        assert s.hold_time > 0
        s.apply_sync(0.1, match_result(both=False))
        assert s.hold_time == 0.0
        # 완료는 안 됨
        assert s.completions == 0

    def test_bonus_added(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_sync(0.7, match_result(both=True, bonus=True))
        assert s.completions == 1
        assert s.bonus_count == 1
        assert s.score == SCORE_PER_COMPLETION + SCORE_BONUS

    def test_no_bonus_in_score_when_not_match(self):
        # both_match=False이면 bonus 플래그 있어도 카운트 X
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_sync(0.7, match_result(both=False, bonus=True))
        assert s.score == 0
        assert s.bonus_count == 0

    def test_no_apply_before_start(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        completed = s.apply_sync(0.7, match_result(both=True))
        assert completed is False
        assert s.score == 0

    def test_no_apply_after_end(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_EASY, seed=0, time_provider=clock)
        s.start_game()
        # easy target 80, 1 완료 = 20점. 4번 완료 시 80점
        for _ in range(4):
            s.apply_sync(0.5, match_result(both=True))
        assert s.check_end() == END_WIN
        # 이후 호출 무시
        before = s.score
        s.apply_sync(0.5, match_result(both=True))
        assert s.score == before

    def test_completion_picks_new_mission(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        first_mission = s.current_mission
        s.apply_sync(0.7, match_result(both=True))
        # 미션이 바뀜 (직전 회피)
        assert s.current_mission != first_mission


# ============================================================
# 4. 종료 사유
# ============================================================
class TestEndConditions:
    def test_win_when_score_reached(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_EASY, seed=0, time_provider=clock)
        s.start_game()
        # easy target 80 → 4 완료
        for _ in range(4):
            s.apply_sync(0.5, match_result(both=True))
        assert s.check_end() == END_WIN
        assert s.is_win()

    def test_timeout_when_time_runs_out(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        clock.t = TIME_LIMIT + 1
        assert s.check_end() == END_TIMEOUT
        assert not s.is_win()

    def test_no_end_before_start(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        assert s.check_end() == END_NONE

    def test_no_end_in_progress(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_sync(0.7, match_result(both=True))  # 20점
        clock.t = 30
        assert s.check_end() == END_NONE

    def test_win_priority_at_timeout(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_EASY, seed=0, time_provider=clock)
        s.start_game()
        clock.t = TIME_LIMIT  # 시간 만료
        # 점수 달성
        for _ in range(4):
            s.apply_sync(0.5, match_result(both=True))
        assert s.check_end() == END_WIN

    def test_end_reason_cached(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        clock.t = TIME_LIMIT + 1
        first = s.check_end()
        for _ in range(5):
            assert s.check_end() == first


# ============================================================
# 5. 시간
# ============================================================
class TestTime:
    def test_remaining_clamped(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        clock.t = TIME_LIMIT + 10
        assert s.get_remaining() == 0.0

    def test_elapsed_zero_before_start(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        assert s.get_elapsed() == 0.0

    def test_hold_progress(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=0, time_provider=clock)
        s.start_game()
        s.apply_sync(0.35, match_result(both=True))  # 절반
        # NORMAL hold_required = 0.7, 0.35/0.7 = 0.5
        assert s.get_hold_progress() == pytest.approx(0.5)


# ============================================================
# 6. 난이도 / 상수
# ============================================================
class TestDifficulty:
    def test_easy(self):
        s = SyncState(DIFFICULTY_EASY, seed=0, time_provider=FakeClock(0))
        assert s.get_match_threshold() == 0.55
        assert s.get_hold_required() == 0.5
        assert s.target_score == 80

    def test_normal(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        assert s.get_match_threshold() == 0.65
        assert s.get_hold_required() == 0.7
        assert s.target_score == 120

    def test_hard(self):
        s = SyncState(DIFFICULTY_HARD, seed=0, time_provider=FakeClock(0))
        assert s.get_match_threshold() == 0.75
        assert s.get_hold_required() == 1.0
        assert s.target_score == 200

    def test_monotonic_threshold(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["match_threshold"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["match_threshold"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["match_threshold"]
        assert e < n < h

    def test_monotonic_hold(self):
        e = DIFFICULTY_CONFIG[DIFFICULTY_EASY]["hold_required"]
        n = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["hold_required"]
        h = DIFFICULTY_CONFIG[DIFFICULTY_HARD]["hold_required"]
        assert e < n < h

    def test_invalid_difficulty(self):
        with pytest.raises(ValueError):
            SyncState("nightmare", seed=0,
                      time_provider=FakeClock(0))


# ============================================================
# 7. Summary / reset
# ============================================================
class TestSummaryReset:
    def test_summary_keys(self):
        s = SyncState(DIFFICULTY_NORMAL, seed=0,
                      time_provider=FakeClock(0))
        summary = s.get_summary()
        for k in ("difficulty", "score", "target_score", "progress",
                  "completions", "bonus_count", "current_mission",
                  "hold_time", "elapsed", "remaining", "end_reason",
                  "history"):
            assert k in summary

    def test_reset(self):
        clock = FakeClock(0)
        s = SyncState(DIFFICULTY_NORMAL, seed=42, time_provider=clock)
        s.start_game()
        s.apply_sync(0.7, match_result(both=True))
        s.reset()
        assert s.score == 0
        assert s.completions == 0
        assert s.current_mission is None
        assert s.start_time is None
        assert s.history == []
