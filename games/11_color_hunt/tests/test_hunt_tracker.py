"""
test_hunt_tracker.py — 라운드 상태 단위 테스트
======================================================

검증:
- apply_matches로 collected 누적
- 색/신뢰도/dedup 거부 분류
- dedup window: 같은 객체 5초 이내 재검출은 카운트 X
- 다른 위치/다른 class는 dedup 적용 X
- 종료 사유: WIN(collected≥goal) vs TIMEOUT(remaining≤0)
- 종료 우선순위 (WIN > TIMEOUT)
- time_provider 주입으로 시간 제어
- reset / summary
"""

from dataclasses import dataclass

import pytest

from color_classifier import COLOR_BURGUNDY, COLOR_NEUTRAL
from mission_generator import Mission, DIFFICULTY_NORMAL
from hunt_tracker import (
    DEFAULT_DEDUP_WINDOW,
    DEFAULT_PROXIMITY_PX,
    END_NONE,
    END_TIMEOUT,
    END_WIN,
    HuntTracker,
    _bbox_center,
)


# ============================================================
# 테스트 픽스처
# ============================================================
@dataclass(frozen=True)
class FakeMatch:
    """object_pipeline.ObjectMatch와 동일 인터페이스. dataclass 가벼움."""
    yolo_class: str
    yolo_confidence: float
    bbox: tuple
    detected_color: str
    color_confidence: float
    is_target_match: bool


def make_mission(target=COLOR_BURGUNDY, goal=3, time_limit=60.0,
                 confidence_gate=0.65):
    return Mission(
        target_color=target,
        goal_count=goal,
        time_limit=time_limit,
        confidence_gate=confidence_gate,
        difficulty=DIFFICULTY_NORMAL,
    )


class FakeClock:
    """주입형 시간 — t를 set으로 진행."""
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def match(cls, cx, cy, color=COLOR_BURGUNDY, conf=0.9, is_match=True,
          bbox_size=40):
    """단일 fake match. bbox는 (cx, cy) 중심·크기 정사각형."""
    h = bbox_size // 2
    return FakeMatch(
        yolo_class=cls,
        yolo_confidence=0.9,
        bbox=(cx - h, cy - h, cx + h, cy + h),
        detected_color=color,
        color_confidence=conf,
        is_target_match=is_match,
    )


# ============================================================
# 1. apply_matches 기본
# ============================================================
class TestApplyMatchesBasic:
    def test_no_apply_before_start(self):
        t = HuntTracker(make_mission(), time_provider=FakeClock(0))
        gained = t.apply_matches([match("cup", 100, 100)])
        assert gained == 0
        assert t.collected == 0

    def test_single_match_counts(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(), time_provider=clock)
        t.start()
        gained = t.apply_matches([match("cup", 100, 100)])
        assert gained == 1
        assert t.collected == 1

    def test_non_target_color_rejected(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(), time_provider=clock)
        t.start()
        gained = t.apply_matches([
            match("cup", 100, 100, color="navy", is_match=False),
        ])
        assert gained == 0
        assert t.rejected_by_color == 1
        assert t.collected == 0

    def test_low_confidence_rejected(self):
        clock = FakeClock(0)
        mission = make_mission(confidence_gate=0.75)
        t = HuntTracker(mission, time_provider=clock)
        t.start()
        gained = t.apply_matches([
            match("cup", 100, 100, conf=0.70),  # < 0.75
        ])
        assert gained == 0
        assert t.rejected_by_confidence == 1
        assert t.collected == 0

    def test_attempts_counts_all(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(), time_provider=clock)
        t.start()
        t.apply_matches([
            match("cup", 100, 100),
            match("bottle", 200, 200, is_match=False),
            match("vase", 300, 300, conf=0.3),
        ])
        assert t.attempts == 3


# ============================================================
# 2. Dedup window
# ============================================================
class TestDedup:
    def test_same_object_within_window_is_duplicate(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(), time_provider=clock)
        t.start()
        # 같은 cup이 같은 자리에서 다시 잡힘
        t.apply_matches([match("cup", 100, 100)])
        clock.t = 2.0  # 2초 후 (dedup window 5초 이내)
        gained = t.apply_matches([match("cup", 100, 100)])
        assert gained == 0
        assert t.rejected_by_dedup == 1
        assert t.collected == 1

    def test_same_object_after_window_counts_again(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=10), time_provider=clock)
        t.start()
        t.apply_matches([match("cup", 100, 100)])
        clock.t = DEFAULT_DEDUP_WINDOW + 0.1  # window 만료
        gained = t.apply_matches([match("cup", 100, 100)])
        assert gained == 1
        assert t.collected == 2

    def test_same_class_far_apart_not_duplicate(self):
        # 같은 클래스라도 위치가 멀면 다른 객체
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=5), time_provider=clock)
        t.start()
        t.apply_matches([match("cup", 100, 100)])
        # 멀리 떨어진 위치 (DEFAULT_PROXIMITY_PX=80보다 크게)
        gained = t.apply_matches([match("cup", 300, 300)])
        assert gained == 1
        assert t.collected == 2

    def test_different_class_same_position_not_duplicate(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=5), time_provider=clock)
        t.start()
        t.apply_matches([match("cup", 100, 100)])
        gained = t.apply_matches([match("bottle", 100, 100)])
        # 클래스가 다르면 dedup 적용 X
        assert gained == 1
        assert t.collected == 2

    def test_proximity_boundary(self):
        # 정확히 PROXIMITY_PX 거리인 객체는 같은 객체로 친다 (<= 임계)
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=5), time_provider=clock)
        t.start()
        t.apply_matches([match("cup", 100, 100)])
        # 같은 y, x만 정확히 PROXIMITY_PX 만큼
        gained = t.apply_matches([
            match("cup", 100 + DEFAULT_PROXIMITY_PX, 100),
        ])
        assert gained == 0  # 임계점은 같은 객체
        assert t.rejected_by_dedup == 1


# ============================================================
# 3. 종료 사유
# ============================================================
class TestEndConditions:
    def test_win_when_goal_reached(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=3), time_provider=clock)
        t.start()
        # 다른 위치 3개 (dedup 회피)
        t.apply_matches([
            match("cup", 100, 100),
            match("cup", 300, 100),
            match("cup", 500, 100),
        ])
        assert t.collected == 3
        assert t.check_end() == END_WIN
        assert t.is_win()
        assert t.is_game_over()

    def test_timeout_when_time_runs_out(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=3, time_limit=10.0),
                         time_provider=clock)
        t.start()
        t.apply_matches([match("cup", 100, 100)])
        clock.t = 11.0  # 시간 초과
        assert t.check_end() == END_TIMEOUT
        assert not t.is_win()
        assert t.is_game_over()

    def test_no_end_before_start(self):
        t = HuntTracker(make_mission(), time_provider=FakeClock(0))
        assert t.check_end() == END_NONE
        assert not t.is_game_over()

    def test_no_end_in_progress(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=3), time_provider=clock)
        t.start()
        t.apply_matches([match("cup", 100, 100)])
        clock.t = 5.0  # 시간 여유
        assert t.check_end() == END_NONE

    def test_win_priority_over_timeout(self):
        # 정확히 시간 만료 + goal 달성 시점에 WIN 우선
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=1, time_limit=5.0),
                         time_provider=clock)
        t.start()
        clock.t = 5.0  # 시간 만료
        t.apply_matches([match("cup", 100, 100)])
        # collected가 goal에 도달했으므로 WIN
        assert t.check_end() == END_WIN

    def test_end_reason_cached(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=1), time_provider=clock)
        t.start()
        t.apply_matches([match("cup", 100, 100)])
        first = t.check_end()
        assert first == END_WIN
        # 이후 호출 모두 같은 결과
        for _ in range(5):
            assert t.check_end() == first

    def test_no_apply_after_end(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(goal=1), time_provider=clock)
        t.start()
        t.apply_matches([match("cup", 100, 100)])
        assert t.is_win()
        gained = t.apply_matches([match("cup", 200, 200)])
        assert gained == 0
        assert t.collected == 1


# ============================================================
# 4. 시간 / get_remaining
# ============================================================
class TestTime:
    def test_remaining_decreases(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(time_limit=10.0), time_provider=clock)
        t.start()
        assert t.get_remaining() == pytest.approx(10.0)
        clock.t = 3.5
        assert t.get_remaining() == pytest.approx(6.5)
        clock.t = 15.0
        assert t.get_remaining() == 0.0  # clipped

    def test_elapsed_zero_before_start(self):
        t = HuntTracker(make_mission(), time_provider=FakeClock(0))
        assert t.get_elapsed() == 0.0


# ============================================================
# 5. summary / reset
# ============================================================
class TestSummaryAndReset:
    def test_summary_keys(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(), time_provider=clock)
        t.start()
        s = t.get_summary()
        for k in (
            "target_color", "goal_count", "collected", "progress",
            "time_limit", "elapsed", "remaining", "end_reason",
            "attempts", "rejected_by_color",
            "rejected_by_confidence", "rejected_by_dedup",
        ):
            assert k in s

    def test_reset_clears_everything(self):
        clock = FakeClock(0)
        t = HuntTracker(make_mission(), time_provider=clock)
        t.start()
        t.apply_matches([match("cup", 100, 100)])
        t.reset()
        assert t.collected == 0
        assert t.attempts == 0
        assert t.start_time is None
        assert t.check_end() == END_NONE


# ============================================================
# 6. _bbox_center 유틸
# ============================================================
class TestBboxCenter:
    def test_center_simple(self):
        assert _bbox_center((10, 20, 30, 40)) == (20, 30)

    def test_center_negative_safe(self):
        # 음수 좌표라도 산술은 동작 (clipping은 호출측 책임)
        assert _bbox_center((-10, -20, 10, 20)) == (0, 0)
