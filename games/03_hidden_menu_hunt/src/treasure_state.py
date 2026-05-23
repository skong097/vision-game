"""
treasure_state.py — W10 보물찾기 상태
============================================

60초 / 발견 카운트 / 유지 시간 누적 / 종료 사유.

흐름:
- 매 프레임 apply_detection(in_roi: bool, dt: float) 호출
- in_roi=True: hold_time 누적
- hold_time ≥ hold_required: 발견 카운트 +1, +20점 + 다음 보물 자동 출제
- in_roi=False: hold_time 0으로 리셋
- 종료:
    1) found_count ≥ goal → WIN (즉시) + 잔여 시간 보너스
    2) 60초 경과 → WIN(목표 충족) 또는 TIMEOUT

W8 sync_state + W7 dodge_state 패턴 결합.

Author: Stephen (gjkong)
Date: 2026-05-12 (W10 Step 3)
"""

import time

try:
    from .treasure_clues import (
        EASY_POOL,
        HARD_POOL,
        NORMAL_POOL,
        TreasureGenerator,
    )
except ImportError:
    from treasure_clues import (
        EASY_POOL,
        HARD_POOL,
        NORMAL_POOL,
        TreasureGenerator,
    )


# ============================================================
# 1. 종료 사유
# ============================================================
END_NONE = "none"
END_WIN = "win"
END_TIMEOUT = "timeout"


# ============================================================
# 2. 난이도
# ============================================================
DIFFICULTY_EASY = "easy"
DIFFICULTY_NORMAL = "normal"
DIFFICULTY_HARD = "hard"
DIFFICULTIES = (DIFFICULTY_EASY, DIFFICULTY_NORMAL, DIFFICULTY_HARD)

DIFFICULTY_CONFIG = {
    DIFFICULTY_EASY: {
        "goal":          3,
        "hold_required": 0.7,
        "pool":          EASY_POOL,
        "min_confidence": 0.40,
    },
    DIFFICULTY_NORMAL: {
        "goal":          5,
        "hold_required": 1.0,
        "pool":          NORMAL_POOL,
        "min_confidence": 0.45,
    },
    DIFFICULTY_HARD: {
        "goal":          7,
        "hold_required": 1.5,
        "pool":          HARD_POOL,
        "min_confidence": 0.50,
    },
}

# 점수
SCORE_PER_FIND = 20
TIME_BONUS_PER_SEC = 1

TIME_LIMIT = 60.0


# ============================================================
# 3. TreasureState
# ============================================================
class TreasureState:
    """W10 라운드 상태."""

    TIME_LIMIT = TIME_LIMIT

    def __init__(self, difficulty: str = DIFFICULTY_NORMAL,
                 seed=None, time_provider=None):
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError(f"알 수 없는 난이도: {difficulty}")
        self.difficulty = difficulty
        self.config = DIFFICULTY_CONFIG[difficulty]
        self._generator = TreasureGenerator(
            pool=self.config["pool"], seed=seed,
        )
        self._now = time_provider if time_provider is not None else time.time
        self.reset()

    def reset(self):
        self.score = 0
        self.found_count = 0
        self.attempts = 0          # 매 프레임 in_roi=True 누계 (통계)
        self.hold_time = 0.0
        self.found_history = []    # 발견된 yolo_class 리스트 (UI 통계)
        self.current_target = None
        self.start_time = None
        self._end_reason = END_NONE
        self._time_bonus = 0       # WIN 시 추가 보너스

    def start_game(self):
        self.start_time = self._now()
        if self.current_target is None:
            self.next_target()

    # ----------------------------------------
    # 출제
    # ----------------------------------------
    def next_target(self):
        """다음 보물 (직전 회피, 풀 내)."""
        t = self._generator.next_treasure()
        self.current_target = t
        self.hold_time = 0.0
        return t

    # ----------------------------------------
    # 프레임 적용
    # ----------------------------------------
    def apply_detection(self, in_roi: bool, dt: float) -> bool:
        """한 프레임 적용.

        Args:
            in_roi: 현재 미션 객체가 중앙 ROI 안에 있는가?
            dt: 이번 프레임 경과(초)

        Returns:
            True if 이번 호출로 발견 카운트가 늘었음.
        """
        if self._end_reason != END_NONE:
            return False
        if self.start_time is None:
            return False
        if self.current_target is None:
            return False

        if in_roi:
            self.attempts += 1
            self.hold_time += dt
            if self.hold_time >= self.config["hold_required"]:
                # 발견!
                self.score += SCORE_PER_FIND
                self.found_count += 1
                self.found_history.append(self.current_target.yolo_class)
                # 종료 체크 (목표 도달 시 시간 보너스 가산)
                if self.found_count >= self.config["goal"]:
                    remaining = self.get_remaining()
                    self._time_bonus = int(remaining * TIME_BONUS_PER_SEC)
                    self.score += self._time_bonus
                    self._end_reason = END_WIN
                else:
                    self.next_target()
                return True
        else:
            self.hold_time = 0.0
        return False

    # ----------------------------------------
    # 시간
    # ----------------------------------------
    def get_elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return self._now() - self.start_time

    def get_remaining(self) -> float:
        return max(0.0, self.TIME_LIMIT - self.get_elapsed())

    def get_hold_progress(self) -> float:
        req = self.config["hold_required"]
        if req <= 0:
            return 1.0
        return min(1.0, self.hold_time / req)

    # ----------------------------------------
    # 종료
    # ----------------------------------------
    def check_end(self) -> str:
        if self._end_reason != END_NONE:
            return self._end_reason
        if self.start_time is not None and self.get_remaining() <= 0:
            if self.found_count >= self.config["goal"]:
                self._end_reason = END_WIN
            else:
                self._end_reason = END_TIMEOUT
        return self._end_reason

    def is_game_over(self) -> bool:
        return self.check_end() != END_NONE

    def is_win(self) -> bool:
        return self.check_end() == END_WIN

    # ----------------------------------------
    # 조회
    # ----------------------------------------
    def get_goal(self) -> int:
        return self.config["goal"]

    def get_hold_required(self) -> float:
        return self.config["hold_required"]

    def get_min_confidence(self) -> float:
        return self.config["min_confidence"]

    def get_summary(self) -> dict:
        return {
            "difficulty":       self.difficulty,
            "score":            self.score,
            "found_count":      self.found_count,
            "goal":             self.get_goal(),
            "elapsed":          self.get_elapsed(),
            "remaining":        self.get_remaining(),
            "attempts":         self.attempts,
            "end_reason":       self.check_end(),
            "time_bonus":       self._time_bonus,
            "current_target_ko": (self.current_target.ko
                                  if self.current_target else None),
            "current_target_class": (self.current_target.yolo_class
                                     if self.current_target else None),
            "found_history":    list(self.found_history),
        }

    def __repr__(self):
        return (
            f"TreasureState(diff={self.difficulty}, "
            f"found={self.found_count}/{self.get_goal()}, "
            f"score={self.score}, end={self.check_end()})"
        )
