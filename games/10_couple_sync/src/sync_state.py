"""
sync_state.py — W8 미션·점수·유지 시간 누적
==================================================

흐름
----
- 미션 생성기: GAME_POSES 중 1 — 직전 회피, RNG 격리 (W2 트러블 #10)
- 매 프레임 sync 결과를 apply_sync() — both_match이면 유지 시간 누적
- 유지 시간 ≥ hold_required: 미션 완료, +score, +bonus(있으면), 다음 미션
- 유지 끊김(both_match=False): 유지 시간 0으로 리셋
- 종료:
    1) score ≥ target → WIN
    2) elapsed ≥ time_limit → TIMEOUT(점수 미달) 또는 WIN(점수 충족)

time 의존부는 외부에서 dt를 주입 — 단위 테스트 친화.

Author: Stephen (gjkong)
Date: 2026-05-12 (W8 Step 3)
"""

import random
import time

try:
    from .pose_classifier import GAME_POSES
except ImportError:
    from pose_classifier import GAME_POSES


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

# match_threshold: judge_sync의 임계
# hold_required: 유지 시간(초)
# target_score: 목표 점수
DIFFICULTY_CONFIG = {
    DIFFICULTY_EASY: {
        "match_threshold": 0.55,
        "hold_required":   0.5,
        "target_score":    80,
    },
    DIFFICULTY_NORMAL: {
        "match_threshold": 0.65,
        "hold_required":   0.7,
        "target_score":    120,
    },
    DIFFICULTY_HARD: {
        "match_threshold": 0.75,
        "hold_required":   1.0,
        "target_score":    200,
    },
}

# 점수
SCORE_PER_COMPLETION = 20
SCORE_BONUS = 5  # 둘 다 0.80 이상이면 추가

TIME_LIMIT = 60.0


# ============================================================
# 3. SyncState
# ============================================================
class SyncState:
    """W8 라운드 상태 (1 게임 = 60초 + 미션 회전).

    Attributes:
        difficulty
        target_score
        score
        completions: 완료한 미션 수
        bonus_count: 보너스 발생 수
        current_mission: 현재 출제 포즈
        hold_time: 두 사람이 매칭 유지한 누적 시간(초)
        history: 출제 이력
        start_time
        _end_reason: cache
    """

    TIME_LIMIT = TIME_LIMIT

    def __init__(self, difficulty: str = DIFFICULTY_NORMAL,
                 seed=None, time_provider=None):
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError(f"알 수 없는 난이도: {difficulty}")
        self.difficulty = difficulty
        self.config = DIFFICULTY_CONFIG[difficulty]
        self.target_score = self.config["target_score"]
        self._rng = random.Random(seed)
        self._now = time_provider if time_provider is not None else time.time
        self.reset()

    def reset(self):
        self.score = 0
        self.completions = 0
        self.bonus_count = 0
        self.hold_time = 0.0
        self.history = []
        self.current_mission = None
        self.start_time = None
        self._end_reason = END_NONE

    # ----------------------------------------------------------
    # 미션 회전
    # ----------------------------------------------------------
    def next_mission(self) -> str:
        """다음 미션 (직전 회피)."""
        candidates = list(GAME_POSES)
        if self.history:
            last = self.history[-1]
            candidates = [p for p in candidates if p != last]
        chosen = self._rng.choice(candidates)
        self.history.append(chosen)
        self.current_mission = chosen
        self.hold_time = 0.0
        return chosen

    def start_game(self):
        self.start_time = self._now()
        # 첫 미션 출제
        if self.current_mission is None:
            self.next_mission()

    # ----------------------------------------------------------
    # 매 프레임 sync 적용
    # ----------------------------------------------------------
    def apply_sync(self, dt: float, sync_result) -> bool:
        """매 프레임 sync 결과로 유지 시간 누적.

        Args:
            dt: 이번 프레임 경과(초)
            sync_result: sync_judge.SyncResult

        Returns:
            True if 이번 호출로 미션 완료 (다음 미션 자동 출제됨).
        """
        if self._end_reason != END_NONE:
            return False
        if self.start_time is None:
            return False

        if sync_result.both_match:
            self.hold_time += dt
            if self.hold_time >= self.config["hold_required"]:
                # 미션 완료
                gained = SCORE_PER_COMPLETION
                if sync_result.bonus:
                    gained += SCORE_BONUS
                    self.bonus_count += 1
                self.score += gained
                self.completions += 1
                # 다음 미션 자동 출제
                self.next_mission()
                return True
        else:
            # 매칭 끊김 — 유지 시간 리셋
            self.hold_time = 0.0
        return False

    # ----------------------------------------------------------
    # 시간
    # ----------------------------------------------------------
    def get_elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return self._now() - self.start_time

    def get_remaining(self) -> float:
        return max(0.0, self.TIME_LIMIT - self.get_elapsed())

    def get_hold_progress(self) -> float:
        """현재 미션의 유지 진행률 0~1."""
        req = self.config["hold_required"]
        if req <= 0:
            return 1.0
        return min(1.0, self.hold_time / req)

    # ----------------------------------------------------------
    # 종료
    # ----------------------------------------------------------
    def check_end(self) -> str:
        if self._end_reason != END_NONE:
            return self._end_reason
        if self.score >= self.target_score:
            self._end_reason = END_WIN
            return END_WIN
        if self.start_time is not None and self.get_elapsed() >= self.TIME_LIMIT:
            if self.score >= self.target_score:
                self._end_reason = END_WIN
                return END_WIN
            self._end_reason = END_TIMEOUT
            return END_TIMEOUT
        return END_NONE

    def is_game_over(self) -> bool:
        return self.check_end() != END_NONE

    def is_win(self) -> bool:
        return self.check_end() == END_WIN

    # ----------------------------------------------------------
    # 조회
    # ----------------------------------------------------------
    def get_match_threshold(self) -> float:
        return self.config["match_threshold"]

    def get_hold_required(self) -> float:
        return self.config["hold_required"]

    def get_summary(self) -> dict:
        return {
            "difficulty":       self.difficulty,
            "score":            self.score,
            "target_score":     self.target_score,
            "progress":         (self.score / self.target_score)
                                if self.target_score else 0.0,
            "completions":      self.completions,
            "bonus_count":      self.bonus_count,
            "current_mission":  self.current_mission,
            "hold_time":        self.hold_time,
            "elapsed":          self.get_elapsed(),
            "remaining":        self.get_remaining(),
            "end_reason":       self.check_end(),
            "history":          list(self.history),
        }

    def __repr__(self):
        return (
            f"SyncState(score={self.score}/{self.target_score}, "
            f"mission={self.current_mission}, "
            f"hold={self.hold_time:.2f}s, end={self.check_end()})"
        )
