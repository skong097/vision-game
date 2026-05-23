"""
mission_generator.py — 난이도 → 미션(색·개수·시간) 생성
============================================================

W5 컬러 헌트의 라운드 출제기. 순수 모듈 (cv2/YOLO/카메라 의존성 없음).

설계 메모
---------
- **인스턴스 RNG** 사용 (random.Random) → 전역 random 오염 방지 (W2 트러블 #10 패턴)
- **직전 색 회피** — 같은 색 연속 출제 X (W2 question_gen 패턴)
- 난이도별 미션 풀:
    easy:   흔한 색 (크림·차콜) 위주, 2개, 90초
    normal: 8색 균등, 3개, 60초
    hard:   톤 비슷한 그룹 (모브·버건디, 테라코타·머스타드, 세이지·네이비) 위주,
            4개, 45초, 분류 신뢰도 게이트 0.75
- hard의 "톤 비슷한 색" 컨셉: 미션을 풀기 위해 손님이 색 차이를 능동적으로
  판단해야 함 — UI에서 두 색을 함께 보여주면 학습 효과까지.

Author: Stephen (gjkong)
Date: 2026-05-12 (W5 Step 3)
"""

import random
from dataclasses import dataclass

try:
    from .color_classifier import (
        COLOR_BURGUNDY, COLOR_CHARCOAL, COLOR_CREAM,
        COLOR_MAUVE, COLOR_MUSTARD, COLOR_NAVY,
        COLOR_SAGE, COLOR_TERRACOTTA,
        MISSION_COLORS,
    )
except ImportError:
    from color_classifier import (
        COLOR_BURGUNDY, COLOR_CHARCOAL, COLOR_CREAM,
        COLOR_MAUVE, COLOR_MUSTARD, COLOR_NAVY,
        COLOR_SAGE, COLOR_TERRACOTTA,
        MISSION_COLORS,
    )


# ============================================================
# 1. 난이도 상수
# ============================================================
DIFFICULTY_EASY = "easy"
DIFFICULTY_NORMAL = "normal"
DIFFICULTY_HARD = "hard"
DIFFICULTIES = (DIFFICULTY_EASY, DIFFICULTY_NORMAL, DIFFICULTY_HARD)


# ============================================================
# 2. 난이도별 설정
# ============================================================
# pool: 출제 후보 색 (편중 가능)
# goal: 목표 객체 개수
# time_limit: 제한 시간(초)
# confidence_gate: 객체의 색 분류 신뢰도가 이 미만이면 카운트 X
DIFFICULTY_CONFIG = {
    DIFFICULTY_EASY: {
        # 흔한 색 위주 (크림/차콜이 매장에 자주 등장 — 컵·접시·메뉴판)
        "pool": (COLOR_CREAM, COLOR_CHARCOAL, COLOR_NAVY, COLOR_MUSTARD),
        "goal": 2,
        "time_limit": 90.0,
        "confidence_gate": 0.60,
    },
    DIFFICULTY_NORMAL: {
        # 8색 균등
        "pool": tuple(MISSION_COLORS),
        "goal": 3,
        "time_limit": 60.0,
        "confidence_gate": 0.65,
    },
    DIFFICULTY_HARD: {
        # 톤 비슷한 색 위주 (분류 게이트 0.75로 strict)
        "pool": (
            COLOR_BURGUNDY, COLOR_MAUVE,
            COLOR_TERRACOTTA, COLOR_MUSTARD,
            COLOR_SAGE, COLOR_NAVY,
        ),
        "goal": 4,
        "time_limit": 45.0,
        "confidence_gate": 0.75,
    },
}


# ============================================================
# 3. Mission 레코드
# ============================================================
@dataclass(frozen=True)
class Mission:
    """한 라운드 미션.

    Attributes:
        target_color: 손님이 찾아야 할 색 (MISSION_COLORS 중 하나)
        goal_count: 모아야 하는 매칭 객체 수
        time_limit: 제한 시간 (초)
        confidence_gate: 색 분류 신뢰도 임계 (객체별 color_confidence ≥ 이 값일 때만 카운트)
        difficulty: 'easy' / 'normal' / 'hard'
    """
    target_color: str
    goal_count: int
    time_limit: float
    confidence_gate: float
    difficulty: str


# ============================================================
# 4. 미션 출제기
# ============================================================
class MissionGenerator:
    """난이도별 미션 출제기 — 직전 색 회피 + 인스턴스 RNG 격리."""

    def __init__(self, difficulty: str = DIFFICULTY_NORMAL, seed=None):
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError(f"알 수 없는 난이도: {difficulty}")
        self.difficulty = difficulty
        self.config = DIFFICULTY_CONFIG[difficulty]
        self._rng = random.Random(seed)
        self.history = []  # 출제된 target_color 리스트

    def reset(self):
        """이력 초기화 (RNG 상태는 보존)."""
        self.history = []

    def next_mission(self) -> Mission:
        """다음 미션 생성 — 직전 색과 다른 색 선택.

        pool 크기가 1이면 직전 회피 불가, 같은 색 반복.
        """
        pool = list(self.config["pool"])
        if self.history and len(pool) > 1:
            last = self.history[-1]
            candidates = [c for c in pool if c != last]
            if not candidates:
                candidates = pool
        else:
            candidates = pool

        target = self._rng.choice(candidates)
        self.history.append(target)
        return Mission(
            target_color=target,
            goal_count=self.config["goal"],
            time_limit=self.config["time_limit"],
            confidence_gate=self.config["confidence_gate"],
            difficulty=self.difficulty,
        )

    # ----------------------------------------------------------
    # 조회
    # ----------------------------------------------------------
    def get_pool(self) -> tuple:
        return tuple(self.config["pool"])

    def get_goal(self) -> int:
        return self.config["goal"]

    def get_time_limit(self) -> float:
        return self.config["time_limit"]

    def get_confidence_gate(self) -> float:
        return self.config["confidence_gate"]
