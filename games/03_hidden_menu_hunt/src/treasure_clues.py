"""
treasure_clues.py — W10 보물 풀 + 출제기
================================================

매장 친화 COCO 클래스 12개 + AR 힌트 텍스트 + 출제기.

순수 모듈 — cv2/YOLO 의존 X. yolo_class는 W5 yolo_engine의 .class_name과 일치.

Author: Stephen (gjkong)
Date: 2026-05-12 (W10 Step 2)
"""

import random
from dataclasses import dataclass


# ============================================================
# 1. 보물 카탈로그
# ============================================================
@dataclass(frozen=True)
class Treasure:
    """한 보물 항목.

    Attributes:
        yolo_class: YOLO COCO 클래스 영문 (예: "cup")
        ko: 한글 표기
        hint: AR 힌트 한글 (한 줄)
    """
    yolo_class: str
    ko: str
    hint: str


# 매장 친화 — 카페·식당에서 흔히 보이는 객체
TREASURE_CATALOG = (
    Treasure("cup",         "컵",     "카페의 단짝!"),
    Treasure("bottle",      "병",     "음료 한 잔"),
    Treasure("book",        "책",     "조용한 시간"),
    Treasure("cell phone",  "휴대폰", "SNS 공유 좋아요"),
    Treasure("chair",       "의자",   "편안한 자리"),
    Treasure("laptop",      "노트북", "카공족 친구"),
    Treasure("keyboard",    "키보드", "타이핑 박자"),
    Treasure("vase",        "화병",   "인테리어 포인트"),
    Treasure("scissors",    "가위",   "작은 도구"),
    Treasure("mouse",       "마우스", "노트북 친구"),
    Treasure("fork",        "포크",   "디저트 시간"),
    Treasure("spoon",       "스푼",   "커피와 함께"),
)

ALL_CLASSES = tuple(t.yolo_class for t in TREASURE_CATALOG)


# ============================================================
# 2. 난이도별 풀 — 흔한 것부터 어려운 것까지
# ============================================================
EASY_POOL = ("cup", "book", "cell phone", "bottle", "chair")
NORMAL_POOL = EASY_POOL + ("laptop", "keyboard", "vase")
HARD_POOL = NORMAL_POOL + ("scissors", "mouse", "fork", "spoon")


# ============================================================
# 3. 보물 조회
# ============================================================
def get_treasure_by_class(yolo_class: str) -> Treasure:
    """YOLO 클래스 이름으로 Treasure 찾기."""
    for t in TREASURE_CATALOG:
        if t.yolo_class == yolo_class:
            return t
    raise ValueError(f"알 수 없는 보물: {yolo_class}")


# ============================================================
# 4. 출제기 — 직전 회피 + RNG 격리
# ============================================================
class TreasureGenerator:
    """난이도별 보물 출제기 (W2 트러블 #10 패턴)."""

    def __init__(self, pool=None, seed=None):
        """
        Args:
            pool: 출제 후보 클래스 튜플. None이면 NORMAL_POOL.
            seed: 시드
        """
        if pool is None:
            pool = NORMAL_POOL
        invalid = [c for c in pool if c not in ALL_CLASSES]
        if invalid:
            raise ValueError(f"풀에 모르는 클래스: {invalid}")
        self.pool = tuple(pool)
        self._rng = random.Random(seed)
        self.history = []

    def reset(self):
        self.history = []

    def next_treasure(self) -> Treasure:
        """다음 보물 (직전 회피 — 풀 크기 ≥ 2일 때만)."""
        candidates = list(self.pool)
        if self.history and len(candidates) > 1:
            last = self.history[-1]
            candidates = [c for c in candidates if c != last]
            if not candidates:
                candidates = list(self.pool)
        chosen = self._rng.choice(candidates)
        self.history.append(chosen)
        return get_treasure_by_class(chosen)
