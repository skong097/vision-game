"""
silent_words.py — W9 단어 풀 + 출제기
==============================================

한국어 단어 10개 + 출제기 (직전 회피, 인스턴스 RNG).
각 단어에 "core action" 힌트(텍스트)도 정의 — LLM 프롬프트에 활용.

순수 모듈. cv2/MediaPipe/API 의존 X.

Author: Stephen (gjkong)
Date: 2026-05-12 (W9 Step 2)
"""

import random
from dataclasses import dataclass


# ============================================================
# 1. 단어 카탈로그
# ============================================================
@dataclass(frozen=True)
class Word:
    """한 단어 항목.

    Attributes:
        ko: 한국어 표기
        slug: 영문 id (test/log용)
        hint: LLM 평가 가이드용 핵심 동작 설명
    """
    ko: str
    slug: str
    hint: str


# 단어 풀 — V1은 10개로 시작
WORD_CATALOG = (
    Word("강아지", "puppy",
         "양손을 머리 위 귀 모양 OR 네발 자세"),
    Word("비행기", "airplane",
         "양팔을 좌우로 길게 펴고 좌우로 기울이기"),
    Word("농구", "basketball",
         "한 손을 위로 올려 슛 자세 또는 드리블 동작"),
    Word("노래", "sing",
         "한 손을 입 가까이 가져가 마이크 잡은 동작"),
    Word("잠자기", "sleep",
         "양손을 모아 뺨 옆에 대고 눈 감기"),
    Word("책 읽기", "read",
         "양손을 펼쳐 들고 시선을 아래로 내림"),
    Word("박수", "clap",
         "양손을 가슴 앞에서 모아 빠르게 부딪힘"),
    Word("춤추기", "dance",
         "어깨를 흔들거나 양팔을 좌우로 흔들기"),
    Word("화남", "angry",
         "두 주먹을 쥐고 어깨를 들썩이는 화난 자세"),
    Word("놀람", "surprised",
         "입을 크게 벌리고 양손을 얼굴 옆으로 올림"),
)

ALL_SLUGS = tuple(w.slug for w in WORD_CATALOG)


def get_word_by_slug(slug: str) -> Word:
    """slug → Word. 없으면 ValueError."""
    for w in WORD_CATALOG:
        if w.slug == slug:
            return w
    raise ValueError(f"알 수 없는 단어 slug: {slug}")


# ============================================================
# 2. 출제기
# ============================================================
class WordGenerator:
    """단어 출제기 — 직전 회피 + 인스턴스 RNG 격리 (W2 트러블 #10)."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self.history = []   # 출제된 slug 리스트

    def reset(self):
        """이력 초기화 (RNG 상태는 보존)."""
        self.history = []

    def next_word(self) -> Word:
        """다음 출제 단어 — 직전 회피."""
        candidates = list(WORD_CATALOG)
        if self.history:
            last_slug = self.history[-1]
            candidates = [w for w in candidates if w.slug != last_slug]
        chosen = self._rng.choice(candidates)
        self.history.append(chosen.slug)
        return chosen
