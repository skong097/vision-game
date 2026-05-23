"""
test_silent_words.py — 단어 풀 + 출제기 단위 테스트
==========================================================

검증:
- 단어 풀 10개, 한국어 + slug + hint 모두 존재
- 직전 회피
- 시드 재현성, 인스턴스 격리
- get_word_by_slug 정확성
- 잘못된 slug 예외
"""

import pytest

from silent_words import (
    ALL_SLUGS,
    WORD_CATALOG,
    Word,
    WordGenerator,
    get_word_by_slug,
)


# ============================================================
# 1. 카탈로그
# ============================================================
class TestCatalog:
    def test_ten_words(self):
        assert len(WORD_CATALOG) == 10

    def test_all_slugs_unique(self):
        slugs = [w.slug for w in WORD_CATALOG]
        assert len(slugs) == len(set(slugs))

    def test_all_korean_unique(self):
        kos = [w.ko for w in WORD_CATALOG]
        assert len(kos) == len(set(kos))

    def test_all_have_hint(self):
        for w in WORD_CATALOG:
            assert isinstance(w.hint, str)
            assert len(w.hint) > 5

    def test_all_slugs_tuple_matches(self):
        assert ALL_SLUGS == tuple(w.slug for w in WORD_CATALOG)


# ============================================================
# 2. get_word_by_slug
# ============================================================
class TestGetWordBySlug:
    def test_puppy(self):
        w = get_word_by_slug("puppy")
        assert isinstance(w, Word)
        assert w.ko == "강아지"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            get_word_by_slug("nonexistent")


# ============================================================
# 3. 출제기 — 출제 결과
# ============================================================
class TestNextWord:
    def test_returns_word(self):
        g = WordGenerator(seed=0)
        w = g.next_word()
        assert isinstance(w, Word)

    def test_in_catalog(self):
        g = WordGenerator(seed=42)
        slugs = {g.next_word().slug for _ in range(30)}
        assert slugs.issubset(set(ALL_SLUGS))

    def test_avoids_immediate_repeat(self):
        g = WordGenerator(seed=42)
        seq = [g.next_word().slug for _ in range(50)]
        for i in range(len(seq) - 1):
            assert seq[i] != seq[i + 1]

    def test_history_recorded(self):
        g = WordGenerator(seed=0)
        for _ in range(5):
            g.next_word()
        assert len(g.history) == 5
        assert all(s in ALL_SLUGS for s in g.history)


# ============================================================
# 4. Seed
# ============================================================
class TestSeed:
    def test_same_seed_same_sequence(self):
        a = WordGenerator(seed=777)
        b = WordGenerator(seed=777)
        seq_a = [a.next_word().slug for _ in range(20)]
        seq_b = [b.next_word().slug for _ in range(20)]
        assert seq_a == seq_b

    def test_instance_isolation(self):
        a = WordGenerator(seed=42)
        b = WordGenerator(seed=42)
        a.next_word()
        a.next_word()
        seq_b = [b.next_word().slug for _ in range(3)]
        c = WordGenerator(seed=42)
        seq_c = [c.next_word().slug for _ in range(3)]
        assert seq_b == seq_c


# ============================================================
# 5. Reset
# ============================================================
class TestReset:
    def test_reset_clears_history(self):
        g = WordGenerator(seed=0)
        for _ in range(3):
            g.next_word()
        g.reset()
        assert g.history == []
