"""
test_treasure_clues.py — 보물 풀 + 출제기 단위 테스트
"""

import pytest

from treasure_clues import (
    ALL_CLASSES,
    EASY_POOL,
    HARD_POOL,
    NORMAL_POOL,
    TREASURE_CATALOG,
    Treasure,
    TreasureGenerator,
    get_treasure_by_class,
)


# ============================================================
# 1. 카탈로그
# ============================================================
class TestCatalog:
    def test_count(self):
        assert len(TREASURE_CATALOG) == 12

    def test_unique_classes(self):
        classes = [t.yolo_class for t in TREASURE_CATALOG]
        assert len(classes) == len(set(classes))

    def test_unique_korean(self):
        kos = [t.ko for t in TREASURE_CATALOG]
        assert len(kos) == len(set(kos))

    def test_all_have_hint(self):
        for t in TREASURE_CATALOG:
            assert isinstance(t.hint, str)
            assert len(t.hint) > 2


# ============================================================
# 2. Pool monotonic
# ============================================================
class TestPools:
    def test_easy_subset_of_normal(self):
        assert set(EASY_POOL).issubset(set(NORMAL_POOL))

    def test_normal_subset_of_hard(self):
        assert set(NORMAL_POOL).issubset(set(HARD_POOL))

    def test_size_monotonic(self):
        assert len(EASY_POOL) < len(NORMAL_POOL) < len(HARD_POOL)

    def test_pools_in_catalog(self):
        for pool in (EASY_POOL, NORMAL_POOL, HARD_POOL):
            for cls in pool:
                assert cls in ALL_CLASSES


# ============================================================
# 3. get_treasure_by_class
# ============================================================
class TestGetByClass:
    def test_cup(self):
        t = get_treasure_by_class("cup")
        assert isinstance(t, Treasure)
        assert t.ko == "컵"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            get_treasure_by_class("dragon")


# ============================================================
# 4. TreasureGenerator
# ============================================================
class TestGenerator:
    def test_default_pool_is_normal(self):
        g = TreasureGenerator(seed=0)
        assert g.pool == NORMAL_POOL

    def test_returns_treasure(self):
        g = TreasureGenerator(seed=0)
        t = g.next_treasure()
        assert isinstance(t, Treasure)

    def test_target_in_pool(self):
        g = TreasureGenerator(pool=EASY_POOL, seed=42)
        seen = {g.next_treasure().yolo_class for _ in range(40)}
        assert seen.issubset(set(EASY_POOL))

    def test_avoids_immediate_repeat(self):
        g = TreasureGenerator(pool=NORMAL_POOL, seed=42)
        seq = [g.next_treasure().yolo_class for _ in range(50)]
        for i in range(len(seq) - 1):
            assert seq[i] != seq[i + 1]

    def test_seed_reproducibility(self):
        a = TreasureGenerator(seed=777)
        b = TreasureGenerator(seed=777)
        seq_a = [a.next_treasure().yolo_class for _ in range(20)]
        seq_b = [b.next_treasure().yolo_class for _ in range(20)]
        assert seq_a == seq_b

    def test_instance_isolation(self):
        a = TreasureGenerator(seed=42)
        b = TreasureGenerator(seed=42)
        a.next_treasure()
        a.next_treasure()
        seq_b = [b.next_treasure().yolo_class for _ in range(3)]
        c = TreasureGenerator(seed=42)
        seq_c = [c.next_treasure().yolo_class for _ in range(3)]
        assert seq_b == seq_c

    def test_invalid_pool_raises(self):
        with pytest.raises(ValueError):
            TreasureGenerator(pool=("unicorn",), seed=0)

    def test_single_class_pool_repeats(self):
        # pool 크기 1이면 회피 불가 → 같은 보물 반복
        g = TreasureGenerator(pool=("cup",), seed=0)
        t1 = g.next_treasure()
        t2 = g.next_treasure()
        assert t1.yolo_class == t2.yolo_class == "cup"

    def test_reset_clears_history(self):
        g = TreasureGenerator(seed=42)
        for _ in range(3):
            g.next_treasure()
        g.reset()
        assert g.history == []


# ============================================================
# 5. 한글 라벨 일관성
# ============================================================
class TestKoreanLabels:
    def test_cup_korean(self):
        assert get_treasure_by_class("cup").ko == "컵"

    def test_book_korean(self):
        assert get_treasure_by_class("book").ko == "책"

    def test_cell_phone_korean(self):
        assert get_treasure_by_class("cell phone").ko == "휴대폰"
