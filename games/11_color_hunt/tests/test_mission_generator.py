"""
test_mission_generator.py — 미션 출제기 단위 테스트
========================================================

검증:
- 난이도별 풀·목표·시간·신뢰도 게이트
- 직전 색 회피
- 시드 재현성 + 인스턴스 격리
- 단조성 (easy↔hard 시간/목표/게이트)
- 잘못된 입력 예외
- pool 풀 사이즈 1 edge (회피 불가)
"""

import pytest

from color_classifier import COLOR_CREAM, COLOR_NEUTRAL, MISSION_COLORS
from mission_generator import (
    DIFFICULTIES,
    DIFFICULTY_CONFIG,
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_NORMAL,
    Mission,
    MissionGenerator,
)


# ============================================================
# 1. 난이도별 config
# ============================================================
class TestDifficultyConfig:
    def test_all_difficulties_present(self):
        assert set(DIFFICULTIES) == set(DIFFICULTY_CONFIG.keys())

    def test_easy_config(self):
        c = DIFFICULTY_CONFIG[DIFFICULTY_EASY]
        assert c["goal"] == 2
        assert c["time_limit"] == 90.0
        assert c["confidence_gate"] == 0.60
        # 흔한 색이 포함되어야 함
        assert COLOR_CREAM in c["pool"]

    def test_normal_uses_all_colors(self):
        c = DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]
        assert set(c["pool"]) == set(MISSION_COLORS)
        assert c["goal"] == 3
        assert c["time_limit"] == 60.0

    def test_hard_strict(self):
        c = DIFFICULTY_CONFIG[DIFFICULTY_HARD]
        assert c["goal"] == 4
        assert c["time_limit"] == 45.0
        assert c["confidence_gate"] == 0.75

    def test_monotonic_goal(self):
        # 목표 개수는 단조 증가
        assert (DIFFICULTY_CONFIG[DIFFICULTY_EASY]["goal"]
                < DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["goal"]
                < DIFFICULTY_CONFIG[DIFFICULTY_HARD]["goal"])

    def test_monotonic_time(self):
        # 제한 시간은 단조 감소
        assert (DIFFICULTY_CONFIG[DIFFICULTY_EASY]["time_limit"]
                > DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["time_limit"]
                > DIFFICULTY_CONFIG[DIFFICULTY_HARD]["time_limit"])

    def test_monotonic_gate(self):
        # 게이트는 단조 증가
        assert (DIFFICULTY_CONFIG[DIFFICULTY_EASY]["confidence_gate"]
                <= DIFFICULTY_CONFIG[DIFFICULTY_NORMAL]["confidence_gate"]
                < DIFFICULTY_CONFIG[DIFFICULTY_HARD]["confidence_gate"])


# ============================================================
# 2. next_mission — 출제 결과
# ============================================================
class TestNextMission:
    def test_returns_mission_dataclass(self):
        g = MissionGenerator(DIFFICULTY_NORMAL, seed=0)
        m = g.next_mission()
        assert isinstance(m, Mission)
        assert m.target_color in MISSION_COLORS
        assert m.target_color != COLOR_NEUTRAL
        assert m.goal_count == 3
        assert m.time_limit == 60.0
        assert m.difficulty == DIFFICULTY_NORMAL

    def test_target_within_pool(self):
        for diff in DIFFICULTIES:
            g = MissionGenerator(diff, seed=1)
            pool = set(DIFFICULTY_CONFIG[diff]["pool"])
            for _ in range(30):
                m = g.next_mission()
                assert m.target_color in pool

    def test_easy_pool_subset(self):
        # easy의 pool은 MISSION_COLORS의 진부분집합 (흔한 색만)
        g = MissionGenerator(DIFFICULTY_EASY, seed=0)
        targets = {g.next_mission().target_color for _ in range(80)}
        easy_pool = set(DIFFICULTY_CONFIG[DIFFICULTY_EASY]["pool"])
        assert targets.issubset(easy_pool)
        assert targets.issubset(set(MISSION_COLORS))
        assert len(easy_pool) < len(MISSION_COLORS)

    def test_avoids_immediate_repeat(self):
        g = MissionGenerator(DIFFICULTY_NORMAL, seed=42)
        targets = [g.next_mission().target_color for _ in range(50)]
        for i in range(len(targets) - 1):
            assert targets[i] != targets[i + 1], (
                f"직전 회피 실패: idx {i}={targets[i]}, idx {i+1}={targets[i+1]}"
            )

    def test_history_recorded(self):
        g = MissionGenerator(DIFFICULTY_NORMAL, seed=0)
        for _ in range(5):
            g.next_mission()
        assert len(g.history) == 5

    def test_hard_avoids_neighbors_only_if_pool_allows(self):
        # hard pool에서 30회 출제 시에도 직전 회피 유지
        g = MissionGenerator(DIFFICULTY_HARD, seed=7)
        targets = [g.next_mission().target_color for _ in range(30)]
        for i in range(len(targets) - 1):
            assert targets[i] != targets[i + 1]


# ============================================================
# 3. Seed reproducibility / isolation
# ============================================================
class TestSeed:
    def test_same_seed_same_sequence(self):
        a = MissionGenerator(DIFFICULTY_NORMAL, seed=999)
        b = MissionGenerator(DIFFICULTY_NORMAL, seed=999)
        seq_a = [a.next_mission().target_color for _ in range(20)]
        seq_b = [b.next_mission().target_color for _ in range(20)]
        assert seq_a == seq_b

    def test_instance_isolation(self):
        # 두 인스턴스가 같은 시드라도 서로 영향 X
        a = MissionGenerator(DIFFICULTY_NORMAL, seed=42)
        b = MissionGenerator(DIFFICULTY_NORMAL, seed=42)
        a.next_mission()
        a.next_mission()
        b_seq = [b.next_mission().target_color for _ in range(3)]
        c = MissionGenerator(DIFFICULTY_NORMAL, seed=42)
        c_seq = [c.next_mission().target_color for _ in range(3)]
        assert b_seq == c_seq  # b·c 신선 인스턴스의 첫 결과는 같아야


# ============================================================
# 4. Reset & helpers
# ============================================================
class TestResetAndHelpers:
    def test_reset_clears_history_keeps_rng(self):
        g = MissionGenerator(DIFFICULTY_NORMAL, seed=42)
        g.next_mission()
        g.next_mission()
        g.reset()
        assert g.history == []
        # reset 후 next_mission도 정상
        m = g.next_mission()
        assert m.target_color in MISSION_COLORS

    def test_getter_methods(self):
        g = MissionGenerator(DIFFICULTY_HARD, seed=0)
        assert g.get_goal() == 4
        assert g.get_time_limit() == 45.0
        assert g.get_confidence_gate() == 0.75
        assert isinstance(g.get_pool(), tuple)
        assert len(g.get_pool()) >= 2


# ============================================================
# 5. Edge cases
# ============================================================
class TestEdgeCases:
    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError):
            MissionGenerator("insane", seed=0)

    def test_single_color_pool_repeats(self, monkeypatch):
        # pool 크기가 1이면 직전 회피 불가 → 같은 색 반복 허용
        from mission_generator import DIFFICULTY_CONFIG
        monkeypatch.setitem(
            DIFFICULTY_CONFIG, DIFFICULTY_NORMAL,
            {**DIFFICULTY_CONFIG[DIFFICULTY_NORMAL],
             "pool": (COLOR_CREAM,)},
        )
        g = MissionGenerator(DIFFICULTY_NORMAL, seed=0)
        m1 = g.next_mission()
        m2 = g.next_mission()
        assert m1.target_color == m2.target_color == COLOR_CREAM
