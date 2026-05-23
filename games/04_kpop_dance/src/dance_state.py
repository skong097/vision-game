"""
dance_state.py — W6 K-Pop 댄스 라운드 상태
================================================

5라운드 출제·점수·종료 조건. W4 round_state와 동일 구조:
- 5라운드 만점 500점
- 4정답 이상 = WIN, 그 이하 = FINISHED (도전 부족)
- 인스턴스 RNG로 시드 격리 (W2 트러블 #10)
- 직전 포즈 회피 (W2 question_gen 패턴)
- 난이도별 측정 시간·정답 임계·부분 임계

모듈명 prefix `dance_`로 W5 트러블 #17 학습 반영 (평범한 이름 `round_state`도
W4와 충돌하므로 게임-specific prefix 사용).

Author: Stephen (gjkong)
Date: 2026-05-12 (W6 Step 3)
"""

import random

try:
    from .pose_classifier import GAME_POSES
except ImportError:
    from pose_classifier import GAME_POSES


# ============================================================
# 1. 종료 사유
# ============================================================
END_NONE = "none"
END_WIN = "win"
END_FINISHED = "finished"  # 5라운드 완료, 4정답 미만


# ============================================================
# 2. 난이도
# ============================================================
DIFFICULTY_EASY = "easy"
DIFFICULTY_NORMAL = "normal"
DIFFICULTY_HARD = "hard"
DIFFICULTIES = (DIFFICULTY_EASY, DIFFICULTY_NORMAL, DIFFICULTY_HARD)

DIFFICULTY_CONFIG = {
    DIFFICULTY_EASY:   {"measure_time": 3.5, "correct": 0.60, "partial": 0.30},
    DIFFICULTY_NORMAL: {"measure_time": 2.5, "correct": 0.70, "partial": 0.40},
    DIFFICULTY_HARD:   {"measure_time": 1.5, "correct": 0.80, "partial": 0.50},
}

SCORE_CORRECT = 100
SCORE_PARTIAL = 50
SCORE_FAIL = 0

TOTAL_ROUNDS = 5
WIN_THRESHOLD = 4


# ============================================================
# 3. 라운드 결과 (W4 동일 형식)
# ============================================================
def make_outcome(target: str, accuracy: float,
                 correct_threshold: float, partial_threshold: float) -> dict:
    """정확도(0~1)와 임계로부터 outcome dict 생성."""
    a = max(0.0, min(1.0, accuracy))
    if a >= correct_threshold:
        verdict = "correct"
        score = SCORE_CORRECT
    elif a >= partial_threshold:
        verdict = "partial"
        score = SCORE_PARTIAL
    else:
        verdict = "fail"
        score = SCORE_FAIL
    return {
        "target": target,
        "accuracy": a,
        "verdict": verdict,
        "score": score,
    }


# ============================================================
# 4. DanceState
# ============================================================
class DanceState:
    """5라운드 진행 상태 (W4 RoundState 동일 인터페이스)."""

    TOTAL_ROUNDS = TOTAL_ROUNDS
    WIN_THRESHOLD = WIN_THRESHOLD

    def __init__(self, difficulty: str = DIFFICULTY_NORMAL, seed=None):
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError(f"알 수 없는 난이도: {difficulty}")
        self.difficulty = difficulty
        self.config = DIFFICULTY_CONFIG[difficulty]
        self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.round_index = 0
        self.outcomes = []
        self.total_score = 0
        self.correct_count = 0
        self.partial_count = 0
        self.fail_count = 0
        self.history = []
        self._end_reason = END_NONE

    # ----------------------------------------------------------
    # 출제
    # ----------------------------------------------------------
    def next_target(self) -> str:
        """다음 출제 포즈 (직전 회피)."""
        candidates = list(GAME_POSES)
        if self.history:
            last = self.history[-1]
            candidates = [p for p in candidates if p != last]
        chosen = self._rng.choice(candidates)
        self.history.append(chosen)
        return chosen

    # ----------------------------------------------------------
    # 라운드 결과 적용
    # ----------------------------------------------------------
    def apply_round(self, target: str, accuracy: float) -> dict:
        if self.round_index >= self.TOTAL_ROUNDS:
            raise RuntimeError(
                f"이미 {self.TOTAL_ROUNDS}라운드 완료."
            )
        outcome = make_outcome(
            target=target,
            accuracy=accuracy,
            correct_threshold=self.config["correct"],
            partial_threshold=self.config["partial"],
        )
        self.outcomes.append(outcome)
        self.total_score += outcome["score"]
        if outcome["verdict"] == "correct":
            self.correct_count += 1
        elif outcome["verdict"] == "partial":
            self.partial_count += 1
        else:
            self.fail_count += 1
        self.round_index += 1
        return outcome

    # ----------------------------------------------------------
    # 종료
    # ----------------------------------------------------------
    def check_end(self) -> str:
        if self._end_reason != END_NONE:
            return self._end_reason
        if self.round_index < self.TOTAL_ROUNDS:
            return END_NONE
        if self.correct_count >= self.WIN_THRESHOLD:
            self._end_reason = END_WIN
        else:
            self._end_reason = END_FINISHED
        return self._end_reason

    def is_game_over(self) -> bool:
        return self.check_end() != END_NONE

    def is_win(self) -> bool:
        return self.check_end() == END_WIN

    # ----------------------------------------------------------
    # 조회
    # ----------------------------------------------------------
    def get_measure_time(self) -> float:
        return self.config["measure_time"]

    def get_correct_threshold(self) -> float:
        return self.config["correct"]

    def get_partial_threshold(self) -> float:
        return self.config["partial"]

    def get_max_score(self) -> int:
        return self.TOTAL_ROUNDS * SCORE_CORRECT

    def get_summary(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "round_index": self.round_index,
            "total_rounds": self.TOTAL_ROUNDS,
            "score": self.total_score,
            "max_score": self.get_max_score(),
            "correct": self.correct_count,
            "partial": self.partial_count,
            "fail": self.fail_count,
            "win_threshold": self.WIN_THRESHOLD,
            "end_reason": self.check_end(),
            "history": list(self.history),
        }

    def __repr__(self):
        return (
            f"DanceState(diff={self.difficulty}, "
            f"round={self.round_index}/{self.TOTAL_ROUNDS}, "
            f"score={self.total_score}, "
            f"correct={self.correct_count})"
        )
