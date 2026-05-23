"""
round_state.py — W4 표정 미러링 라운드 진행 + 점수 + 종료조건
================================================================

W2의 ScoreTracker, W3의 ScoreState와 비슷한 위치. W4는 다음 점이 다름:
- 5라운드 고정 (시간 누적 X)
- 라운드마다 정확도(0~1) → 점수(0/50/100) 변환
- 직전 표정 회피하며 4종 중 1종 랜덤 출제

순수 클래스 (vision/UI/MediaPipe 의존성 없음).
인스턴스 RNG(random.Random)로 시드 격리 — W2 트러블 #10 패턴.

Author: Stephen (gjkong)
Date: 2026-05-12 (W4 Step 3)
"""

import random

try:
    from .expression_classifier import GAME_EXPRESSIONS
except ImportError:
    from expression_classifier import GAME_EXPRESSIONS


# ============================================================
# 1. 종료 사유
# ============================================================
END_NONE = "none"
END_WIN = "win"           # 4정답 이상
END_FINISHED = "finished"  # 5라운드 완료, 4정답 미만 (도전 부족)


# ============================================================
# 2. 난이도별 측정 시간 + 정답 임계 + 부분 임계
# ============================================================
DIFFICULTY_EASY = "easy"
DIFFICULTY_NORMAL = "normal"
DIFFICULTY_HARD = "hard"
DIFFICULTIES = (DIFFICULTY_EASY, DIFFICULTY_NORMAL, DIFFICULTY_HARD)

# 측정 시간(초), 정답 임계, 부분 임계
DIFFICULTY_CONFIG = {
    DIFFICULTY_EASY:   {"measure_time": 3.0, "correct": 0.60, "partial": 0.30},
    DIFFICULTY_NORMAL: {"measure_time": 2.0, "correct": 0.70, "partial": 0.40},
    DIFFICULTY_HARD:   {"measure_time": 1.2, "correct": 0.80, "partial": 0.50},
}

# 점수
SCORE_CORRECT = 100
SCORE_PARTIAL = 50
SCORE_FAIL = 0

TOTAL_ROUNDS = 5
WIN_THRESHOLD = 4   # 4정답 이상 = 승리


# ============================================================
# 3. RoundOutcome — 한 라운드 결과 (불변 dict)
# ============================================================
def make_outcome(target: str, accuracy: float,
                 correct_threshold: float, partial_threshold: float) -> dict:
    """정확도(0~1)와 임계값으로부터 라운드 결과 dict 생성

    Args:
        target: 출제 표정 (smile/sad/surprised/angry)
        accuracy: 0.0~1.0 (clipped). expression_similarity() 결과
        correct_threshold: 정답 임계
        partial_threshold: 부분 정답 임계

    Returns:
        {
            "target": str,
            "accuracy": float,        # clipped 0~1
            "verdict": "correct"/"partial"/"fail",
            "score": int,             # 100/50/0
        }
    """
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
# 4. RoundState — 5라운드 진행 + 점수 + 출제
# ============================================================
class RoundState:
    """5라운드 진행 상태

    Attributes:
        difficulty: 난이도 문자열
        round_index: 현재 라운드 번호 (0-based, 0이면 아직 시작 전)
        outcomes: 각 라운드의 결과 dict 리스트
        total_score: 누적 점수
        correct_count: 정답 수
        partial_count: 부분 정답 수
        fail_count: 실패 수
        history: 출제된 target 리스트 (UI/통계용)
    """

    TOTAL_ROUNDS = TOTAL_ROUNDS
    WIN_THRESHOLD = WIN_THRESHOLD

    def __init__(self, difficulty: str = DIFFICULTY_NORMAL, seed=None):
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError(f"알 수 없는 난이도: {difficulty}")
        self.difficulty = difficulty
        self.config = DIFFICULTY_CONFIG[difficulty]
        # 인스턴스 RNG (전역 random 오염 방지)
        self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        """라운드 진행 상태 초기화 (RNG 상태는 보존)"""
        self.round_index = 0
        self.outcomes = []
        self.total_score = 0
        self.correct_count = 0
        self.partial_count = 0
        self.fail_count = 0
        self.history = []
        self._end_reason = END_NONE

    # ------------------------------------------------------------
    # 출제
    # ------------------------------------------------------------
    def next_target(self) -> str:
        """다음 라운드의 출제 표정 (직전 표정 회피)

        Returns:
            GAME_EXPRESSIONS 중 하나. 직전 라운드와 다름.
        """
        candidates = list(GAME_EXPRESSIONS)
        if self.history:
            last = self.history[-1]
            candidates = [e for e in candidates if e != last]
        chosen = self._rng.choice(candidates)
        self.history.append(chosen)
        return chosen

    # ------------------------------------------------------------
    # 라운드 결과 적용
    # ------------------------------------------------------------
    def apply_round(self, target: str, accuracy: float) -> dict:
        """라운드 결과를 누적 상태에 반영하고 outcome dict 반환

        Args:
            target: 출제된 표정 (UI 표시·로깅용)
            accuracy: 0.0~1.0 (clipping은 make_outcome에서 처리)

        Returns:
            outcome dict (make_outcome 결과)

        Raises:
            RuntimeError: 이미 5라운드 완료된 상태
        """
        if self.round_index >= self.TOTAL_ROUNDS:
            raise RuntimeError(
                f"이미 {self.TOTAL_ROUNDS}라운드 완료. apply_round 호출 불가."
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

    # ------------------------------------------------------------
    # 종료
    # ------------------------------------------------------------
    def check_end(self) -> str:
        """게임 종료 사유

        - 5라운드 미달: END_NONE
        - 5라운드 완료 + 4정답 이상: END_WIN
        - 5라운드 완료 + 4정답 미만: END_FINISHED (도전 부족)

        Returns:
            END_NONE / END_WIN / END_FINISHED
        """
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

    # ------------------------------------------------------------
    # 조회/UI
    # ------------------------------------------------------------
    def get_measure_time(self) -> float:
        """현재 난이도의 측정 시간(초)"""
        return self.config["measure_time"]

    def get_correct_threshold(self) -> float:
        return self.config["correct"]

    def get_partial_threshold(self) -> float:
        return self.config["partial"]

    def get_max_score(self) -> int:
        return self.TOTAL_ROUNDS * SCORE_CORRECT  # 500

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
            f"RoundState(diff={self.difficulty}, "
            f"round={self.round_index}/{self.TOTAL_ROUNDS}, "
            f"score={self.total_score}, "
            f"correct={self.correct_count})"
        )
