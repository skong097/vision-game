"""
silent_state.py — W9 라운드 진행·점수·종료
================================================

5라운드, W4 round_state + W6 dance_state 패턴.
LLM 점수(0~100) → outcome (correct/partial/fail) 변환.

순수 클래스 — cv2/MediaPipe/anthropic 의존 X.

Author: Stephen (gjkong)
Date: 2026-05-12 (W9 Step 4)
"""

import random

try:
    from .silent_words import WordGenerator
except ImportError:
    from silent_words import WordGenerator


# ============================================================
# 1. 종료 사유
# ============================================================
END_NONE = "none"
END_WIN = "win"
END_FINISHED = "finished"   # 5라운드 완료, 4정답 미만


# ============================================================
# 2. 난이도
# ============================================================
DIFFICULTY_EASY = "easy"
DIFFICULTY_NORMAL = "normal"
DIFFICULTY_HARD = "hard"
DIFFICULTIES = (DIFFICULTY_EASY, DIFFICULTY_NORMAL, DIFFICULTY_HARD)

# 표현 시간(초) + 정답 임계 + 부분 임계 (W4 패턴)
DIFFICULTY_CONFIG = {
    DIFFICULTY_EASY:   {"express_time": 20.0, "correct": 60, "partial": 30},
    DIFFICULTY_NORMAL: {"express_time": 15.0, "correct": 70, "partial": 40},
    DIFFICULTY_HARD:   {"express_time": 10.0, "correct": 80, "partial": 50},
}

SCORE_CORRECT = 100
SCORE_PARTIAL = 50
SCORE_FAIL = 0

TOTAL_ROUNDS = 5
WIN_THRESHOLD = 4


# ============================================================
# 3. outcome 변환
# ============================================================
def make_outcome(word_ko: str, llm_score: int, llm_comment: str,
                 correct_threshold: int, partial_threshold: int,
                 used_api: bool) -> dict:
    """LLM 점수(0~100)와 임계로부터 outcome dict 생성."""
    s = max(0, min(100, int(llm_score)))
    if s >= correct_threshold:
        verdict = "correct"
        round_score = SCORE_CORRECT
    elif s >= partial_threshold:
        verdict = "partial"
        round_score = SCORE_PARTIAL
    else:
        verdict = "fail"
        round_score = SCORE_FAIL
    return {
        "word": word_ko,
        "llm_score": s,
        "comment": llm_comment,
        "verdict": verdict,
        "round_score": round_score,
        "used_api": used_api,
    }


# ============================================================
# 4. SilentState
# ============================================================
class SilentState:
    """W9 라운드 상태 (W4 RoundState와 유사 인터페이스)."""

    TOTAL_ROUNDS = TOTAL_ROUNDS
    WIN_THRESHOLD = WIN_THRESHOLD

    def __init__(self, difficulty: str = DIFFICULTY_NORMAL, seed=None):
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError(f"알 수 없는 난이도: {difficulty}")
        self.difficulty = difficulty
        self.config = DIFFICULTY_CONFIG[difficulty]
        self._word_gen = WordGenerator(seed=seed)
        self.reset()

    def reset(self):
        self.round_index = 0
        self.outcomes = []
        self.total_score = 0
        self.correct_count = 0
        self.partial_count = 0
        self.fail_count = 0
        self.api_used_count = 0      # LLM 호출 성공 수
        self.current_word = None
        self._end_reason = END_NONE

    # ----------------------------------------------------------
    # 출제
    # ----------------------------------------------------------
    def next_word(self):
        """다음 라운드 단어 (직전 회피)."""
        word = self._word_gen.next_word()
        self.current_word = word
        return word

    # ----------------------------------------------------------
    # 라운드 결과 적용
    # ----------------------------------------------------------
    def apply_round(self, llm_score: int, llm_comment: str,
                    used_api: bool = True) -> dict:
        """LLM 평가 결과를 적용. current_word 사용.

        Raises:
            RuntimeError: current_word가 없거나 이미 5라운드 완료
        """
        if self.current_word is None:
            raise RuntimeError("next_word를 먼저 호출하세요.")
        if self.round_index >= self.TOTAL_ROUNDS:
            raise RuntimeError(
                f"이미 {self.TOTAL_ROUNDS}라운드 완료."
            )
        outcome = make_outcome(
            word_ko=self.current_word.ko,
            llm_score=llm_score,
            llm_comment=llm_comment,
            correct_threshold=self.config["correct"],
            partial_threshold=self.config["partial"],
            used_api=used_api,
        )
        self.outcomes.append(outcome)
        self.total_score += outcome["round_score"]
        if outcome["verdict"] == "correct":
            self.correct_count += 1
        elif outcome["verdict"] == "partial":
            self.partial_count += 1
        else:
            self.fail_count += 1
        if used_api:
            self.api_used_count += 1
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
    def get_express_time(self) -> float:
        return self.config["express_time"]

    def get_correct_threshold(self) -> int:
        return self.config["correct"]

    def get_partial_threshold(self) -> int:
        return self.config["partial"]

    def get_max_score(self) -> int:
        return self.TOTAL_ROUNDS * SCORE_CORRECT  # 500

    def get_summary(self) -> dict:
        return {
            "difficulty":   self.difficulty,
            "round_index":  self.round_index,
            "total_rounds": self.TOTAL_ROUNDS,
            "score":        self.total_score,
            "max_score":    self.get_max_score(),
            "correct":      self.correct_count,
            "partial":      self.partial_count,
            "fail":         self.fail_count,
            "win_threshold": self.WIN_THRESHOLD,
            "api_used":     self.api_used_count,
            "end_reason":   self.check_end(),
            "current_word_ko": self.current_word.ko if self.current_word else None,
        }

    def __repr__(self):
        return (
            f"SilentState(diff={self.difficulty}, "
            f"round={self.round_index}/{self.TOTAL_ROUNDS}, "
            f"score={self.total_score}, "
            f"correct={self.correct_count})"
        )
