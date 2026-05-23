"""
game.py — W9 고요 속의 외침 (Silent Charades)
================================================

PlayWait W9 — Doby가 출제한 한국어 단어를 손짓·몸짓으로 표현. 마지막 frame을
Claude API(Vision)로 평가받아 0~100점. 5라운드 / 4정답 이상 = 승리.

기술 스택
---------
- MediaPipe Pose + Hands (시각 피드백용 — 매 프레임 추론, 평가는 LLM)
- silent_words: 한국어 단어 풀 (순수)
- silent_llm: Claude Haiku 4.5 + Vision (또는 fallback)
- silent_state: 5라운드 진행 (순수)

W8 인프라 재활용
-----------------
- theme.py / ui_renderer.py / sound_manager.py + W9 단어 카드 UI
- 안전 종료 4중 방어 + Pose/Hands close

핵심 결정
---------
- **LLM은 EVALUATING 페이즈에서 차단 호출** (~2~3초). 다른 페이즈에 영향 X.
- **lazy import 0건** (W4 트러블 #16). 단, anthropic SDK는 silent_llm 내부에서만
  필요하므로 game.py는 silent_llm.evaluate_expression만 import.
- **모듈명 prefix**: silent_* (W5 트러블 #17)

실행
----
    cd ~/PlayWait
    # API 키 있으면 (권장)
    ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python -m games.05_silent_charades.src.game
    # 없어도 동작 (fallback 모드)
    .venv/bin/python -m games.05_silent_charades.src.game

조작
----
    [1/2/3]  난이도 / [SPACE] 시작 / [R] 재시작 / [Q/ESC] 종료
    [F]      전체 화면 / [+/-/0] 화면 크기

Author: Stephen (gjkong)
Date: 2026-05-12 (W9 Step 5)
"""

import atexit
import os
import signal
import sys
import time
from enum import Enum

import cv2
import mediapipe as mp

# ============================================================
# 모듈 import — 파일 상단 한 곳 (W4 트러블 #16)
# ============================================================
try:
    from .silent_words import WordGenerator
    from .silent_llm import evaluate_expression
    from .silent_state import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        END_NONE,
        END_WIN,
        SilentState,
    )
    from .sound_manager import SoundManager
    from . import theme
    from . import ui_renderer as ui
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from silent_words import WordGenerator
    from silent_llm import evaluate_expression
    from silent_state import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        END_NONE,
        END_WIN,
        SilentState,
    )
    from sound_manager import SoundManager
    import theme
    import ui_renderer as ui


# ============================================================
# MediaPipe
# ============================================================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


# ============================================================
# 페이즈 (W4 패턴 + EVALUATING)
# ============================================================
class GamePhase(Enum):
    DIFFICULTY_SELECT = "difficulty_select"
    READY = "ready"
    SHOW_WORD = "show_word"
    COUNTDOWN = "countdown"
    EXPRESS = "express"
    EVALUATING = "evaluating"
    ROUND_RESULT = "round_result"
    GAME_OVER = "game_over"


SHOW_WORD_DURATION = 2.0
COUNTDOWN_DURATION = 3.0
ROUND_RESULT_DURATION = 3.0
MIN_EVALUATING_DISPLAY = 0.8  # LLM이 너무 빨라도 spinner 보여주는 최소 시간


# ============================================================
# 메인 게임
# ============================================================
class SilentCharadesGame:

    def __init__(self):
        self.cap = None
        self.pose = None
        self.hands = None

        self.difficulty = DIFFICULTY_NORMAL
        self.state = SilentState(difficulty=self.difficulty)
        self.phase = GamePhase.DIFFICULTY_SELECT

        self.phase_start_time = 0.0
        self.last_outcome = None
        self.last_frame_bgr = None     # EVALUATING에서 LLM에 보낼 frame
        self.has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

        current_dir = os.path.dirname(os.path.abspath(__file__))
        sounds_dir = os.path.join(
            os.path.dirname(current_dir), "assets", "sounds"
        )
        self.sound = SoundManager(sounds_dir)

        self.window_name = "PlayWait - Silent Charades"
        self.display_scale = theme.DISPLAY_SCALE
        self.is_fullscreen = False
        self.fps = 0.0
        self.prev_fps_time = time.time()

    # ----------------------------------------
    # 초기화
    # ----------------------------------------
    def setup(self) -> bool:
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
            print("   sudo fuser -k /dev/video0")
            self.cap = None
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, theme.SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, theme.SCREEN_HEIGHT)
        ok, _ = self.cap.read()
        if not ok:
            print("❌ 카메라 프레임을 읽을 수 없습니다.")
            self.cap.release()
            self.cap = None
            return False

        self.pose = mp_pose.Pose(
            model_complexity=0,        # 가벼움 — Pose는 시각 피드백용
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._apply_window_size()
        return True

    # ----------------------------------------
    # 키 입력
    # ----------------------------------------
    def handle_key(self, key: int) -> bool:
        if key == ord('q') or key == 27:
            return False

        if key in (ord('f'), ord('F')):
            self._toggle_fullscreen()
            return True
        if key in (ord('+'), ord('=')):
            self.display_scale = min(3.0, self.display_scale + 0.25)
            self._apply_window_size()
            print(f"🔍 화면: {self.display_scale:.2f}x")
            return True
        if key in (ord('-'), ord('_')):
            self.display_scale = max(0.5, self.display_scale - 0.25)
            self._apply_window_size()
            print(f"🔍 화면: {self.display_scale:.2f}x")
            return True
        if key == ord('0'):
            self.display_scale = theme.DISPLAY_SCALE
            self.is_fullscreen = False
            self._apply_window_size()
            return True

        if self.phase == GamePhase.DIFFICULTY_SELECT:
            if key in (ord('1'), ord('2'), ord('3')):
                diff_map = {
                    ord('1'): DIFFICULTY_EASY,
                    ord('2'): DIFFICULTY_NORMAL,
                    ord('3'): DIFFICULTY_HARD,
                }
                self.difficulty = diff_map[key]
                self.state = SilentState(difficulty=self.difficulty)
                self.last_outcome = None
                self.sound.play("click")
                print(
                    f"🎮 난이도: {theme.DIFFICULTY_KOREAN[self.difficulty]} | "
                    f"표현 {self.state.get_express_time():.0f}s, "
                    f"정답 임계 {self.state.get_correct_threshold()}%"
                )
                if not self.has_api_key:
                    print("⚠️  ANTHROPIC_API_KEY 환경변수 없음 — fallback 모드로 동작합니다.")
                self.phase = GamePhase.READY

        elif self.phase == GamePhase.READY:
            if key == ord(' '):
                self.sound.play("click")
                self._enter_show_word()

        elif self.phase == GamePhase.GAME_OVER:
            if key in (ord('r'), ord('R')):
                self.sound.play("click")
                self.state.reset()
                self.last_outcome = None
                self.last_frame_bgr = None
                print("\n🔄 게임 재시작")
                self.phase = GamePhase.DIFFICULTY_SELECT

        return True

    # ----------------------------------------
    # 페이즈 전이
    # ----------------------------------------
    def _enter_show_word(self):
        word = self.state.next_word()
        self.phase = GamePhase.SHOW_WORD
        self.phase_start_time = time.time()
        self.sound.play("reveal")
        print(
            f"\n=== 라운드 {self.state.round_index + 1}"
            f"/{self.state.TOTAL_ROUNDS} — 단어: {word.ko} ==="
        )

    def _enter_countdown(self):
        self.phase = GamePhase.COUNTDOWN
        self.phase_start_time = time.time()
        self.sound.play("countdown")

    def _enter_express(self):
        self.phase = GamePhase.EXPRESS
        self.phase_start_time = time.time()

    def _enter_evaluating(self, frame_bgr):
        # frame을 캡처 — 평가에 사용
        self.last_frame_bgr = frame_bgr.copy()
        self.phase = GamePhase.EVALUATING
        self.phase_start_time = time.time()

    def _do_evaluation(self):
        """블록킹 LLM 호출. EVALUATING 페이즈 안에서 한 번만 수행."""
        # frame → JPEG bytes
        ok, jpeg_buf = cv2.imencode(".jpg", self.last_frame_bgr,
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            # 인코딩 실패 → fallback
            from silent_llm import _fallback_eval, API_ERROR_COMMENT
            result = _fallback_eval(reason=API_ERROR_COMMENT)
        else:
            print("  🤖 Doby 평가 중...")
            t0 = time.time()
            result = evaluate_expression(
                word=self.state.current_word.ko,
                image_bytes=jpeg_buf.tobytes(),
            )
            t_eval = time.time() - t0
            api_str = "API" if result.used_api else "FALLBACK"
            print(f"  → [{api_str}] {result.score}점  "
                  f"({t_eval:.1f}s) — {result.comment}")

        outcome = self.state.apply_round(
            llm_score=result.score,
            llm_comment=result.comment,
            used_api=result.used_api,
        )
        self.last_outcome = outcome

        sound_key = theme.SILENT_VERDICT_SOUND.get(
            outcome["verdict"], "lose",
        )
        self.sound.play(sound_key)
        print(
            f"   결과: {theme.SILENT_VERDICT_KOREAN.get(outcome['verdict'], '?')} "
            f"(+{outcome['round_score']}점)  →  누적 {self.state.total_score}점, "
            f"정답 {self.state.correct_count}/{self.state.WIN_THRESHOLD}"
        )

    def _enter_round_result(self):
        self.phase = GamePhase.ROUND_RESULT
        self.phase_start_time = time.time()

    def _enter_game_over(self):
        if self.state.is_win():
            self.sound.play("victory")
            print("\n" + "=" * 40)
            print(f"🏆 우승! 정답 {self.state.correct_count}/5  "
                  f"점수 {self.state.total_score}/"
                  f"{self.state.get_max_score()}")
            print("=" * 40)
        else:
            self.sound.play("lose")
            print("\n" + "=" * 40)
            print(f"💔 도전 부족: 정답 {self.state.correct_count}/5  "
                  f"점수 {self.state.total_score}/"
                  f"{self.state.get_max_score()}")
            print("=" * 40)
        self.phase = GamePhase.GAME_OVER

    # ----------------------------------------
    # 페이즈 업데이트
    # ----------------------------------------
    def update_phase(self, frame_bgr):
        now = time.time()
        elapsed = now - self.phase_start_time

        if self.phase == GamePhase.SHOW_WORD:
            if elapsed >= SHOW_WORD_DURATION:
                self._enter_countdown()

        elif self.phase == GamePhase.COUNTDOWN:
            if elapsed >= COUNTDOWN_DURATION:
                self._enter_express()

        elif self.phase == GamePhase.EXPRESS:
            total = self.state.get_express_time()
            if elapsed >= total:
                # 표현 시간 끝 → 현재 frame을 캡처해서 평가
                self._enter_evaluating(frame_bgr)

        elif self.phase == GamePhase.EVALUATING:
            # 평가는 한 번만 수행 (last_outcome으로 가드)
            if self.last_outcome is None or (
                self.state.outcomes
                and self.state.outcomes[-1] is not self.last_outcome
            ):
                pass
            if self.last_outcome is None:
                # 첫 진입 → 블록킹 호출
                self._do_evaluation()
                # spinner 최소 시간 확보를 위해 즉시 결과로 전이하지 않음
                # → MIN_EVALUATING_DISPLAY 경과 후 ROUND_RESULT
            elif elapsed >= MIN_EVALUATING_DISPLAY:
                self._enter_round_result()

        elif self.phase == GamePhase.ROUND_RESULT:
            if elapsed >= ROUND_RESULT_DURATION:
                if self.state.check_end() != END_NONE:
                    self._enter_game_over()
                else:
                    self.last_outcome = None  # reset for next round
                    self._enter_show_word()

    # ----------------------------------------
    # 렌더
    # ----------------------------------------
    def render(self, frame, pose_lm, hands_results):
        ui.begin_frame()
        h, w, _ = frame.shape

        if self.phase == GamePhase.DIFFICULTY_SELECT:
            ui.draw_silent_difficulty_select(frame)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        if self.phase == GamePhase.READY:
            ui.draw_silent_ready(frame, has_api_key=self.has_api_key)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        # 공통 HUD (SHOW_WORD 이후)
        ui.draw_silent_score_bar(
            frame,
            round_index=self.state.round_index + 1,
            total_rounds=self.state.TOTAL_ROUNDS,
            score=self.state.total_score,
            max_score=self.state.get_max_score(),
            correct_count=self.state.correct_count,
        )

        if self.phase == GamePhase.SHOW_WORD:
            ui.draw_silent_word_card(
                frame,
                word_ko=self.state.current_word.ko,
                hint=self.state.current_word.hint,
            )

        elif self.phase == GamePhase.COUNTDOWN:
            now = time.time()
            elapsed = now - self.phase_start_time
            count = max(1, int(COUNTDOWN_DURATION - elapsed) + 1)
            count = min(count, 3)
            # 카운트다운 큰 숫자 + 상단 단어
            cx_n = w // 2
            cy_n = h // 2
            overlay = frame.copy()
            cv2.circle(overlay, (cx_n, cy_n), 75, theme.COLOR_BLACK, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            ui.draw_text_korean(frame, str(count), (cx_n, cy_n),
                                 90, theme.PINKLAB_PINK, anchor="mm")
            ui.draw_text_korean(frame,
                                 f"미션: {self.state.current_word.ko}",
                                 (cx_n, theme.HEADER_HEIGHT + 20),
                                 theme.FONT_SIZE_MEDIUM,
                                 theme.PINKLAB_PINK_LIGHT, anchor="mm")

        elif self.phase == GamePhase.EXPRESS:
            # Pose + Hands landmark 가벼운 점 표시
            if pose_lm is not None:
                ui.draw_sync_pose_landmarks(
                    frame, pose_lm, w, h,
                    side_color=(180, 130, 230),
                )
            if hands_results and hands_results.multi_hand_landmarks:
                for hand_lm in hands_results.multi_hand_landmarks:
                    for lm in hand_lm.landmark:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 3,
                                   theme.PINKLAB_PINK_LIGHT, -1, cv2.LINE_AA)

            now = time.time()
            elapsed = now - self.phase_start_time
            total = self.state.get_express_time()
            remaining = max(0.0, total - elapsed)
            ui.draw_silent_express_overlay(
                frame,
                word_ko=self.state.current_word.ko,
                remaining=remaining,
                total=total,
            )

        elif self.phase == GamePhase.EVALUATING:
            t = time.time() - self.phase_start_time
            ui.draw_silent_evaluating(frame, t)

        elif self.phase == GamePhase.ROUND_RESULT:
            if self.last_outcome is not None:
                ui.draw_silent_round_result(frame, self.last_outcome)

        elif self.phase == GamePhase.GAME_OVER:
            ui.draw_silent_game_over(
                frame,
                win=self.state.is_win(),
                summary=self.state.get_summary(),
            )

        ui.draw_fps(frame, self.fps)
        ui.flush_text(frame)
        return frame

    # ----------------------------------------
    # 화면 크기
    # ----------------------------------------
    def _toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            cv2.setWindowProperty(self.window_name,
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            print("🖥️  전체 화면 ON")
        else:
            cv2.setWindowProperty(self.window_name,
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL)
            self._apply_window_size()
            print("🖥️  전체 화면 OFF")

    def _apply_window_size(self):
        if self.is_fullscreen:
            return
        w = int(theme.SCREEN_WIDTH * self.display_scale)
        h = int(theme.SCREEN_HEIGHT * self.display_scale)
        cv2.resizeWindow(self.window_name, w, h)

    # ----------------------------------------
    # 메인 루프
    # ----------------------------------------
    def run(self):
        if not self.setup():
            return

        print("\n" + "=" * 50)
        print("🤫 PlayWait W9 - 고요 속의 외침 (Silent Charades)")
        print("=" * 50)
        print("\n[게임 조작]")
        print("  [1/2/3] 난이도 (쉬움/보통/어려움)")
        print("  [SPACE] 게임 시작")
        print("  [R]     재시작")
        print("  [Q/ESC] 종료")
        print("\n[화면 조작]")
        print("  [F] 전체 화면 / [+] 확대 / [-] 축소 / [0] 리셋")
        print()
        if self.has_api_key:
            print("✅ Claude AI 평가 활성화 (ANTHROPIC_API_KEY 감지)")
        else:
            print("⚠️  ANTHROPIC_API_KEY 없음 — fallback 모드로 동작")
        print()

        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Pose + Hands (EXPRESS 페이즈에서만 시각 피드백)
                pose_lm = None
                hands_results = None
                if self.phase == GamePhase.EXPRESS:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    pose_res = self.pose.process(rgb)
                    hands_results = self.hands.process(rgb)
                    if pose_res.pose_landmarks:
                        pose_lm = pose_res.pose_landmarks.landmark

                # 페이즈 업데이트 (EVALUATING은 블록킹)
                self.update_phase(frame)

                now = time.time()
                fps_dt = now - self.prev_fps_time
                if fps_dt > 0:
                    self.fps = 1.0 / fps_dt
                self.prev_fps_time = now

                frame = self.render(frame, pose_lm, hands_results)

                if self.is_fullscreen:
                    cv2.imshow(self.window_name, frame)
                else:
                    if self.display_scale != 1.0:
                        display_frame = cv2.resize(
                            frame,
                            (int(theme.SCREEN_WIDTH * self.display_scale),
                             int(theme.SCREEN_HEIGHT * self.display_scale)),
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        display_frame = frame
                    cv2.imshow(self.window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break

        except KeyboardInterrupt:
            print("\n⚠️  사용자가 Ctrl+C로 중단")
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    # ----------------------------------------
    # 안전 종료
    # ----------------------------------------
    def cleanup(self):
        print("\n🧹 리소스 정리 중...")

        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
                    print("   ✓ 카메라 해제")
            except Exception:
                pass
            finally:
                self.cap = None

        if self.pose is not None:
            try:
                self.pose.close()
                print("   ✓ Pose 해제")
            except Exception:
                pass
            finally:
                self.pose = None

        if self.hands is not None:
            try:
                self.hands.close()
                print("   ✓ Hands 해제")
            except Exception:
                pass
            finally:
                self.hands = None

        if self.sound is not None:
            try:
                self.sound.cleanup()
                print("   ✓ 사운드 해제")
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
            for _ in range(4):
                cv2.waitKey(1)
            print("   ✓ 윈도우 닫음")
        except Exception:
            pass

        print("✅ 게임 종료 완료")


# ============================================================
# 시그널 + atexit
# ============================================================
_game_instance = None


def _signal_handler(signum, frame):
    print(f"\n⚠️  시그널: {signal.Signals(signum).name}")
    if _game_instance is not None:
        _game_instance.cleanup()
    sys.exit(0)


def _atexit_cleanup():
    if _game_instance is not None:
        _game_instance.cleanup()


def main():
    global _game_instance

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_atexit_cleanup)

    _game_instance = SilentCharadesGame()
    try:
        _game_instance.run()
    finally:
        _game_instance = None


if __name__ == "__main__":
    main()
