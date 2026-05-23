"""
game.py — W6 K-Pop 랜덤 댄스 (KPop Random Dance)
==================================================

PlayWait W6 — Doby가 시범 보이는 5종 포즈를 따라 추는 미러링 게임.
5라운드 / 4정답 이상 = 승리.

기술 스택
---------
- MediaPipe Pose (33 landmark, model_complexity=1)
- pose_features.extract_pose_features() — 각도/비율 (순수)
- pose_classifier.pose_similarity() — profile + min 집계 (순수)
- dance_state.DanceState — 5라운드 진행 (순수, prefix 적용)

W1+W2+W3+W4+W5 인프라 재활용
-----------------------------
- theme.py        (PinkLAB + W6 5종 포즈 토큰)
- ui_renderer.py  (한글 PIL + 친근 도형 헬퍼 + Doby stick figure)
- sound_manager.py
- 안전 종료 4중 방어

핵심 설계 결정
---------------
- **lazy import 0건** (W4 트러블 #16 학습)
- **모듈명 prefix**: pose_features / pose_classifier / dance_state (W5 트러블 #17)
- 측정 중 best accuracy를 라운드 결과로 채택 + EMA(0.35)로 게이지 보간 (W4 결정)

실행
----
    cd ~/PlayWait
    .venv/bin/python -m games.04_kpop_dance.src.game

조작
----
    [1/2/3]  난이도 (쉬움/보통/어려움)
    [SPACE]  게임 시작
    [R]      재시작 (게임 종료 후)
    [Q/ESC]  종료
    [F]      전체 화면 / [+] 확대 / [-] 축소 / [0] 리셋

Author: Stephen (gjkong)
Date: 2026-05-12 (W6 Step 4)
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
# 모듈 import — 한 곳 집중 (W4 트러블 #16 학습)
# ============================================================
try:
    from .pose_features import extract_pose_features
    from .pose_classifier import (
        GAME_POSES,
        POSE_NEUTRAL,
        classify_pose,
        pose_similarity,
    )
    from .dance_state import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        DanceState,
        END_NONE,
        END_WIN,
    )
    from .sound_manager import SoundManager
    from . import theme
    from . import ui_renderer as ui
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pose_features import extract_pose_features
    from pose_classifier import (
        GAME_POSES,
        POSE_NEUTRAL,
        classify_pose,
        pose_similarity,
    )
    from dance_state import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        DanceState,
        END_NONE,
        END_WIN,
    )
    from sound_manager import SoundManager
    import theme
    import ui_renderer as ui


# ============================================================
# MediaPipe Pose
# ============================================================
mp_pose = mp.solutions.pose


# ============================================================
# 페이즈 (W4 표정 미러링과 동일 7페이즈)
# ============================================================
class GamePhase(Enum):
    DIFFICULTY_SELECT = "difficulty_select"
    READY = "ready"
    SHOW_DOBY = "show_doby"
    COUNTDOWN = "countdown"
    MEASURE = "measure"
    ROUND_RESULT = "round_result"
    GAME_OVER = "game_over"


SHOW_DOBY_DURATION = 1.5
COUNTDOWN_DURATION = 3.0
ROUND_RESULT_DURATION = 2.0
ACCURACY_EMA_ALPHA = 0.35


# ============================================================
# 메인 게임
# ============================================================
class KPopDanceGame:

    def __init__(self):
        self.cap = None
        self.pose = None

        self.difficulty = DIFFICULTY_NORMAL
        self.state = DanceState(difficulty=self.difficulty)
        self.phase = GamePhase.DIFFICULTY_SELECT

        self.current_target = None
        self.phase_start_time = 0.0
        self.measure_best_accuracy = 0.0
        self.measure_smoothed_accuracy = 0.0
        self.measure_frames = 0
        self.last_outcome = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        sounds_dir = os.path.join(
            os.path.dirname(current_dir), "assets", "sounds"
        )
        self.sound = SoundManager(sounds_dir)

        self.window_name = "PlayWait - KPop Dance"
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
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
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
                self.state = DanceState(difficulty=self.difficulty)
                self.sound.play("click")
                print(
                    f"🎮 난이도: {theme.DIFFICULTY_KOREAN[self.difficulty]} "
                    f"(측정 {self.state.get_measure_time():.1f}s, "
                    f"정답 임계 {int(self.state.get_correct_threshold()*100)}%)"
                )
                self.phase = GamePhase.READY

        elif self.phase == GamePhase.READY:
            if key == ord(' '):
                self.sound.play("click")
                self._enter_show_doby()

        elif self.phase == GamePhase.GAME_OVER:
            if key in (ord('r'), ord('R')):
                self.sound.play("click")
                self.state.reset()
                self.last_outcome = None
                self.current_target = None
                print("\n🔄 게임 재시작")
                self.phase = GamePhase.DIFFICULTY_SELECT

        return True

    # ----------------------------------------
    # 페이즈 전이
    # ----------------------------------------
    def _enter_show_doby(self):
        self.current_target = self.state.next_target()
        self.phase = GamePhase.SHOW_DOBY
        self.phase_start_time = time.time()
        self.measure_best_accuracy = 0.0
        self.measure_smoothed_accuracy = 0.0
        self.measure_frames = 0
        self.sound.play("reveal")
        print(
            f"\n=== 라운드 {self.state.round_index + 1}"
            f"/{self.state.TOTAL_ROUNDS} — "
            f"목표: {theme.POSE_KOREAN.get(self.current_target, '??')} ==="
        )

    def _enter_countdown(self):
        self.phase = GamePhase.COUNTDOWN
        self.phase_start_time = time.time()
        self.sound.play("countdown")

    def _enter_measure(self):
        self.phase = GamePhase.MEASURE
        self.phase_start_time = time.time()
        self.measure_best_accuracy = 0.0
        self.measure_smoothed_accuracy = 0.0
        self.measure_frames = 0

    def _finalize_round(self):
        accuracy = self.measure_best_accuracy
        outcome = self.state.apply_round(
            target=self.current_target, accuracy=accuracy,
        )
        self.last_outcome = outcome
        sound_key = theme.DANCE_VERDICT_SOUND.get(outcome["verdict"], "lose")
        self.sound.play(sound_key)
        print(
            f"   결과: {theme.DANCE_VERDICT_KOREAN.get(outcome['verdict'], '?')} "
            f"(정확도 {int(outcome['accuracy']*100)}%, +{outcome['score']}점) "
            f"  → 누적 {self.state.total_score}점, "
            f"정답 {self.state.correct_count}/{self.state.WIN_THRESHOLD}"
        )
        self.phase = GamePhase.ROUND_RESULT
        self.phase_start_time = time.time()

    def _enter_game_over(self):
        win = self.state.is_win()
        if win:
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
    def update_phase(self, pose_landmarks):
        now = time.time()
        elapsed = now - self.phase_start_time

        if self.phase == GamePhase.SHOW_DOBY:
            if elapsed >= SHOW_DOBY_DURATION:
                self._enter_countdown()

        elif self.phase == GamePhase.COUNTDOWN:
            if elapsed >= COUNTDOWN_DURATION:
                self._enter_measure()

        elif self.phase == GamePhase.MEASURE:
            if pose_landmarks is not None:
                try:
                    features = extract_pose_features(pose_landmarks)
                    acc = pose_similarity(self.current_target, features)
                except (IndexError, ValueError):
                    acc = 0.0
            else:
                acc = 0.0

            if acc > self.measure_best_accuracy:
                self.measure_best_accuracy = acc
            if self.measure_frames == 0:
                self.measure_smoothed_accuracy = acc
            else:
                self.measure_smoothed_accuracy = (
                    ACCURACY_EMA_ALPHA * acc
                    + (1.0 - ACCURACY_EMA_ALPHA)
                    * self.measure_smoothed_accuracy
                )
            self.measure_frames += 1

            total = self.state.get_measure_time()
            if elapsed >= total:
                self._finalize_round()

        elif self.phase == GamePhase.ROUND_RESULT:
            if elapsed >= ROUND_RESULT_DURATION:
                if self.state.check_end() != END_NONE:
                    self._enter_game_over()
                else:
                    self._enter_show_doby()

    # ----------------------------------------
    # 렌더
    # ----------------------------------------
    def render(self, frame, pose_landmarks):
        ui.begin_frame()
        h, w, _ = frame.shape

        if self.phase == GamePhase.DIFFICULTY_SELECT:
            ui.draw_dance_difficulty_select(frame)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        if self.phase == GamePhase.READY:
            ui.draw_dance_ready(frame)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        ui.draw_dance_score_bar(
            frame,
            round_index=self.state.round_index + 1,
            total_rounds=self.state.TOTAL_ROUNDS,
            score=self.state.total_score,
            max_score=self.state.get_max_score(),
            correct_count=self.state.correct_count,
        )

        if self.phase == GamePhase.SHOW_DOBY:
            ui.draw_dance_show_doby(frame, self.current_target)

        elif self.phase == GamePhase.COUNTDOWN:
            now = time.time()
            elapsed = now - self.phase_start_time
            count = max(1, int(COUNTDOWN_DURATION - elapsed) + 1)
            count = min(count, 3)
            ui.draw_dance_countdown(frame, count, self.current_target)

        elif self.phase == GamePhase.MEASURE:
            if pose_landmarks is not None:
                ui.draw_dance_pose_landmarks(frame, pose_landmarks, w, h)

            now = time.time()
            elapsed = now - self.phase_start_time
            total = self.state.get_measure_time()
            remaining = max(0.0, total - elapsed)
            ui.draw_dance_measure(
                frame,
                self.current_target,
                self.measure_smoothed_accuracy,
                remaining, total,
                self.state.get_correct_threshold(),
                self.state.get_partial_threshold(),
            )

        elif self.phase == GamePhase.ROUND_RESULT:
            if self.last_outcome is not None:
                ui.draw_dance_round_result(frame, self.last_outcome)

        elif self.phase == GamePhase.GAME_OVER:
            ui.draw_dance_game_over(
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
        print("💃 PlayWait W6 - K-Pop 댄스")
        print("=" * 50)
        print("\n[게임 조작]")
        print("  [1/2/3] 난이도 (쉬움/보통/어려움)")
        print("  [SPACE] 게임 시작")
        print("  [R]     재시작")
        print("  [Q/ESC] 종료")
        print("\n[화면 조작]")
        print("  [F] 전체 화면 / [+] 확대 / [-] 축소 / [0] 리셋")
        print()

        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.pose.process(rgb)
                rgb.flags.writeable = True

                pose_landmarks = None
                if results.pose_landmarks:
                    pose_landmarks = results.pose_landmarks.landmark

                self.update_phase(pose_landmarks)

                now = time.time()
                fps_dt = now - self.prev_fps_time
                if fps_dt > 0:
                    self.fps = 1.0 / fps_dt
                self.prev_fps_time = now

                frame = self.render(frame, pose_landmarks)

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

    _game_instance = KPopDanceGame()
    try:
        _game_instance.run()
    finally:
        _game_instance = None


if __name__ == "__main__":
    main()
