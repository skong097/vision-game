"""
game.py — W8 커플 싱크 (Couple Sync)
======================================

PlayWait W8 — 두 사람이 같은 미션 포즈를 동시에 유지하면 점수. 발렌타인 시즌.
60초 / 미션 회전 / 목표 점수 달성 시 승리.

기술 스택
---------
- MediaPipe Pose × 2 (화면 좌/우 분할에 각각 추론)
- pose_features.extract_pose_features() — 각도/비율 (순수, W6 재활용)
- pose_classifier.pose_similarity() — profile + min 집계 (순수, W6 재활용)
- sync_judge.judge_sync() — 두 features 동시 매칭 (순수)
- sync_state.SyncState — 미션 회전 + 유지 시간 + 점수 (순수, time_provider 주입 가능)
- couple_detector — 좌/우 분할 + landmark 좌표 변환 (순수)

W7 인프라 재활용
-----------------
- theme.py        (PinkLAB + W8 발렌타인 토큰)
- ui_renderer.py  (한글 PIL + 친근 도형 헬퍼 + 하트 + 듀얼 박스)
- sound_manager.py
- 안전 종료 4중 방어 (pose.close() 2번)

핵심 설계 결정
---------------
- **lazy import 0건** (W4 트러블 #16)
- **모듈명 prefix**: sync_*, couple_* (W5 트러블 #17)
- W6 pose_features/classifier는 src/로 복사 — core 추출 시점에 통합
- Pose 두 인스턴스 (좌/우 각각) — FPS는 다소 줄지만 멀티 인물 지원

실행
----
    cd ~/PlayWait
    .venv/bin/python -m games.10_couple_sync.src.game

조작
----
    [1/2/3]  난이도 / [SPACE] 시작 / [R] 재시작 / [Q/ESC] 종료
    [F]      전체 화면 / [+/-/0] 화면 크기

Author: Stephen (gjkong)
Date: 2026-05-12 (W8 Step 5)
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
    from .pose_features import extract_pose_features
    from .pose_classifier import classify_pose
    from .sync_judge import judge_sync
    from .sync_state import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        END_NONE,
        END_TIMEOUT,
        END_WIN,
        SyncState,
    )
    from .couple_detector import (
        SIDE_LEFT,
        SIDE_RIGHT,
        remap_landmarks,
        split_frame_x,
    )
    from .sound_manager import SoundManager
    from . import theme
    from . import ui_renderer as ui
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pose_features import extract_pose_features
    from pose_classifier import classify_pose
    from sync_judge import judge_sync
    from sync_state import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        END_NONE,
        END_TIMEOUT,
        END_WIN,
        SyncState,
    )
    from couple_detector import (
        SIDE_LEFT,
        SIDE_RIGHT,
        remap_landmarks,
        split_frame_x,
    )
    from sound_manager import SoundManager
    import theme
    import ui_renderer as ui


# ============================================================
# MediaPipe Pose
# ============================================================
mp_pose = mp.solutions.pose


# ============================================================
# 페이즈
# ============================================================
class GamePhase(Enum):
    DIFFICULTY_SELECT = "difficulty_select"
    READY = "ready"
    PLAYING = "playing"
    GAME_OVER = "game_over"


# ============================================================
# 메인 게임
# ============================================================
class CoupleSyncGame:

    def __init__(self):
        self.cap = None
        self.pose_left = None
        self.pose_right = None

        self.difficulty = DIFFICULTY_NORMAL
        self.state = SyncState(difficulty=self.difficulty)
        self.phase = GamePhase.DIFFICULTY_SELECT
        self.last_frame_time = time.time()

        # 마지막 판정 결과 캐시 (렌더에서 하트 등 시각 피드백용)
        self.last_sync_result = None
        # 가장 최근 분류된 포즈 (좌/우)
        self.last_left_pose = None
        self.last_right_pose = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        sounds_dir = os.path.join(
            os.path.dirname(current_dir), "assets", "sounds"
        )
        self.sound = SoundManager(sounds_dir)

        self.window_name = "PlayWait - Couple Sync"
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

        # Pose 인스턴스 2개 — 좌/우 ROI 각각
        self.pose_left = mp_pose.Pose(
            model_complexity=0,        # 듀얼이라 0으로 (FPS 우선)
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_right = mp_pose.Pose(
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
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
                self.state = SyncState(difficulty=self.difficulty)
                self.last_sync_result = None
                self.sound.play("click")
                print(
                    f"🎮 난이도: {theme.DIFFICULTY_KOREAN[self.difficulty]} | "
                    f"매칭 임계 {int(self.state.get_match_threshold()*100)}%, "
                    f"유지 {self.state.get_hold_required():.1f}s, "
                    f"목표 {self.state.target_score}점"
                )
                self.phase = GamePhase.READY

        elif self.phase == GamePhase.READY:
            if key == ord(' '):
                self.sound.play("click")
                self.state.start_game()
                self.last_frame_time = time.time()
                self.phase = GamePhase.PLAYING
                kor = theme.POSE_KOREAN.get(self.state.current_mission, "??")
                print(f"\n=== 시작! 첫 미션: {kor} ===")

        elif self.phase == GamePhase.GAME_OVER:
            if key in (ord('r'), ord('R')):
                self.sound.play("click")
                self.state.reset()
                self.last_sync_result = None
                self.last_left_pose = None
                self.last_right_pose = None
                print("\n🔄 게임 재시작")
                self.phase = GamePhase.DIFFICULTY_SELECT

        return True

    # ----------------------------------------
    # PLAYING 갱신
    # ----------------------------------------
    def update_playing(self, dt, frame_bgr, frame_w, frame_h):
        """좌/우 ROI에 Pose 추론 → features → sync_judge → state 적용.

        Returns:
            (left_landmarks_full, right_landmarks_full) — render용
            (None일 수 있음)
        """
        half_x, _ = split_frame_x(frame_w)
        left_roi = frame_bgr[:, :half_x]
        right_roi = frame_bgr[:, half_x:]

        # MediaPipe RGB 전환
        left_rgb = cv2.cvtColor(left_roi, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_roi, cv2.COLOR_BGR2RGB)
        left_rgb.flags.writeable = False
        right_rgb.flags.writeable = False

        left_res = self.pose_left.process(left_rgb)
        right_res = self.pose_right.process(right_rgb)

        # ROI 좌표 → 전체 frame 좌표 변환
        left_lm_full = None
        right_lm_full = None
        if left_res.pose_landmarks:
            left_lm_full = remap_landmarks(
                left_res.pose_landmarks.landmark, SIDE_LEFT,
            )
        if right_res.pose_landmarks:
            right_lm_full = remap_landmarks(
                right_res.pose_landmarks.landmark, SIDE_RIGHT,
            )

        # features 추출
        left_features = None
        right_features = None
        if left_lm_full is not None:
            try:
                left_features = extract_pose_features(left_lm_full)
                self.last_left_pose, _ = classify_pose(left_features)
            except (IndexError, ValueError):
                left_features = None
                self.last_left_pose = None
        else:
            self.last_left_pose = None
        if right_lm_full is not None:
            try:
                right_features = extract_pose_features(right_lm_full)
                self.last_right_pose, _ = classify_pose(right_features)
            except (IndexError, ValueError):
                right_features = None
                self.last_right_pose = None
        else:
            self.last_right_pose = None

        # sync 판정
        sync_result = judge_sync(
            target_pose=self.state.current_mission,
            left_features=left_features,
            right_features=right_features,
            threshold=self.state.get_match_threshold(),
        )
        self.last_sync_result = sync_result

        # state 적용
        completed = self.state.apply_sync(dt, sync_result)
        if completed:
            self.sound.play(theme.SYNC_COMPLETION_SOUND)
            kor = theme.POSE_KOREAN.get(self.state.current_mission, "??")
            bonus_str = " (보너스!)" if sync_result.bonus else ""
            print(f"  ✨ 미션 완료{bonus_str}! 다음: {kor}  "
                  f"→ {self.state.score}/{self.state.target_score}점")

        # 종료 체크
        end = self.state.check_end()
        if end != END_NONE:
            if end == END_WIN:
                self.sound.play(theme.SYNC_VICTORY_SOUND)
                print(f"\n{'='*40}")
                print(f"🏆 성공! 점수 {self.state.score}/"
                      f"{self.state.target_score}, "
                      f"완료 {self.state.completions}회, "
                      f"보너스 {self.state.bonus_count}회")
                print(f"{'='*40}")
            else:
                self.sound.play(theme.SYNC_LOSE_SOUND)
                print(f"\n{'='*40}")
                print(f"💔 시간 초과: {self.state.score}/"
                      f"{self.state.target_score}점")
                print(f"{'='*40}")
            self.phase = GamePhase.GAME_OVER

        return left_lm_full, right_lm_full

    # ----------------------------------------
    # 렌더
    # ----------------------------------------
    def render(self, frame, left_lm, right_lm):
        ui.begin_frame()
        h, w, _ = frame.shape

        if self.phase == GamePhase.DIFFICULTY_SELECT:
            ui.draw_sync_difficulty_select(frame)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        if self.phase == GamePhase.READY:
            ui.draw_sync_ready(frame)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        # HUD (PLAYING / GAME_OVER 공통)
        mission_kor = theme.POSE_KOREAN.get(
            self.state.current_mission, "??",
        )
        ui.draw_sync_score_bar(
            frame,
            score=self.state.score,
            target=self.state.target_score,
            mission_korean=mission_kor,
            remaining=self.state.get_remaining(),
            completions=self.state.completions,
        )

        if self.phase == GamePhase.PLAYING:
            # 좌/우 분할 라벨 + 박스
            ui.draw_couple_split_overlay(frame)
            ui.draw_couple_person_box(
                frame, "left",
                has_pose=(left_lm is not None),
                pose_label=(theme.POSE_KOREAN.get(self.last_left_pose)
                            if self.last_left_pose else None),
            )
            ui.draw_couple_person_box(
                frame, "right",
                has_pose=(right_lm is not None),
                pose_label=(theme.POSE_KOREAN.get(self.last_right_pose)
                            if self.last_right_pose else None),
            )

            # 좌·우 landmark 점 (디버그 효과)
            if left_lm is not None:
                ui.draw_sync_pose_landmarks(
                    frame, left_lm, w, h,
                    side_color=theme.COUPLE_LEFT_COLOR,
                )
            if right_lm is not None:
                ui.draw_sync_pose_landmarks(
                    frame, right_lm, w, h,
                    side_color=theme.COUPLE_RIGHT_COLOR,
                )

            # 중앙 싱크 하트 (HUD 아래)
            heart_center = (w // 2, theme.HEADER_HEIGHT + 80)
            if self.last_sync_result is not None:
                ui.draw_sync_heart(
                    frame, heart_center,
                    both_match=self.last_sync_result.both_match,
                    hold_progress=self.state.get_hold_progress(),
                    bonus=self.last_sync_result.bonus,
                )
            else:
                ui.draw_sync_heart(frame, heart_center,
                                    both_match=False, hold_progress=0.0)

            # 하단 안내 — Doby hint
            hint = theme.POSE_HINT.get(self.state.current_mission, "")
            ui.draw_text_korean(
                frame, f"Doby: {mission_kor}!",
                (w // 2, h - 75),
                theme.FONT_SIZE_MEDIUM, theme.PINKLAB_PINK, anchor="mm",
            )
            ui.draw_text_korean(
                frame, hint, (w // 2, h - 40),
                theme.FONT_SIZE_NORMAL, theme.COLOR_WHITE, anchor="mm",
            )

        elif self.phase == GamePhase.GAME_OVER:
            ui.draw_sync_game_over(
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
        print("💕 PlayWait W8 - 커플 싱크 (Couple Sync)")
        print("=" * 50)
        print("\n[게임 조작]")
        print("  [1/2/3] 난이도 (쉬움/보통/어려움)")
        print("  [SPACE] 게임 시작")
        print("  [R]     재시작")
        print("  [Q/ESC] 종료")
        print("\n[화면 조작]")
        print("  [F] 전체 화면 / [+] 확대 / [-] 축소 / [0] 리셋")
        print("\nNOTE: 두 사람이 화면 좌·우에 한 명씩 자리잡아 주세요.")
        print()

        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                now = time.time()
                dt = now - self.last_frame_time
                self.last_frame_time = now

                left_lm = right_lm = None
                if self.phase == GamePhase.PLAYING:
                    left_lm, right_lm = self.update_playing(
                        dt, frame, w, h,
                    )

                fps_dt = now - self.prev_fps_time
                if fps_dt > 0:
                    self.fps = 1.0 / fps_dt
                self.prev_fps_time = now

                frame = self.render(frame, left_lm, right_lm)

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
    # 안전 종료 (Pose 두 인스턴스)
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

        if self.pose_left is not None:
            try:
                self.pose_left.close()
                print("   ✓ Pose(좌) 해제")
            except Exception:
                pass
            finally:
                self.pose_left = None
        if self.pose_right is not None:
            try:
                self.pose_right.close()
                print("   ✓ Pose(우) 해제")
            except Exception:
                pass
            finally:
                self.pose_right = None

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

    _game_instance = CoupleSyncGame()
    try:
        _game_instance.run()
    finally:
        _game_instance = None


if __name__ == "__main__":
    main()
