"""
game.py — W10 AR 보물찾기 (Hidden Menu Hunt) — 마지막 게임!
=============================================================

PlayWait W10 — 매장의 사물을 보물처럼 찾는 게임.
YOLO로 객체 검출 + 중앙 ROI에 1초 유지 → 발견 + AR 박스 강조.

기술 스택
---------
- core/vision/yolo_engine.py (W5 재활용) — duck-typed detector
- ar_overlay: 중앙 ROI + 매칭 검출 (순수 함수)
- treasure_clues / treasure_state — 출제 + 진행 (순수)

W9 인프라 재활용
----------------
- theme/ui_renderer/sound_manager (친근 도형 + AR 박스 추가)
- 안전 종료 4중 방어

핵심 설계 결정
---------------
- **lazy import 0건** (W4 트러블 #16)
- **모듈명 prefix**: treasure_*, ar_* (W5 트러블 #17)
- W5 yolo_engine 재활용 — 추가 의존성 X

실행
----
    cd ~/PlayWait
    .venv/bin/python -m games.03_hidden_menu_hunt.src.game

조작
----
    [1/2/3]  난이도 / [SPACE] 시작 / [R] 재시작 / [Q/ESC] 종료
    [F]      전체 화면 / [+/-/0] 화면 크기

Author: Stephen (gjkong)
Date: 2026-05-12 (W10 Step 5 — 마지막 게임!)
"""

import atexit
import os
import signal
import sys
import time
from enum import Enum

import cv2

# ============================================================
# 모듈 import — 파일 상단 한 곳 (W4 트러블 #16)
# ============================================================
try:
    from .treasure_clues import TreasureGenerator
    from .treasure_state import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        END_NONE,
        END_WIN,
        SCORE_PER_FIND,
        TreasureState,
    )
    from .ar_overlay import (
        collect_targets,
        compute_center_roi,
        find_target_in_roi,
    )
    from .sound_manager import SoundManager
    from . import theme
    from . import ui_renderer as ui
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from treasure_clues import TreasureGenerator
    from treasure_state import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        END_NONE,
        END_WIN,
        SCORE_PER_FIND,
        TreasureState,
    )
    from ar_overlay import (
        collect_targets,
        compute_center_roi,
        find_target_in_roi,
    )
    from sound_manager import SoundManager
    import theme
    import ui_renderer as ui

# YoloEngine — W5 재활용 (core/vision/yolo_engine.py)
try:
    from core.vision.yolo_engine import YoloEngine
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from core.vision.yolo_engine import YoloEngine


# ============================================================
# 페이즈 (W7 카페닌자 패턴 — 실시간 시뮬레이션)
# ============================================================
class GamePhase(Enum):
    DIFFICULTY_SELECT = "difficulty_select"
    READY = "ready"
    PLAYING = "playing"
    GAME_OVER = "game_over"


# 발견 직후 플래시 지속 시간(초)
FIND_FLASH_DURATION = 1.0


# ============================================================
# 메인 게임
# ============================================================
class HiddenMenuHuntGame:

    def __init__(self):
        self.cap = None
        self.engine = None

        self.difficulty = DIFFICULTY_NORMAL
        self.state = TreasureState(difficulty=self.difficulty)
        self.phase = GamePhase.DIFFICULTY_SELECT
        self.last_frame_time = time.time()

        # 발견 플래시 (W7 패턴)
        self.find_flash_until = 0.0
        self.last_find_ko = ""
        self.last_find_score = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        sounds_dir = os.path.join(
            os.path.dirname(current_dir), "assets", "sounds"
        )
        self.sound = SoundManager(sounds_dir)

        self.window_name = "PlayWait - AR Treasure Hunt"
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

        # YOLO lazy — 첫 detect 시 모델 로드
        self.engine = YoloEngine(min_confidence=0.35)

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
                self.state = TreasureState(difficulty=self.difficulty)
                self.sound.play("click")
                print(
                    f"🎮 난이도: {theme.DIFFICULTY_KOREAN[self.difficulty]} | "
                    f"목표 {self.state.get_goal()}개, "
                    f"유지 {self.state.get_hold_required():.1f}s"
                )
                self.phase = GamePhase.READY

        elif self.phase == GamePhase.READY:
            if key == ord(' '):
                self.sound.play("click")
                self.state.start_game()
                self.last_frame_time = time.time()
                self.sound.play(theme.TREASURE_REVEAL_SOUND)
                kor = (self.state.current_target.ko
                       if self.state.current_target else "??")
                print(f"\n=== 시작! 첫 보물: {kor} ===")
                self.phase = GamePhase.PLAYING

        elif self.phase == GamePhase.GAME_OVER:
            if key in (ord('r'), ord('R')):
                self.sound.play("click")
                self.state.reset()
                self.find_flash_until = 0.0
                print("\n🔄 게임 재시작")
                self.phase = GamePhase.DIFFICULTY_SELECT

        return True

    # ----------------------------------------
    # PLAYING 갱신
    # ----------------------------------------
    def update_playing(self, dt: float, frame_bgr, frame_w, frame_h):
        """YOLO + ROI 판정 → state 적용. detections 리턴 (render용)."""
        # YOLO 검출
        detections = self.engine.detect(frame_bgr)

        # 중앙 ROI
        roi = compute_center_roi(frame_w, frame_h)

        # 현재 보물 검출
        target_class = self.state.current_target.yolo_class
        target_in_roi = find_target_in_roi(
            detections, target_class=target_class, roi=roi,
            min_confidence=self.state.get_min_confidence(),
        )
        in_roi = (target_in_roi is not None)

        # state 적용
        before_found = self.state.found_count
        found_now = self.state.apply_detection(in_roi=in_roi, dt=dt)

        if found_now:
            now = time.time()
            self.find_flash_until = now + FIND_FLASH_DURATION
            self.last_find_ko = self.state.found_history[-1]
            # 발견된 보물의 한글 찾기
            from treasure_clues import get_treasure_by_class
            try:
                self.last_find_ko = get_treasure_by_class(
                    self.state.found_history[-1]
                ).ko
            except ValueError:
                pass
            self.last_find_score = SCORE_PER_FIND
            self.sound.play(theme.TREASURE_FIND_SOUND)
            new_target_ko = (self.state.current_target.ko
                             if self.state.current_target else "??")
            print(f"  ✨ 발견! {self.last_find_ko} (+{SCORE_PER_FIND}점) "
                  f"→ 누적 {self.state.score}점, "
                  f"{self.state.found_count}/{self.state.get_goal()}")
            if self.state.is_win():
                print(f"  → 🏆 목표 달성!")
            else:
                print(f"  → 다음 보물: {new_target_ko}")

        # 종료 체크
        end = self.state.check_end()
        if end != END_NONE:
            if end == END_WIN:
                self.sound.play(theme.TREASURE_VICTORY_SOUND)
                summary = self.state.get_summary()
                print(f"\n{'='*40}")
                print(f"🏆 보물 발견 성공! "
                      f"{summary['found_count']}/{summary['goal']}  "
                      f"({summary['score']}점, 시간 보너스 "
                      f"{summary['time_bonus']}점 포함)")
                print(f"{'='*40}")
            else:
                self.sound.play(theme.TREASURE_LOSE_SOUND)
                print(f"\n{'='*40}")
                print(f"💔 시간 초과: "
                      f"{self.state.found_count}/{self.state.get_goal()} 발견")
                print(f"{'='*40}")
            self.phase = GamePhase.GAME_OVER

        return detections, roi, in_roi

    # ----------------------------------------
    # 렌더
    # ----------------------------------------
    def render(self, frame, detections, roi, in_roi):
        ui.begin_frame()
        h, w, _ = frame.shape

        if self.phase == GamePhase.DIFFICULTY_SELECT:
            ui.draw_treasure_difficulty_select(frame)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        if self.phase == GamePhase.READY:
            ui.draw_treasure_ready(frame)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        # HUD
        ui.draw_treasure_score_bar(
            frame,
            found_count=self.state.found_count,
            goal=self.state.get_goal(),
            remaining=self.state.get_remaining(),
        )

        if self.phase == GamePhase.PLAYING:
            target = self.state.current_target
            # YOLO 박스 (즉시 — PIL 큐 안 거침)
            if detections:
                # 현재 보물만 보여줄지, 모든 검출을 보여줄지 — 모든 검출
                # 표시(매장 탐험감 ↑). 보물 클래스는 강조.
                ui.draw_treasure_ar_boxes(
                    frame, detections,
                    target_class=target.yolo_class,
                    found_just_now=False,
                )

            # 중앙 ROI 타겟 (둥근 사각 + 십자선 + 진행 게이지)
            ui.draw_treasure_center_target(
                frame, roi,
                hold_progress=self.state.get_hold_progress(),
                active=in_roi,
            )

            # 좌하단 미션 카드
            ui.draw_treasure_card(frame, target)

            # 발견 플래시
            now = time.time()
            if now < self.find_flash_until:
                ui.draw_treasure_find_flash(
                    frame,
                    treasure_ko=self.last_find_ko,
                    score_gained=self.last_find_score,
                )

        elif self.phase == GamePhase.GAME_OVER:
            ui.draw_treasure_game_over(
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
        print("🔍 PlayWait W10 - AR 보물찾기 (Hidden Menu Hunt)")
        print("=" * 50)
        print("\n[게임 조작]")
        print("  [1/2/3] 난이도 (쉬움/보통/어려움)")
        print("  [SPACE] 게임 시작")
        print("  [R]     재시작")
        print("  [Q/ESC] 종료")
        print("\n[화면 조작]")
        print("  [F] 전체 화면 / [+] 확대 / [-] 축소 / [0] 리셋")
        print("\nNOTE: 첫 실행 시 YOLO 모델(yolov8n.pt, ~6.5MB) "
              "~3초 동안 로드.")
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

                detections = []
                roi = compute_center_roi(w, h)
                in_roi = False
                if self.phase == GamePhase.PLAYING:
                    detections, roi, in_roi = self.update_playing(
                        dt, frame, w, h,
                    )

                fps_dt = now - self.prev_fps_time
                if fps_dt > 0:
                    self.fps = 1.0 / fps_dt
                self.prev_fps_time = now

                frame = self.render(frame, detections, roi, in_roi)

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

        if self.engine is not None:
            try:
                self.engine = None
                print("   ✓ YOLO 엔진 참조 해제")
            except Exception:
                pass

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

    _game_instance = HiddenMenuHuntGame()
    try:
        _game_instance.run()
    finally:
        _game_instance = None


if __name__ == "__main__":
    main()
