"""
game.py — W5 컬러 헌트 (Color Hunt)
=====================================

PlayWait W5 — YOLO로 매장 사물을 검출하고, 우리가 만든 8색 분류기로 객체의
대표 색을 식별. 미션 색과 일치하는 객체 N개를 시간 안에 모으면 승리.

기술 스택
---------
- YoloEngine (core/vision/yolo_engine.py) — ultralytics 래퍼
- color_classifier.classify_color()  — HSV → 8색 분류 (순수 함수)
- object_pipeline.analyze_frame()    — YOLO + ROI + 분류 통합
- mission_generator                  — 난이도→미션
- score_tracker                      — 수집·dedup·종료 판정

W3·W4 인프라 재활용
---------------------
- theme.py        (PinkLAB + W5 8색 토큰)
- ui_renderer.py  (한글 PIL 배치 + W5 HUD/객체 박스)
- sound_manager.py (pygame.mixer + 폴백)
- 안전 종료 4중 방어 (try-finally + signal + atexit + cap/yolo close)

핵심 설계 결정
---------------
- **모든 import는 파일 최상단 dual try/except 한 곳에 집중** (W4 트러블 #16 학습).
  함수 내부 lazy import 절대 금지.
- YOLO는 lazy load: 첫 detect() 호출 시 모델 로드 → 게임 윈도우 즉시 표시.
- 매칭 카운트는 score_tracker의 dedup 윈도우(5초)로 자동 처리.

실행
----
    cd ~/PlayWait
    .venv/bin/python -m games.11_color_hunt.src.game

조작
----
    [1/2/3]  난이도 (쉬움/보통/어려움)
    [SPACE]  게임 시작
    [R]      재시작 (게임 종료 후)
    [Q/ESC]  종료
    [F]      전체 화면 / [+] 확대 / [-] 축소 / [0] 리셋

Author: Stephen (gjkong)
Date: 2026-05-12 (W5 Step 5)
"""

import atexit
import os
import signal
import sys
import time
from enum import Enum

import cv2

# ============================================================
# 모듈 import — 한 곳에 집중 (W4 트러블 #16 학습)
# ============================================================
try:
    from .color_classifier import COLOR_NEUTRAL
    from .mission_generator import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        MissionGenerator,
    )
    from .object_pipeline import analyze_frame
    from .hunt_tracker import END_NONE, END_TIMEOUT, END_WIN, HuntTracker
    from .sound_manager import SoundManager
    from . import theme
    from . import ui_renderer as ui
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from color_classifier import COLOR_NEUTRAL
    from mission_generator import (
        DIFFICULTIES,
        DIFFICULTY_EASY,
        DIFFICULTY_HARD,
        DIFFICULTY_NORMAL,
        MissionGenerator,
    )
    from object_pipeline import analyze_frame
    from hunt_tracker import END_NONE, END_TIMEOUT, END_WIN, HuntTracker
    from sound_manager import SoundManager
    import theme
    import ui_renderer as ui

# YoloEngine는 core 패키지에서 — 프로젝트 루트가 sys.path에 있어야 함.
# 모듈 실행(-m games...) 시에는 자동 OK. 직접 실행 시 위 sys.path 보강 + 루트 추가.
try:
    from core.vision.yolo_engine import YoloEngine
except ImportError:
    # 직접 실행 시 프로젝트 루트도 sys.path에
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from core.vision.yolo_engine import YoloEngine


# ============================================================
# 페이즈
# ============================================================
class GamePhase(Enum):
    DIFFICULTY_SELECT = "difficulty_select"
    READY = "ready"
    PLAYING = "playing"
    GAME_OVER = "game_over"


# 매칭 플래시 효과 지속 시간
FLASH_DURATION = 0.4


# ============================================================
# 메인 게임
# ============================================================
class ColorHuntGame:

    def __init__(self):
        self.cap = None
        self.engine = None

        self.difficulty = DIFFICULTY_NORMAL
        self.mission_gen = MissionGenerator(difficulty=self.difficulty)
        self.mission = None
        self.tracker = None
        self.phase = GamePhase.DIFFICULTY_SELECT

        # 매칭 플래시 (bbox, expires_at) 리스트
        self.flashes = []

        # 사운드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sounds_dir = os.path.join(
            os.path.dirname(current_dir), "assets", "sounds"
        )
        self.sound = SoundManager(sounds_dir)

        # 화면
        self.window_name = "PlayWait - Color Hunt"
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
            print("   sudo fuser -k /dev/video0  ← 잠금 해제 시도")
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

        # YOLO는 lazy — 첫 detect 호출 시 ~3초 모델 로드
        self.engine = YoloEngine(min_confidence=0.4)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._apply_window_size()
        return True

    # ----------------------------------------
    # 키 입력
    # ----------------------------------------
    def handle_key(self, key: int) -> bool:
        if key == ord('q') or key == 27:
            return False

        # 화면 크기
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
                self.mission_gen = MissionGenerator(difficulty=self.difficulty)
                # 첫 미션 출제 → READY
                self.mission = self.mission_gen.next_mission()
                self.tracker = HuntTracker(self.mission)
                self.sound.play("click")
                color_kr = theme.HUNT_COLOR_KOREAN.get(
                    self.mission.target_color, "??"
                )
                print(
                    f"🎮 난이도: {theme.DIFFICULTY_KOREAN[self.difficulty]} | "
                    f"미션: {color_kr} {self.mission.goal_count}개 / "
                    f"{int(self.mission.time_limit)}초 / 게이트 "
                    f"{int(self.mission.confidence_gate*100)}%"
                )
                self.phase = GamePhase.READY

        elif self.phase == GamePhase.READY:
            if key == ord(' '):
                self.sound.play("click")
                self._start_playing()

        elif self.phase == GamePhase.GAME_OVER:
            if key in (ord('r'), ord('R')):
                self.sound.play("click")
                self.flashes.clear()
                self.mission = None
                self.tracker = None
                print("\n🔄 재시작")
                self.phase = GamePhase.DIFFICULTY_SELECT

        return True

    # ----------------------------------------
    # 페이즈 전이
    # ----------------------------------------
    def _start_playing(self):
        self.tracker.start()
        self.sound.play("reveal")
        print(f"\n=== 시작! ({int(self.mission.time_limit)}초 안에 "
              f"{theme.HUNT_COLOR_KOREAN.get(self.mission.target_color, '??')} "
              f"{self.mission.goal_count}개 찾기) ===")
        self.phase = GamePhase.PLAYING

    def _enter_game_over(self):
        end = self.tracker.check_end()
        if end == END_WIN:
            self.sound.play(theme.HUNT_VICTORY_SOUND)
            print("\n" + "=" * 40)
            print(f"🏆 성공! {self.tracker.collected}/"
                  f"{self.mission.goal_count}개 수집  "
                  f"(남은 {self.tracker.get_remaining():.1f}s)")
            print("=" * 40)
        else:
            self.sound.play(theme.HUNT_LOSE_SOUND)
            print("\n" + "=" * 40)
            print(f"💔 시간 초과: {self.tracker.collected}/"
                  f"{self.mission.goal_count}개 수집")
            print("=" * 40)
        self.phase = GamePhase.GAME_OVER

    # ----------------------------------------
    # PLAYING 갱신
    # ----------------------------------------
    def update_playing(self, frame):
        """YOLO + 색 분류 → tracker 적용 → 매칭 시 플래시·사운드.

        Returns:
            matches list — render에서 다시 사용
        """
        matches = analyze_frame(
            frame,
            self.engine,
            target_color=self.mission.target_color,
        )

        # 매칭 카운트
        gained = self.tracker.apply_matches(matches)
        if gained > 0:
            # 가장 최근 매칭 객체의 bbox로 플래시 (대표 1개만)
            for m in matches:
                if (m.is_target_match
                        and m.color_confidence
                        >= self.mission.confidence_gate):
                    self.flashes.append({
                        "bbox": m.bbox,
                        "expires_at": time.time() + FLASH_DURATION,
                    })
                    break
            self.sound.play(theme.HUNT_MATCH_SOUND)
            print(f"  ✨ +{gained} → 총 {self.tracker.collected}/"
                  f"{self.mission.goal_count}")

        # 만료된 플래시 제거
        now = time.time()
        self.flashes = [f for f in self.flashes if f["expires_at"] > now]

        # 종료 체크
        if self.tracker.check_end() != END_NONE:
            self._enter_game_over()

        return matches

    # ----------------------------------------
    # 렌더
    # ----------------------------------------
    def render(self, frame, matches):
        ui.begin_frame()

        if self.phase == GamePhase.DIFFICULTY_SELECT:
            ui.draw_hunt_difficulty_select(frame)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        if self.phase == GamePhase.READY:
            ui.draw_hunt_ready(frame, self.mission)
            ui.draw_fps(frame, self.fps)
            ui.flush_text(frame)
            return frame

        # HUD (PLAYING / GAME_OVER 공통)
        ui.draw_hunt_score_bar(
            frame,
            mission=self.mission,
            collected=self.tracker.collected,
            remaining=self.tracker.get_remaining(),
        )

        if self.phase == GamePhase.PLAYING:
            # 객체 박스
            if matches:
                ui.draw_hunt_object_boxes(
                    frame,
                    matches,
                    target_color=self.mission.target_color,
                    confidence_gate=self.mission.confidence_gate,
                )
            # 매칭 플래시
            for f in self.flashes:
                ui.draw_hunt_match_flash(frame, f["bbox"])

        elif self.phase == GamePhase.GAME_OVER:
            ui.draw_hunt_game_over(
                frame,
                win=self.tracker.is_win(),
                summary=self.tracker.get_summary(),
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
        print("🎨 PlayWait W5 - 컬러 헌트 (Color Hunt)")
        print("=" * 50)
        print("\n[게임 조작]")
        print("  [1/2/3] 난이도 (쉬움/보통/어려움)")
        print("  [SPACE] 게임 시작")
        print("  [R]     재시작 (게임 종료 후)")
        print("  [Q/ESC] 종료")
        print("\n[화면 조작]")
        print("  [F] 전체 화면 / [+] 확대 / [-] 축소 / [0] 리셋")
        print("\nNOTE: 첫 실행 시 YOLO 모델(yolov8n.pt, ~6.5MB)을 ~3초 동안 로드합니다.")
        print()

        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)  # 셀카 모드

                matches = None
                if self.phase == GamePhase.PLAYING:
                    matches = self.update_playing(frame)

                # FPS
                now = time.time()
                fps_dt = now - self.prev_fps_time
                if fps_dt > 0:
                    self.fps = 1.0 / fps_dt
                self.prev_fps_time = now

                # 렌더
                frame = self.render(frame, matches)

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
    # 안전 종료 (W1 트러블 #2 패턴)
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
                # YOLO 모델은 ultralytics가 알아서 정리 — 참조만 끊음
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

    _game_instance = ColorHuntGame()
    try:
        _game_instance.run()
    finally:
        _game_instance = None


if __name__ == "__main__":
    main()
