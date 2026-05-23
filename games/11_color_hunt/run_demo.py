"""
run_demo.py — 컬러 헌트 라이브 데모 (Step 5 정식 game.py 전 미리보기)
==========================================================================

웹캠 입력 → YOLO 검출 → 8색 분류 → 화면 오버레이.
미션 색 토글로 매칭 객체는 굵은 박스로 강조.

조작
----
- q / ESC : 종료
- m       : 미션 색 다음 (None → burgundy → terracotta → ... → charcoal → None)
- r       : 미션 해제 (target=None)

실행
----
  cd ~/PlayWait
  source .venv/bin/activate
  python games/11_color_hunt/run_demo.py

주의
----
- 처음 실행 시 YOLO 모델(yolov8n.pt ~6.5MB)을 받아 ~3초 대기.
- GUI 환경 필요 (X/wayland). headless 서버에서는 cv2.imshow가 실패.
- 좌우 반전(거울) — 셀카 모드.
"""

import os
import sys
import time

import cv2

# 프로젝트 루트 + 게임 src 경로 등록 (이 파일을 직접 실행)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.join(_HERE, "src")
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.vision.yolo_engine import YoloEngine
from object_pipeline import analyze_frame
from color_classifier import COLOR_NEUTRAL, MISSION_COLORS


# ============================================================
# 8색의 오버레이 BGR — 실제 톤을 반영해 시각 피드백을 분류 결과와 일치시킴
# ============================================================
PALETTE_BGR = {
    "burgundy":   (50, 30, 110),
    "terracotta": (60, 90, 170),
    "mustard":    (60, 180, 200),
    "sage":       (140, 170, 160),
    "navy":       (90, 50, 30),
    "mauve":      (160, 130, 180),
    "cream":      (220, 235, 245),
    "charcoal":   (60, 60, 60),
    COLOR_NEUTRAL: (128, 128, 128),
}


# ============================================================
# 오버레이 그리기
# ============================================================
def draw_overlay(frame, matches, target_color, fps):
    """analyze_frame 결과를 frame에 in-place로 그림."""
    H, _W = frame.shape[:2]
    for m in matches:
        x1, y1, x2, y2 = m.bbox
        color = PALETTE_BGR.get(m.detected_color, (128, 128, 128))
        # 미션 매칭 객체는 굵은 박스 + 흰색 외곽선으로 강조
        if m.is_target_match:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                          (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{m.yolo_class} | {m.detected_color} {m.color_confidence:.2f}"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    target_str = target_color or "OFF"
    status = f"FPS {fps:5.1f}  |  det {len(matches):2d}  |  target: {target_str}"
    cv2.putText(frame, status, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, status, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "q/ESC quit | m next target | r off",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)


# ============================================================
# 메인 루프
# ============================================================
def main():
    print("[demo] YOLO engine 초기화 (lazy — 첫 프레임에 모델 로드)")
    engine = YoloEngine(min_confidence=0.4)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[demo] 카메라 열기 실패 (VideoCapture 0). USB/내장 웹캠 확인.",
              file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cycle = [None] + list(MISSION_COLORS)
    cycle_idx = 0
    target_color = cycle[cycle_idx]

    prev_t = time.time()
    fps = 0.0
    frame_count = 0

    print("[demo] 시작 — 'q' 또는 ESC로 종료")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[demo] 프레임 읽기 실패 — 종료")
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            matches = analyze_frame(frame, engine, target_color=target_color)

            now = time.time()
            dt = now - prev_t
            prev_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            fps = 0.9 * fps + 0.1 * inst_fps if frame_count > 2 else inst_fps
            frame_count += 1

            draw_overlay(frame, matches, target_color, fps)
            cv2.imshow("PlayWait - Color Hunt (demo)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                cycle_idx = (cycle_idx + 1) % len(cycle)
                target_color = cycle[cycle_idx]
                print(f"[demo] target -> {target_color}")
            elif key == ord("r"):
                cycle_idx = 0
                target_color = None
                print("[demo] target off")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
