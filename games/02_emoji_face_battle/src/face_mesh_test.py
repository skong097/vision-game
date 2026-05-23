"""
face_mesh_test.py — Face Mesh 단독 + 표정 분류기 실시간 시각화
================================================================

W4 Step 2 단독 테스트. 카메라에서 face mesh를 실시간으로 추출하고
face_features → expression_classifier 파이프라인 결과를 화면에 표시.

실행:
    cd ~/PlayWait
    python -m games.02_emoji_face_battle.src.face_mesh_test

종료: 'q' 또는 ESC
체크 포인트:
    - FPS 30+ 유지 여부 (refine_landmarks=True 영향)
    - 4종 표정 인식 정확도 (본인 얼굴로 직접 짓기)
    - 분류 결과가 EXPR_NEUTRAL → 표정 → EXPR_NEUTRAL 로 자연스럽게 전이

Author: Stephen (gjkong)
Date: 2026-05-05 (W4 Step 2)
"""

import os
import sys
import time

import cv2
import mediapipe as mp

try:
    from .face_features import extract_face_features
    from .expression_classifier import (
        ALL_EXPRESSIONS, EXPR_NEUTRAL, EXPRESSION_PROFILES,
        classify_expression, expression_similarity,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_features import extract_face_features
    from expression_classifier import (
        ALL_EXPRESSIONS, EXPR_NEUTRAL, EXPRESSION_PROFILES,
        classify_expression, expression_similarity,
    )


# 한글 표정명
EXPR_KOREAN = {
    "smile": "웃음",
    "sad": "슬픔",
    "surprised": "놀람",
    "angry": "화남",
    "neutral": "무표정",
}


def main():
    print("=" * 50)
    print("face_mesh_test.py - W4 Step 2 시각화")
    print("=" * 50)
    print("\n4종 표정을 직접 지어 인식 정확도 확인:")
    print(f"  {' / '.join(EXPR_KOREAN[e] for e in ALL_EXPRESSIONS)}")
    print("\n종료: 'q' 또는 ESC\n")

    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps = 0.0
    prev_t = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            features = None
            classified = (EXPR_NEUTRAL, 0.0)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = extract_face_features(landmarks)
                classified = classify_expression(features)

                # 핵심 landmark 작은 점으로 표시 (디버그)
                for idx in (61, 291, 13, 14, 159, 145, 386, 374,
                             33, 263, 55, 285):
                    pt = landmarks[idx]
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)),
                               2, (0, 255, 255), -1)

            # FPS
            now = time.time()
            dt = now - prev_t
            if dt > 0:
                fps = 1.0 / dt
            prev_t = now

            # HUD
            expr_name, confidence = classified
            kor = EXPR_KOREAN.get(expr_name, expr_name)
            color = (0, 255, 255) if expr_name != EXPR_NEUTRAL else (180, 180, 180)
            cv2.rectangle(frame, (0, 0), (300, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Expr: {expr_name}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 4종 표정 유사도 게이지 (오른쪽)
            if features:
                for i, expr in enumerate(ALL_EXPRESSIONS):
                    s = expression_similarity(expr, features)
                    bar_y = 25 + i * 24
                    cv2.putText(frame, expr, (w - 220, bar_y + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    bar_x = w - 130
                    bar_w = 100
                    cv2.rectangle(frame, (bar_x, bar_y),
                                   (bar_x + bar_w, bar_y + 16),
                                   (60, 60, 60), -1)
                    fill = int(bar_w * s)
                    bar_color = (0, 200, 50) if s >= 0.70 else (50, 150, 200)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                   (bar_x + fill, bar_y + 16),
                                   bar_color, -1)
                    cv2.putText(frame, f"{int(s * 100)}%",
                                (bar_x + bar_w + 5, bar_y + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1)

            cv2.imshow("PlayWait W4 - Face Mesh Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n종료")


if __name__ == "__main__":
    main()
