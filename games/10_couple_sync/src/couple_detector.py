"""
couple_detector.py — 화면 좌/우 분할 + landmark 좌표 변환
=============================================================

V1: MediaPipe Pose 단일 인스턴스로 두 사람을 추적하기 위해 frame을 좌/우로 잘라
각각에 detector.process()를 호출. 결과 landmark는 ROI(절반 frame) 기준 정규화
좌표이므로, **전체 frame 좌표계로 변환**해야 함.

이 모듈에서 mp 직접 호출 X — duck-typed detector를 받음 (.process(rgb)
또는 .process_landmarks(rgb) 호환). 게임에서 두 인스턴스를 만들어 주입.

순수 부분(좌표 변환)만 단위 테스트.

Author: Stephen (gjkong)
Date: 2026-05-12 (W8 Step 4)
"""

from dataclasses import dataclass


# ============================================================
# 1. 분할 영역
# ============================================================
SIDE_LEFT = "left"
SIDE_RIGHT = "right"


# ============================================================
# 2. 좌표 변환 — 절반 ROI 정규화 좌표 → 전체 frame 정규화 좌표
# ============================================================
def remap_landmark_to_full_frame(lm_x_in_half: float, side: str) -> float:
    """ROI(절반 frame) 안의 정규화 x → 전체 frame 정규화 x.

    좌측 ROI는 frame x ∈ [0, 0.5] 차지 → ROI x 0.5는 전체 x 0.25.
    우측 ROI는 frame x ∈ [0.5, 1.0] → ROI x 0.5는 전체 x 0.75.

    Args:
        lm_x_in_half: ROI 안에서의 정규화 x (0~1)
        side: "left" or "right"

    Returns:
        전체 frame 정규화 x (0~1)
    """
    if side == SIDE_LEFT:
        return lm_x_in_half * 0.5
    elif side == SIDE_RIGHT:
        return 0.5 + lm_x_in_half * 0.5
    else:
        raise ValueError(f"알 수 없는 side: {side}")


def split_frame_x(frame_w: int) -> tuple:
    """frame을 좌/우 절반으로 잘랐을 때의 (left_end_x, right_start_x).

    좌측 ROI: [0, left_end_x)
    우측 ROI: [right_start_x, frame_w)
    """
    half = frame_w // 2
    return (half, half)


# ============================================================
# 3. 변환된 landmark 컨테이너
# ============================================================
@dataclass(frozen=True)
class RemappedLandmark:
    """전체 frame 정규화 좌표로 변환된 landmark.

    MediaPipe NormalizedLandmark와 호환 (.x, .y 속성).
    """
    x: float
    y: float


def remap_landmarks(landmarks, side: str) -> list:
    """landmark 리스트 전체를 전체 frame 좌표계로 변환.

    Args:
        landmarks: 원본 landmark 리스트 (각각 .x, .y or [0], [1] 인덱싱)
        side: "left" or "right"

    Returns:
        RemappedLandmark 리스트. landmarks가 None이면 None.
    """
    if landmarks is None:
        return None
    if side not in (SIDE_LEFT, SIDE_RIGHT):
        raise ValueError(f"알 수 없는 side: {side}")

    out = []
    for lm in landmarks:
        if hasattr(lm, "x"):
            x_in = lm.x
            y_in = lm.y
        else:
            x_in = lm[0]
            y_in = lm[1]
        new_x = remap_landmark_to_full_frame(x_in, side)
        out.append(RemappedLandmark(x=new_x, y=y_in))
    return out
