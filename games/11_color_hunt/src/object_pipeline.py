"""
object_pipeline.py — YOLO 검출 + ROI 추출 + 컬러 분류 통합
============================================================

frame_bgr → detector.detect() → 박스별 중앙 ROI → median HSV → 색 분류.

설계 원칙
---------
- **detector는 duck-typed**: .detect(frame_bgr) -> [obj with class_name,
  confidence, bbox] 만 충족하면 됨. YoloEngine이든 stub이든 OK.
  → 이 모듈은 ultralytics 직접 import 안 함, 순수 파이프라인 로직만 담당.
- **ROI는 박스 중앙 60%**: bbox 가장자리는 배경 침범 가능 → 중앙만 색 추출.
- **HSV는 median**: 하이라이트·그림자 픽셀의 outlier 영향을 평균보다 약화.

Author: Stephen (gjkong)
Date: 2026-05-11 (W5 Step 4)
"""

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from .color_classifier import classify_color
except ImportError:
    # 직접 실행 / pytest sys.path 주입 경로
    from color_classifier import classify_color


# ============================================================
# 1. 결과 레코드
# ============================================================
@dataclass(frozen=True)
class ObjectMatch:
    """한 객체 분석 결과.

    Attributes:
        yolo_class: YOLO가 인식한 클래스 (예: "cup")
        yolo_confidence: YOLO 신뢰도 0~1
        bbox: (x1, y1, x2, y2)
        detected_color: 분류기 결과 (8색 중 하나 또는 "neutral")
        color_confidence: 색 분류 신뢰도 0~1
        is_target_match: 미션 색과 일치하는지 (target_color=None이면 항상 False)
    """
    yolo_class: str
    yolo_confidence: float
    bbox: tuple
    detected_color: str
    color_confidence: float
    is_target_match: bool


# ============================================================
# 2. 기본 상수
# ============================================================
DEFAULT_ROI_RATIO = 0.60  # 박스 중앙 60% — 배경 침범 줄임


# ============================================================
# 3. ROI 추출 — 박스 중앙 일정 비율
# ============================================================
def extract_center_roi(frame_bgr, bbox, ratio: float = DEFAULT_ROI_RATIO):
    """bbox 가운데 ratio 비율 영역만 잘라내 반환.

    Args:
        frame_bgr: 원본 프레임 (H, W, 3)
        bbox: (x1, y1, x2, y2)
        ratio: 중앙 비율 (0<ratio<=1). 1.0이면 전체 박스.
    Returns:
        잘라낸 ndarray. ROI가 비거나 박스가 너무 작으면 None.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    margin_x = int(w * (1 - ratio) / 2)
    margin_y = int(h * (1 - ratio) / 2)
    cx1, cy1 = x1 + margin_x, y1 + margin_y
    cx2, cy2 = x2 - margin_x, y2 - margin_y
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    # 프레임 경계 clip
    H, W = frame_bgr.shape[:2]
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(W, cx2), min(H, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame_bgr[cy1:cy2, cx1:cx2]


# ============================================================
# 4. 대표 HSV — 채널별 median
# ============================================================
def compute_median_hsv(roi_bgr):
    """ROI(BGR)의 채널별 median을 정규화된 HSV (H[0,360), S/V[0,1])로 반환.

    color_classifier가 기대하는 HSV convention과 맞춤.
    OpenCV cvtColor: H[0,179], S[0,255], V[0,255].

    Args:
        roi_bgr: BGR ndarray. None이거나 빈 배열이면 None 반환.
    Returns:
        (h, s, v) 튜플 또는 None
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_med = float(np.median(hsv[:, :, 0]))
    s_med = float(np.median(hsv[:, :, 1]))
    v_med = float(np.median(hsv[:, :, 2]))
    return (h_med * 2.0, s_med / 255.0, v_med / 255.0)


# ============================================================
# 5. 통합 분석
# ============================================================
def analyze_frame(
    frame_bgr,
    detector,
    target_color: str = None,
    roi_ratio: float = DEFAULT_ROI_RATIO,
) -> list:
    """프레임 한 장 → 객체별 (YOLO + 색 분류) 결과 리스트.

    Args:
        frame_bgr: BGR ndarray (H, W, 3)
        detector: .detect(frame_bgr) 메서드를 가진 객체. 반환 항목은
                  class_name·confidence·bbox 속성 필수.
        target_color: 미션 색. None이면 is_target_match는 항상 False.
        roi_ratio: extract_center_roi에 전달
    Returns:
        List[ObjectMatch]
    """
    detections = detector.detect(frame_bgr)
    out = []
    for det in detections:
        roi = extract_center_roi(frame_bgr, det.bbox, roi_ratio)
        if roi is None:
            continue
        hsv = compute_median_hsv(roi)
        if hsv is None:
            continue
        color, conf = classify_color(hsv)
        is_match = (target_color is not None) and (color == target_color)
        out.append(ObjectMatch(
            yolo_class=det.class_name,
            yolo_confidence=det.confidence,
            bbox=det.bbox,
            detected_color=color,
            color_confidence=conf,
            is_target_match=is_match,
        ))
    return out
