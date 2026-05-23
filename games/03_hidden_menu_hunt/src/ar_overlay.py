"""
ar_overlay.py — 중앙 ROI 판정 + 헬퍼
==========================================

순수 함수: center_roi() / contains_center() / closest_target_bbox().
실제 cv2 도형 렌더는 ui_renderer 쪽 (W10 컴포넌트).

Author: Stephen (gjkong)
Date: 2026-05-12 (W10 Step 4)
"""

from dataclasses import dataclass


# ============================================================
# 1. 중앙 ROI 기본 비율
# ============================================================
DEFAULT_ROI_RATIO = 0.30   # 화면의 중앙 30% (양쪽 35% margin)


# ============================================================
# 2. ROI dataclass
# ============================================================
@dataclass(frozen=True)
class CenterROI:
    """중앙 ROI 박스 (픽셀 좌표)."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def contains_point(self, px: int, py: int) -> bool:
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def to_tuple(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


# ============================================================
# 3. ROI 계산
# ============================================================
def compute_center_roi(frame_w: int, frame_h: int,
                       ratio: float = DEFAULT_ROI_RATIO) -> CenterROI:
    """프레임 가운데 ratio 비율 만큼의 정사각 ROI."""
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"ratio는 (0, 1] — 받은 값: {ratio}")
    size = int(min(frame_w, frame_h) * ratio)
    cx = frame_w // 2
    cy = frame_h // 2
    half = size // 2
    return CenterROI(
        x1=max(0, cx - half),
        y1=max(0, cy - half),
        x2=min(frame_w, cx + half),
        y2=min(frame_h, cy + half),
    )


# ============================================================
# 4. 매칭 검출 — bbox center가 ROI 안에 있나?
# ============================================================
def bbox_center(bbox) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def find_target_in_roi(detections, target_class: str,
                       roi: CenterROI,
                       min_confidence: float = 0.0):
    """target_class 객체 중 bbox 중심이 ROI 안에 있는 가장 신뢰도 높은 detection.

    Args:
        detections: yolo_engine.Detection 리스트 (.class_name/.confidence/.bbox)
        target_class: 찾고 있는 클래스 이름
        roi: CenterROI
        min_confidence: 이 미만은 무시

    Returns:
        Detection 또는 None
    """
    best = None
    best_conf = min_confidence - 1e-9
    for d in detections:
        if d.class_name != target_class:
            continue
        if d.confidence < min_confidence:
            continue
        cx, cy = bbox_center(d.bbox)
        if not roi.contains_point(cx, cy):
            continue
        if d.confidence > best_conf:
            best = d
            best_conf = d.confidence
    return best


def collect_targets(detections, target_class: str,
                    min_confidence: float = 0.0) -> list:
    """target_class 클래스의 모든 detection (UI에서 박스 그리기용)."""
    return [
        d for d in detections
        if d.class_name == target_class and d.confidence >= min_confidence
    ]
