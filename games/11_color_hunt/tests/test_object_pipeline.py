"""
test_object_pipeline.py — ROI 추출·HSV·통합 분석 단위 테스트

실제 YOLO 없이 stub detector로 통합 동작 검증.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from object_pipeline import (
    DEFAULT_ROI_RATIO, ObjectMatch,
    analyze_frame, compute_median_hsv, extract_center_roi,
)


# ============================================================
# 헬퍼: OpenCV HSV로 솔리드 컬러 프레임 합성 → BGR
# ============================================================
def _solid_bgr(shape, hsv_opencv):
    """hsv_opencv = (H in [0,179], S in [0,255], V in [0,255])."""
    h_img = np.full((*shape, 3), hsv_opencv, dtype=np.uint8)
    return cv2.cvtColor(h_img, cv2.COLOR_HSV2BGR)


@dataclass
class _FakeDet:
    """analyze_frame이 기대하는 duck-typed detection."""
    class_name: str
    confidence: float
    bbox: tuple


class _StubDetector:
    """canned detections를 그대로 반환하는 stub."""
    def __init__(self, dets):
        self._dets = dets
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self._dets


# ============================================================
# 1. extract_center_roi
# ============================================================
def test_center_roi_basic_60_percent():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = extract_center_roi(frame, (0, 0, 100, 100), ratio=0.6)
    # margin = 100 * (1-0.6) / 2 = 20 → ROI [20:80, 20:80] → 60x60
    assert roi.shape == (60, 60, 3)


def test_center_roi_offset_bbox():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    roi = extract_center_roi(frame, (50, 60, 150, 160), ratio=0.5)
    # 100x100 box → margin 25 → 50x50 ROI
    assert roi.shape == (50, 50, 3)


def test_center_roi_ratio_one_returns_full_box():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    roi = extract_center_roi(frame, (10, 10, 70, 70), ratio=1.0)
    assert roi.shape == (60, 60, 3)


def test_center_roi_ratio_zero_returns_none():
    """ratio=0이면 margin이 박스 절반 → cx2 <= cx1 → None."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = extract_center_roi(frame, (10, 10, 50, 50), ratio=0.0)
    assert roi is None


def test_center_roi_degenerate_bbox_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert extract_center_roi(frame, (50, 50, 50, 50)) is None  # zero-size
    assert extract_center_roi(frame, (60, 50, 50, 70)) is None  # x1 > x2


def test_center_roi_clips_to_frame_bounds():
    """bbox가 frame 밖까지 뻗어도 frame 경계로 clip."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = extract_center_roi(frame, (-20, -20, 120, 120), ratio=1.0)
    assert roi is not None
    assert roi.shape[0] <= 100 and roi.shape[1] <= 100


# ============================================================
# 2. compute_median_hsv — 합성 솔리드 컬러로 검증
# ============================================================
def test_median_hsv_pure_red():
    # OpenCV pure red: H=0, S=255, V=255 → 정규화 (0°, 1.0, 1.0)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[:] = (0, 0, 255)  # BGR
    h, s, v = compute_median_hsv(frame)
    assert h == pytest.approx(0.0, abs=2.0)
    assert s == pytest.approx(1.0, abs=0.01)
    assert v == pytest.approx(1.0, abs=0.01)


def test_median_hsv_pure_blue():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[:] = (255, 0, 0)  # BGR blue
    h, s, v = compute_median_hsv(frame)
    # OpenCV blue H=120 → normalize 240°
    assert h == pytest.approx(240.0, abs=2.0)
    assert s == pytest.approx(1.0, abs=0.01)
    assert v == pytest.approx(1.0, abs=0.01)


def test_median_hsv_white_has_zero_saturation():
    frame = np.full((40, 40, 3), 255, dtype=np.uint8)
    h, s, v = compute_median_hsv(frame)
    assert s == pytest.approx(0.0, abs=0.01)
    assert v == pytest.approx(1.0, abs=0.01)


def test_median_hsv_black_has_zero_value():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    _, _, v = compute_median_hsv(frame)
    assert v == pytest.approx(0.0, abs=0.01)


def test_median_hsv_none_input():
    assert compute_median_hsv(None) is None


def test_median_hsv_empty_array():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert compute_median_hsv(empty) is None


def test_median_hsv_robust_to_minority_noise():
    """ROI 대부분이 burgundy, 일부 픽셀이 흰 하이라이트 → median은 burgundy 유지."""
    # OpenCV HSV (175, 166, 89) ≈ 정규화 (350°, 0.65, 0.35) — burgundy
    frame = _solid_bgr((60, 60), (175, 166, 89))
    # 모서리 4픽셀에 흰색 하이라이트
    for (y, x) in [(0, 0), (0, 59), (59, 0), (59, 59)]:
        frame[y, x] = (255, 255, 255)
    h, s, v = compute_median_hsv(frame)
    assert 346 < h < 354  # 175*2=350 근처
    assert 0.60 < s < 0.70
    assert 0.30 < v < 0.40


# ============================================================
# 3. analyze_frame — stub detector로 end-to-end
# ============================================================
def _frame_with_patch(patch_hsv_opencv, bbox, frame_shape=(200, 200)):
    """frame_shape 검정 바탕에 bbox 영역만 지정 HSV로 칠한 BGR 프레임."""
    frame = np.zeros((*frame_shape, 3), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    patch = _solid_bgr((y2 - y1, x2 - x1), patch_hsv_opencv)
    frame[y1:y2, x1:x2] = patch
    return frame


def test_analyze_frame_classifies_burgundy_patch():
    # 검정 프레임 + 좌상단 100x100 버건디 패치
    bbox = (0, 0, 100, 100)
    frame = _frame_with_patch((175, 166, 89), bbox)
    det = _FakeDet("bottle", 0.92, bbox)
    results = analyze_frame(frame, _StubDetector([det]), target_color="burgundy")

    assert len(results) == 1
    m = results[0]
    assert m.yolo_class == "bottle"
    assert m.yolo_confidence == pytest.approx(0.92)
    assert m.detected_color == "burgundy"
    assert m.is_target_match is True


def test_analyze_frame_target_mismatch_flag():
    """미션 색이 다른 색이면 is_target_match=False, 분류 자체는 정상."""
    bbox = (0, 0, 100, 100)
    frame = _frame_with_patch((175, 166, 89), bbox)  # burgundy
    det = _FakeDet("bottle", 0.92, bbox)
    results = analyze_frame(frame, _StubDetector([det]), target_color="mustard")

    assert results[0].detected_color == "burgundy"
    assert results[0].is_target_match is False


def test_analyze_frame_no_target_keeps_is_match_false():
    bbox = (0, 0, 100, 100)
    frame = _frame_with_patch((175, 166, 89), bbox)
    det = _FakeDet("bottle", 0.92, bbox)
    results = analyze_frame(frame, _StubDetector([det]), target_color=None)
    assert results[0].is_target_match is False


def test_analyze_frame_multiple_detections_independent():
    """서로 다른 색 두 영역 + 검출 두 개 → 각자 분류."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # 좌측: burgundy
    frame[0:100, 0:100] = _solid_bgr((100, 100), (175, 166, 89))
    # 우측: mustard (OpenCV H=22(=44°)... 머스타드 hue 45°, 정규화 후. OpenCV H=22→44°)
    # mustard profile: hue [35,55], s [0.55,0.90], v [0.45,0.75]
    # 중앙값: h=45°(OCV 22), s=0.725(OCV 184), v=0.60(OCV 153)
    frame[0:100, 100:200] = _solid_bgr((100, 100), (22, 184, 153))

    dets = [
        _FakeDet("bottle", 0.9, (0, 0, 100, 100)),
        _FakeDet("cup", 0.85, (100, 0, 200, 100)),
    ]
    results = analyze_frame(frame, _StubDetector(dets), target_color="mustard")

    assert len(results) == 2
    by_class = {m.yolo_class: m for m in results}
    assert by_class["bottle"].detected_color == "burgundy"
    assert by_class["bottle"].is_target_match is False
    assert by_class["cup"].detected_color == "mustard"
    assert by_class["cup"].is_target_match is True


def test_analyze_frame_empty_detections():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert analyze_frame(frame, _StubDetector([])) == []


def test_analyze_frame_skips_degenerate_bbox():
    """bbox가 너무 작아 ROI=None이면 결과에 포함 X (스킵)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = _FakeDet("x", 0.9, (50, 50, 50, 50))  # zero size
    results = analyze_frame(frame, _StubDetector([det]))
    assert results == []


def test_analyze_frame_calls_detector_once():
    """detector.detect는 정확히 한 번만 호출."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    stub = _StubDetector([])
    analyze_frame(frame, stub)
    assert stub.calls == 1


# ============================================================
# 4. ObjectMatch 레코드 — frozen, 동등성
# ============================================================
def test_object_match_is_frozen():
    m = ObjectMatch("cup", 0.9, (0, 0, 10, 10), "sage", 0.8, True)
    with pytest.raises(Exception):  # FrozenInstanceError
        m.yolo_class = "bowl"


def test_default_roi_ratio_value():
    """문서화된 60% 기본값 유지."""
    assert DEFAULT_ROI_RATIO == 0.60
