"""
test_yolo_engine.py — YOLO 래퍼 구조·lazy load 테스트

실제 모델 다운로드를 트리거하지 않는 가벼운 검증만.
모델 추론 통합 테스트는 별도 (network·디스크 비용 큼).
"""

import pytest

from core.vision.yolo_engine import Detection, YoloEngine


# ============================================================
# 1. Detection 레코드
# ============================================================
def test_detection_is_frozen():
    d = Detection(class_id=41, class_name="cup", confidence=0.9, bbox=(0, 0, 10, 10))
    with pytest.raises(Exception):  # FrozenInstanceError
        d.confidence = 0.5


def test_detection_equality():
    a = Detection(0, "person", 0.99, (1, 2, 3, 4))
    b = Detection(0, "person", 0.99, (1, 2, 3, 4))
    assert a == b


# ============================================================
# 2. YoloEngine — lazy load
# ============================================================
def test_engine_does_not_load_on_init():
    """인스턴스 생성만으로는 모델 로딩 안 함 (테스트·import 비용 0)."""
    engine = YoloEngine()
    assert engine.is_loaded is False
    assert engine._model is None


def test_engine_default_config():
    engine = YoloEngine()
    assert engine.model_name == "yolov8n.pt"
    assert engine.min_confidence == 0.5


def test_engine_custom_config():
    engine = YoloEngine(model_name="yolov8s.pt", min_confidence=0.7)
    assert engine.model_name == "yolov8s.pt"
    assert engine.min_confidence == 0.7


# ============================================================
# 3. 결과 파싱 — mock ultralytics output
# ============================================================
class _MockBox:
    def __init__(self, cls_id, conf, xyxy):
        # ultralytics는 box.cls[0], box.conf[0], box.xyxy[0] 형태로 접근
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _MockResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


def test_parse_results_single_detection():
    names = {41: "cup", 39: "bottle"}
    boxes = [_MockBox(41, 0.92, [10.0, 20.0, 110.0, 220.0])]
    results = [_MockResult(boxes, names)]
    parsed = YoloEngine._parse_results(results)
    assert len(parsed) == 1
    d = parsed[0]
    assert d.class_id == 41
    assert d.class_name == "cup"
    assert d.confidence == pytest.approx(0.92)
    assert d.bbox == (10, 20, 110, 220)


def test_parse_results_multiple():
    names = {41: "cup", 0: "person"}
    boxes = [
        _MockBox(41, 0.9, [0.0, 0.0, 50.0, 50.0]),
        _MockBox(0, 0.85, [60.0, 60.0, 200.0, 300.0]),
    ]
    parsed = YoloEngine._parse_results([_MockResult(boxes, names)])
    assert len(parsed) == 2
    classes = {d.class_name for d in parsed}
    assert classes == {"cup", "person"}


def test_parse_results_empty_boxes():
    parsed = YoloEngine._parse_results([_MockResult([], {})])
    assert parsed == []


def test_parse_results_boxes_none():
    parsed = YoloEngine._parse_results([_MockResult(None, {})])
    assert parsed == []


def test_parse_results_unknown_class_id_falls_back():
    """names dict에 없는 cls_id면 'class_N' 폴백."""
    parsed = YoloEngine._parse_results([
        _MockResult([_MockBox(999, 0.5, [0, 0, 1, 1])], {})
    ])
    assert parsed[0].class_name == "class_999"
