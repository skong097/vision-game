"""
yolo_engine.py — YOLO (ultralytics) 검출기 래퍼
================================================

ultralytics 라이브러리 의존성을 한 군데로 격리. 게임 코드는 이 클래스의
.detect()만 호출하면 되고, 모델 로딩 정책·결과 파싱은 여기서 처리.

특징
----
- **lazy load**: 인스턴스 생성 시점에는 모델 다운로드/로드 안 함.
  첫 detect() 호출에서 로드 (테스트·import 시간 단축).
- duck-typed 인터페이스: object_pipeline.analyze_frame()이 이 객체를
  detector로 받아 .detect()만 호출 → 테스트에서 FakeDetector로 대체 가능.

Author: Stephen (gjkong)
Date: 2026-05-11 (W5 Step 4)
"""

from dataclasses import dataclass


# ============================================================
# 1. Detection 레코드
# ============================================================
@dataclass(frozen=True)
class Detection:
    """YOLO 검출 결과 한 건.

    Attributes:
        class_id: COCO 클래스 ID (0~79)
        class_name: 클래스 이름 (예: "cup", "bottle", "person")
        confidence: 검출 신뢰도 0~1
        bbox: (x1, y1, x2, y2) 픽셀 정수 좌표
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple


# ============================================================
# 2. YOLO 엔진
# ============================================================
class YoloEngine:
    """ultralytics YOLO 모델 lazy wrapper."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        min_confidence: float = 0.5,
    ):
        """
        Args:
            model_name: ultralytics 모델 식별자 (파일/이름).
                        yolov8n=가장 작음(빠름·정확도↓), yolov8s/m/l 단계.
            min_confidence: 이 미만의 검출은 결과에서 제외.
        """
        self.model_name = model_name
        self.min_confidence = min_confidence
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)

    def detect(self, frame_bgr) -> list:
        """BGR 프레임에서 객체 검출.

        Args:
            frame_bgr: OpenCV BGR ndarray (H, W, 3)
        Returns:
            List[Detection] — min_confidence 이상만
        """
        self._ensure_loaded()
        # verbose=False: ultralytics stdout 로그 억제 (30fps 게임 루프 환경)
        results = self._model.predict(
            frame_bgr,
            conf=self.min_confidence,
            verbose=False,
        )
        return self._parse_results(results)

    @staticmethod
    def _parse_results(results) -> list:
        """ultralytics Results 객체 → List[Detection].

        results는 보통 batch=1이지만 일반화해 모든 frame 결과 합침.
        """
        out = []
        for r in results:
            names = r.names  # {class_id: name}
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # list(...)는 tensor·ndarray·list 모두 수용 → 테스트 친화적
                xyxy = list(box.xyxy[0])
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                out.append(Detection(
                    class_id=cls_id,
                    class_name=names.get(cls_id, f"class_{cls_id}"),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                ))
        return out
