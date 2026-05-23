"""person_mask.py — MediaPipe SelfieSegmentation 픽셀 마스크 래퍼.

다른 게임에서도 재활용 가능한 코어 vision 모듈.

사용 예
-------
>>> from core.vision.person_mask import PersonMaskDetector
>>> with PersonMaskDetector(model_selection=0) as det:
...     mask = det.process(frame_bgr)   # ndarray (H, W) uint8, 1=사람

설계
----
- model_selection=0 일반 모델 (매장 카메라 거리에 충분)
- 내부 처리는 다운샘플(디폴트 180px 높이) → 원본 크기로 nearest upscale
- soft mask threshold 0.5로 binary 변환

Author: Stephen (gjkong)
Date: 2026-05-19
"""

import cv2
import numpy as np
import mediapipe as mp

mp_solutions = mp.solutions


class PersonMaskDetector:
    """MediaPipe SelfieSegmentation 래퍼."""

    def __init__(
        self,
        model_selection: int = 0,
        downsample_height: int = 180,
        threshold: float = 0.5,
    ):
        self.model_selection = model_selection
        self.downsample_height = downsample_height
        self.threshold = threshold
        self._seg = mp_solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR frame → binary mask (uint8 0/1, 원본 크기)."""
        h, w = frame_bgr.shape[:2]
        ds_h = self.downsample_height
        ds_w = max(1, int(w * ds_h / h))
        small = cv2.resize(frame_bgr, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self._seg.process(rgb)
        soft = result.segmentation_mask
        binary_small = (soft > self.threshold).astype(np.uint8)
        mask = cv2.resize(binary_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def close(self):
        self._seg.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
