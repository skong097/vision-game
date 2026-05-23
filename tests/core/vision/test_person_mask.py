"""PersonMaskDetector 단위 테스트 — MediaPipe mock."""
import numpy as np
from unittest.mock import MagicMock, patch

from core.vision.person_mask import PersonMaskDetector


def _soft(h, w, value=0.8):
    return np.full((h, w), value, dtype=np.float32)


@patch("core.vision.person_mask.mp_solutions")
def test_process_returns_binary_mask_at_original_size(mock_mp):
    fake = MagicMock()
    fake.segmentation_mask = _soft(180, 320, 0.8)
    seg = MagicMock()
    seg.process.return_value = fake
    mock_mp.selfie_segmentation.SelfieSegmentation.return_value = seg

    det = PersonMaskDetector(model_selection=0, downsample_height=180)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = det.process(frame)

    assert mask.dtype == np.uint8
    assert mask.shape == (720, 1280)
    assert mask.max() == 1
    det.close()


@patch("core.vision.person_mask.mp_solutions")
def test_process_thresholds_at_half(mock_mp):
    soft = np.zeros((180, 320), dtype=np.float32)
    soft[:90, :] = 0.7
    soft[90:, :] = 0.3
    fake = MagicMock()
    fake.segmentation_mask = soft
    seg = MagicMock()
    seg.process.return_value = fake
    mock_mp.selfie_segmentation.SelfieSegmentation.return_value = seg

    det = PersonMaskDetector(downsample_height=180)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = det.process(frame)

    assert mask[100, 100] == 1
    assert mask[600, 100] == 0
    det.close()


@patch("core.vision.person_mask.mp_solutions")
def test_context_manager_closes(mock_mp):
    seg = MagicMock()
    mock_mp.selfie_segmentation.SelfieSegmentation.return_value = seg

    with PersonMaskDetector():
        pass
    seg.close.assert_called_once()
