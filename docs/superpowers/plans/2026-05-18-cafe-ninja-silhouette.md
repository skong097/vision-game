# 카페 닌자 — 그림자 실루엣 + 눈 복면 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 카페 닌자에서 배경(town.jpg)은 원색 그대로 노출하고 사람 영역만 자주톤(40,25,55) 평면 실루엣으로 표현하되, 손과 양 눈만 카메라 원본으로 노출해 "복면 닌자 그림자" 효과 구현.

**Architecture:** MediaPipe Selfie Segmentation으로 사람/배경 마스크 추출 + Face Detection의 양 눈 keypoint 활용. 합성은 ui_renderer에 신규 `apply_silhouette()` 추가, 폴백은 기존 spotlight 방식 유지. 모듈명 prefix 강제 (`ninja_silhouette.py`).

**Tech Stack:** Python 3.12, MediaPipe 0.10.14 (Selfie Segmentation + Face Detection), OpenCV 4.8.1, NumPy 1.26.4, pytest.

**Spec:** [`docs/superpowers/specs/2026-05-18-cafe-ninja-silhouette-design.md`](../specs/2026-05-18-cafe-ninja-silhouette-design.md)

---

## 파일 구조

**생성**:
- `games/01_cafe_ninja/src/ninja_silhouette.py` — Selfie Seg + Face Detection 래퍼, `SilhouetteResult` dataclass, `NinjaSilhouette` 클래스 (~150줄)
- `games/01_cafe_ninja/tests/test_ninja_silhouette.py` — 6~8 케이스 (mediapipe mock)
- `games/01_cafe_ninja/tests/test_ui_renderer.py` — 신규 (apply_silhouette + mask 헬퍼 6~7 케이스)

**수정**:
- `games/01_cafe_ninja/src/theme.py` — 파라미터 6종 추가 (BACKGROUND_* 유지 — 폴백용)
- `games/01_cafe_ninja/src/ui_renderer.py` — `_build_circular_mask()`, `_build_disk_mask()`, `apply_silhouette()` 추가
- `games/01_cafe_ninja/src/game.py` — silhouette init + run/render 통합, dual-import 블록 갱신

---

## Task 1: theme.py 파라미터 추가

**Files:**
- Modify: `games/01_cafe_ninja/src/theme.py:200-208`

- [ ] **Step 1: theme.py에 실루엣 파라미터 6종 추가**

`theme.py`의 BACKGROUND_* 다음 줄(현재 line 204 이후)에 추가:

```python
# 배경 블렌딩 강도 (0.0 = 카메라만, 1.0 = 배경만) — silhouette 비활성 시 폴백용
BACKGROUND_ALPHA = 0.40
# 배경 이미지 자체의 밝기 곱 (1.0 = 원본, 0.5 = 절반 밝기 → 톤 다운)
BACKGROUND_BRIGHTNESS = 0.25

# 10. W3 폴리싱 - 그림자 실루엣 + 눈 복면 (2026-05-18 silhouette design)
# 사람 영역을 자주톤 평면으로 덮어 닌자 그림자 효과
SILHOUETTE_TINT_BGR = (40, 25, 55)   # BGR 자주톤 (닌자 의상 톤, 기존 SPOTLIGHT_TINT와 동일)
SILHOUETTE_ALPHA = 1.0               # 1.0=평면 그림자, 0.85=카메라 약간 비침
EYE_REVEAL_RADIUS = 12               # px (얼굴 식별 불가 수준)
EYE_REVEAL_FEATHER = 6               # Gaussian sigma, 부드러운 경계
HAND_REVEAL_FEATHER = 30             # px, 손 외곽 Gaussian 페더
PERSON_MASK_THRESHOLD = 0.5          # Selfie Seg 확률 → 이진화 임계
```

- [ ] **Step 2: theme import smoke**

Run: `.venv/bin/python -c "from games.01_cafe_ninja.src import theme; print(theme.SILHOUETTE_TINT_BGR, theme.EYE_REVEAL_RADIUS)"`
Expected: `(40, 25, 55) 12`

- [ ] **Step 3: 통합 회귀로 다른 테스트 깨지지 않음 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/ -q`
Expected: 기존 케이스 100% 통과 (회귀 0건)

- [ ] **Step 4: Commit**

```bash
git add games/01_cafe_ninja/src/theme.py
git commit -m "feat(cafe-ninja): add silhouette params to theme"
```

---

## Task 2: ninja_silhouette.py 모듈 골격 + SilhouetteResult dataclass

**Files:**
- Create: `games/01_cafe_ninja/src/ninja_silhouette.py`
- Create: `games/01_cafe_ninja/tests/test_ninja_silhouette.py`

- [ ] **Step 1: 실패 테스트 작성 — SilhouetteResult dataclass 기본 동작**

`games/01_cafe_ninja/tests/test_ninja_silhouette.py`:

```python
"""ninja_silhouette 모듈 테스트 — mediapipe는 monkeypatch로 mock."""
import numpy as np
import pytest

from ninja_silhouette import SilhouetteResult


class TestSilhouetteResult:
    def test_default_empty(self):
        r = SilhouetteResult()
        assert r.person_mask is None
        assert r.eye_points == []
        assert r.detected is False

    def test_with_values(self):
        mask = np.full((480, 640), 255, dtype=np.uint8)
        r = SilhouetteResult(person_mask=mask, eye_points=[(100, 200), (200, 200)], detected=True)
        assert r.person_mask is mask
        assert len(r.eye_points) == 2
        assert r.detected is True
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ninja_silhouette.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ninja_silhouette'`

- [ ] **Step 3: 모듈 골격 + dataclass 구현**

`games/01_cafe_ninja/src/ninja_silhouette.py`:

```python
"""카페 닌자 — Selfie Segmentation + Face Detection 래퍼.

사람 영역 마스크와 양 눈 keypoint를 한 번에 추출. ui_renderer.apply_silhouette()가
이 결과로 자주톤 그림자 + 손/눈 reveal 합성을 수행한다.

mediapipe.solutions.selfie_segmentation 또는 face_detection이 사용 불가일 경우
self.available = False 로 표시하고 기존 spotlight 방식으로 폴백.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SilhouetteResult:
    """한 프레임의 silhouette 추론 결과."""
    person_mask: Optional[np.ndarray] = None  # uint8 (H, W), 0~255 (사람=255)
    eye_points: List[Tuple[int, int]] = field(default_factory=list)  # 최대 2개
    detected: bool = False                    # 사람 + 얼굴 모두 검출
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ninja_silhouette.py -v`
Expected: PASS (2 cases)

- [ ] **Step 5: Commit**

```bash
git add games/01_cafe_ninja/src/ninja_silhouette.py games/01_cafe_ninja/tests/test_ninja_silhouette.py
git commit -m "feat(cafe-ninja): add SilhouetteResult dataclass"
```

---

## Task 3: NinjaSilhouette 클래스 init + available 폴백

**Files:**
- Modify: `games/01_cafe_ninja/src/ninja_silhouette.py`
- Modify: `games/01_cafe_ninja/tests/test_ninja_silhouette.py`

- [ ] **Step 1: 실패 테스트 작성 — NinjaSilhouette init 동작**

`test_ninja_silhouette.py` 하단에 추가:

```python
from ninja_silhouette import NinjaSilhouette


class TestNinjaSilhouetteInit:
    def test_init_success(self, monkeypatch):
        """mediapipe 정상일 때 available=True."""
        mock_seg = type("MockSeg", (), {"process": lambda self, img: None})()
        mock_fd = type("MockFD", (), {"process": lambda self, img: None})()

        def fake_selfie(*args, **kwargs):
            return mock_seg

        def fake_face(*args, **kwargs):
            return mock_fd

        monkeypatch.setattr("ninja_silhouette._build_selfie_segmentation", fake_selfie)
        monkeypatch.setattr("ninja_silhouette._build_face_detection", fake_face)

        sil = NinjaSilhouette()
        assert sil.available is True
        assert sil._seg is mock_seg
        assert sil._fd is mock_fd

    def test_init_seg_failure_falls_back(self, monkeypatch):
        """selfie_segmentation 빌드 실패 → available=False."""
        def raise_err(*args, **kwargs):
            raise RuntimeError("mediapipe seg not available")

        def fake_face(*args, **kwargs):
            return type("MockFD", (), {})()

        monkeypatch.setattr("ninja_silhouette._build_selfie_segmentation", raise_err)
        monkeypatch.setattr("ninja_silhouette._build_face_detection", fake_face)

        sil = NinjaSilhouette()
        assert sil.available is False

    def test_init_face_failure_keeps_seg(self, monkeypatch):
        """face_detection 실패해도 seg가 살아있으면 available=True (eye reveal만 비활성)."""
        mock_seg = type("MockSeg", (), {"process": lambda self, img: None})()

        def fake_selfie(*args, **kwargs):
            return mock_seg

        def raise_err(*args, **kwargs):
            raise RuntimeError("face detection failed")

        monkeypatch.setattr("ninja_silhouette._build_selfie_segmentation", fake_selfie)
        monkeypatch.setattr("ninja_silhouette._build_face_detection", raise_err)

        sil = NinjaSilhouette()
        assert sil.available is True
        assert sil._fd is None
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ninja_silhouette.py::TestNinjaSilhouetteInit -v`
Expected: FAIL — `ImportError: cannot import name 'NinjaSilhouette'`

- [ ] **Step 3: NinjaSilhouette 클래스 + 헬퍼 함수 구현**

`ninja_silhouette.py` 끝에 추가:

```python
def _build_selfie_segmentation(model_selection: int = 1):
    """mediapipe selfie segmentation 인스턴스 생성. 실패 시 예외 raise."""
    import mediapipe as mp  # NOTE: 모듈 상단 dual-import 블록에 없는 이유 — 빌드 실패 격리
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=model_selection)


def _build_face_detection(model_selection: int = 0, min_confidence: float = 0.5):
    """mediapipe face detection 인스턴스 생성. 실패 시 예외 raise."""
    import mediapipe as mp
    return mp.solutions.face_detection.FaceDetection(
        model_selection=model_selection, min_detection_confidence=min_confidence
    )


class NinjaSilhouette:
    """카메라 frame → 사람 마스크 + 양 눈 좌표.

    Selfie Seg 빌드 실패 시 self.available=False (게임은 spotlight 폴백).
    Face Detection 빌드 실패는 self._fd=None (사람 그림자만, 눈 reveal 생략).
    """

    def __init__(self, seg_model: int = 1, face_model: int = 0, min_face_confidence: float = 0.5):
        self.available: bool = False
        self._seg = None
        self._fd = None
        self._mask_threshold: float = 0.5  # theme.PERSON_MASK_THRESHOLD와 별도 — 기본값

        try:
            self._seg = _build_selfie_segmentation(model_selection=seg_model)
        except Exception as e:
            print(f"⚠️ Selfie Segmentation 빌드 실패: {e} — silhouette 비활성, spotlight 폴백")
            return  # available=False 유지

        try:
            self._fd = _build_face_detection(model_selection=face_model, min_confidence=min_face_confidence)
        except Exception as e:
            print(f"⚠️ Face Detection 빌드 실패: {e} — 눈 reveal 비활성")
            self._fd = None

        self.available = True

    def close(self) -> None:
        """리소스 해제."""
        if self._seg is not None and hasattr(self._seg, "close"):
            self._seg.close()
        if self._fd is not None and hasattr(self._fd, "close"):
            self._fd.close()
        self._seg = None
        self._fd = None
        self.available = False
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ninja_silhouette.py -v`
Expected: PASS (5 cases)

- [ ] **Step 5: Commit**

```bash
git add games/01_cafe_ninja/src/ninja_silhouette.py games/01_cafe_ninja/tests/test_ninja_silhouette.py
git commit -m "feat(cafe-ninja): add NinjaSilhouette init + graceful fallback"
```

---

## Task 4: NinjaSilhouette.process() — 마스크 + 눈 좌표 추출

**Files:**
- Modify: `games/01_cafe_ninja/src/ninja_silhouette.py`
- Modify: `games/01_cafe_ninja/tests/test_ninja_silhouette.py`

- [ ] **Step 1: 실패 테스트 작성 — process() 정상/폴백 케이스**

`test_ninja_silhouette.py` 하단에 추가:

```python
class TestNinjaSilhouetteProcess:
    @pytest.fixture
    def fake_frame(self):
        return np.full((480, 640, 3), 128, dtype=np.uint8)

    def _make_sil(self, monkeypatch, seg_mask=None, eye_norm_points=None, fd_present=True):
        """공통 헬퍼: seg/fd mock + NinjaSilhouette 인스턴스 반환."""
        class FakeSegResult:
            segmentation_mask = seg_mask

        class FakeKeypoint:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class FakeLocation:
            def __init__(self, kps):
                self.relative_keypoints = kps

        class FakeDetection:
            def __init__(self, kps):
                self.location_data = FakeLocation(kps)

        class FakeFDResult:
            detections = (
                [FakeDetection([FakeKeypoint(*p) for p in eye_norm_points])]
                if eye_norm_points is not None else None
            )

        mock_seg = type("MockSeg", (), {"process": lambda self, img: FakeSegResult()})()
        mock_fd = type("MockFD", (), {"process": lambda self, img: FakeFDResult()})()

        monkeypatch.setattr("ninja_silhouette._build_selfie_segmentation", lambda **kw: mock_seg)
        if fd_present:
            monkeypatch.setattr("ninja_silhouette._build_face_detection", lambda **kw: mock_fd)
        else:
            monkeypatch.setattr("ninja_silhouette._build_face_detection",
                                lambda **kw: (_ for _ in ()).throw(RuntimeError("no fd")))

        return NinjaSilhouette()

    def test_process_person_and_eyes(self, monkeypatch, fake_frame):
        """사람 검출 + 양 눈 검출 정상 케이스."""
        mask_prob = np.zeros((480, 640), dtype=np.float32)
        mask_prob[100:380, 200:440] = 0.9  # 사람 영역
        # eye keypoints: 정규화 좌표 (오른눈 0.45,0.30 / 왼눈 0.55,0.30)
        sil = self._make_sil(monkeypatch, seg_mask=mask_prob, eye_norm_points=[(0.45, 0.30), (0.55, 0.30)])

        r = sil.process(fake_frame)
        assert r.person_mask is not None
        assert r.person_mask.shape == (480, 640)
        assert r.person_mask.dtype == np.uint8
        assert r.person_mask[200, 300] == 255  # 사람 영역
        assert r.person_mask[10, 10] == 0      # 배경 영역
        assert len(r.eye_points) == 2
        # (0.45 * 640, 0.30 * 480) ≈ (288, 144)
        assert r.eye_points[0] == (288, 144)
        assert r.eye_points[1] == (352, 144)
        assert r.detected is True

    def test_process_no_person(self, monkeypatch, fake_frame):
        """사람 미검출 (전부 0)."""
        mask_prob = np.zeros((480, 640), dtype=np.float32)
        sil = self._make_sil(monkeypatch, seg_mask=mask_prob, eye_norm_points=None)
        r = sil.process(fake_frame)
        assert r.person_mask is not None
        assert r.person_mask.sum() == 0
        assert r.eye_points == []
        assert r.detected is False

    def test_process_no_face(self, monkeypatch, fake_frame):
        """사람은 있는데 얼굴 검출 실패."""
        mask_prob = np.full((480, 640), 0.9, dtype=np.float32)
        sil = self._make_sil(monkeypatch, seg_mask=mask_prob, eye_norm_points=None)
        r = sil.process(fake_frame)
        assert r.person_mask.sum() > 0
        assert r.eye_points == []
        assert r.detected is False  # 얼굴 검출 안 됨

    def test_process_no_fd_module(self, monkeypatch, fake_frame):
        """face_detection 빌드 실패 (self._fd=None) — 마스크만 반환."""
        mask_prob = np.full((480, 640), 0.9, dtype=np.float32)
        sil = self._make_sil(monkeypatch, seg_mask=mask_prob, eye_norm_points=None, fd_present=False)
        assert sil._fd is None
        r = sil.process(fake_frame)
        assert r.person_mask.sum() > 0
        assert r.eye_points == []

    def test_process_unavailable_returns_empty(self, monkeypatch, fake_frame):
        """available=False일 때 빈 결과."""
        def raise_err(**kw):
            raise RuntimeError("no seg")
        monkeypatch.setattr("ninja_silhouette._build_selfie_segmentation", raise_err)
        monkeypatch.setattr("ninja_silhouette._build_face_detection", lambda **kw: object())
        sil = NinjaSilhouette()
        assert sil.available is False
        r = sil.process(fake_frame)
        assert r.person_mask is None
        assert r.eye_points == []
        assert r.detected is False
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ninja_silhouette.py::TestNinjaSilhouetteProcess -v`
Expected: FAIL — `AttributeError: 'NinjaSilhouette' object has no attribute 'process'`

- [ ] **Step 3: process() 구현**

`ninja_silhouette.py`의 `NinjaSilhouette` 클래스에 메서드 추가:

```python
    def process(self, frame_bgr: np.ndarray) -> SilhouetteResult:
        """프레임 1회 추론. mediapipe는 RGB를 받으므로 색공간 변환 후 호출."""
        if not self.available or self._seg is None:
            return SilhouetteResult()

        import cv2  # 로컬 import 방지 위해 모듈 상단으로 이동도 가능 — 일관성 위해 game.py에서도 cv2 import됨
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1) Selfie Segmentation → person_mask (uint8 0/255)
        seg_res = self._seg.process(rgb)
        seg_mask_prob = getattr(seg_res, "segmentation_mask", None)
        if seg_mask_prob is None:
            return SilhouetteResult()

        person_mask = np.where(seg_mask_prob > self._mask_threshold, 255, 0).astype(np.uint8)
        person_present = bool(person_mask.sum() > 0)

        # 2) Face Detection → eye_points (픽셀 좌표)
        eye_points: List[Tuple[int, int]] = []
        if self._fd is not None:
            fd_res = self._fd.process(rgb)
            detections = getattr(fd_res, "detections", None) or []
            if detections:
                h, w = frame_bgr.shape[:2]
                kps = detections[0].location_data.relative_keypoints
                # mediapipe face detection keypoint 순서:
                # 0=right eye, 1=left eye, 2=nose tip, 3=mouth, 4=right ear, 5=left ear
                for idx in (0, 1):
                    if idx >= len(kps):
                        continue
                    kp = kps[idx]
                    px = int(round(kp.x * w))
                    py = int(round(kp.y * h))
                    # 화면 밖 좌표는 클립 (mediapipe가 가끔 음수/오버 반환)
                    if 0 <= px < w and 0 <= py < h:
                        eye_points.append((px, py))

        detected = person_present and len(eye_points) > 0
        return SilhouetteResult(person_mask=person_mask, eye_points=eye_points, detected=detected)
```

`_mask_threshold` 초기값을 `__init__`에서 `theme.PERSON_MASK_THRESHOLD`로 끌어올 수도 있으나, theme 의존을 모듈에 박지 않기 위해 game.py가 init 시 주입하는 방식도 가능. 본 Task에서는 기본값(0.5) 유지.

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ninja_silhouette.py -v`
Expected: PASS (모든 케이스, 약 10개)

- [ ] **Step 5: Commit**

```bash
git add games/01_cafe_ninja/src/ninja_silhouette.py games/01_cafe_ninja/tests/test_ninja_silhouette.py
git commit -m "feat(cafe-ninja): NinjaSilhouette.process — mask + eye keypoints"
```

---

## Task 5: ui_renderer — mask 헬퍼 (_build_circular_mask, _build_disk_mask)

**Files:**
- Modify: `games/01_cafe_ninja/src/ui_renderer.py`
- Create: `games/01_cafe_ninja/tests/test_ui_renderer.py`

- [ ] **Step 1: 실패 테스트 작성**

`games/01_cafe_ninja/tests/test_ui_renderer.py`:

```python
"""ui_renderer apply_silhouette 및 헬퍼 테스트."""
import numpy as np
import pytest

import ui_renderer as ui


class TestBuildDiskMask:
    def test_basic_disk(self):
        """반지름 10, feather 0인 원형 마스크."""
        mask = ui._build_disk_mask((100, 100), center=(50, 50), radius=10, feather=0)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() <= mask.max() <= 1.0
        assert mask[50, 50] == pytest.approx(1.0, abs=0.01)  # 중심
        assert mask[0, 0] == pytest.approx(0.0, abs=0.01)    # 멀리

    def test_disk_with_feather(self):
        """feather > 0이면 외곽이 부드러움."""
        mask = ui._build_disk_mask((100, 100), center=(50, 50), radius=10, feather=5)
        # 중심은 1.0
        assert mask[50, 50] == pytest.approx(1.0, abs=0.01)
        # 반지름 + feather 외부는 거의 0
        assert mask[50, 80] < 0.05

    def test_disk_clipped_to_frame(self):
        """중심이 화면 밖이거나 가장자리여도 크래시 X."""
        mask = ui._build_disk_mask((100, 100), center=(95, 95), radius=20, feather=5)
        assert mask.shape == (100, 100)
        # 화면 안 영역엔 값 있어야 함
        assert mask.sum() > 0


class TestBuildCircularMask:
    def test_from_bbox(self):
        """손 bbox 기준 원형 마스크."""
        # bbox = (x, y, w, h) 또는 (x1, y1, x2, y2) — Task 5 구현에서 확정
        bbox = (40, 40, 60, 60)  # 우상단(x1,y1) ~ 좌하단(x2,y2): 중심 (50,50), 대각선 sqrt(800) ≈ 28
        mask = ui._build_circular_mask((100, 100), bbox=bbox, feather=10)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32
        assert mask[50, 50] == pytest.approx(1.0, abs=0.05)

    def test_circular_mask_min_radius(self):
        """bbox 작아도 최소 반지름 80px 보장 (spotlight 가독성)."""
        bbox = (50, 50, 52, 52)  # 매우 작은 bbox
        mask = ui._build_circular_mask((200, 200), bbox=bbox, feather=10, min_radius=80)
        # 반지름 80이므로 (50,50)에서 (130,50)까지 mask 영향 있어야 함
        assert mask[51, 90] > 0.5
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ui_renderer.py -v`
Expected: FAIL — `AttributeError: module 'ui_renderer' has no attribute '_build_disk_mask'`

- [ ] **Step 3: 헬퍼 함수 구현**

`ui_renderer.py` 하단(`apply_hand_spotlight` 근처)에 추가:

```python
def _build_disk_mask(shape, center, radius, feather):
    """원형 마스크 (float32 0~1). 중심=1, 외곽=Gaussian 감쇠.

    shape: (H, W). center: (x, y) 픽셀 좌표. radius: 단단한 영역 반지름.
    feather: 경계 부드러움 (sigma 픽셀). 0이면 hard edge.
    """
    h, w = shape
    cx, cy = center
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    if feather <= 0:
        return (dist <= radius).astype(np.float32)
    # dist <= radius → 1.0, dist > radius → exp(-((dist-radius)^2)/(2*feather^2))
    inside = dist <= radius
    outside_falloff = np.exp(-((dist - radius) ** 2) / (2 * feather * feather))
    mask = np.where(inside, 1.0, outside_falloff).astype(np.float32)
    return mask


def _build_circular_mask(shape, bbox, feather, min_radius=80):
    """손 bbox 기준 원형 마스크.

    bbox: (x1, y1, x2, y2) — 21개 landmark의 axis-aligned bbox.
    중심 = bbox 중심, 반지름 = max(min_radius, 대각선 * 0.8).
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    diag = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    radius = max(min_radius, int(diag * 0.8))
    return _build_disk_mask(shape, (cx, cy), radius, feather)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ui_renderer.py -v`
Expected: PASS (5 cases)

- [ ] **Step 5: Commit**

```bash
git add games/01_cafe_ninja/src/ui_renderer.py games/01_cafe_ninja/tests/test_ui_renderer.py
git commit -m "feat(cafe-ninja): add mask helpers for silhouette compositing"
```

---

## Task 6: ui_renderer.apply_silhouette() — 배경 + 사람 그림자 합성

**Files:**
- Modify: `games/01_cafe_ninja/src/ui_renderer.py`
- Modify: `games/01_cafe_ninja/tests/test_ui_renderer.py`

- [ ] **Step 1: 실패 테스트 작성 — 사람/배경 기본 합성**

`test_ui_renderer.py` 하단에 추가:

```python
class TestApplySilhouetteBasic:
    @pytest.fixture
    def setup_frames(self):
        h, w = 100, 100
        # 카메라: 회색 균일
        camera = np.full((h, w, 3), 128, dtype=np.uint8)
        # 배경(town.jpg 대용): 노란색 균일
        bg = np.full((h, w, 3), 0, dtype=np.uint8)
        bg[:, :, 1] = 200  # green channel
        bg[:, :, 2] = 200  # red channel
        # 사람 마스크: 중앙 사각형
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        return camera, bg, mask

    def test_person_mask_none_fills_with_bg(self, setup_frames):
        """사람 미검출 → frame 전체가 bg_image."""
        camera, bg, _ = setup_frames
        frame = camera.copy()
        ui.apply_silhouette(frame, person_mask=None, eye_points=[], hand_bbox=None, bg_image=bg)
        # bg의 BGR=(0,200,200), 모든 픽셀 일치
        assert frame[10, 10, 0] == 0
        assert frame[10, 10, 1] == 200
        assert frame[10, 10, 2] == 200
        assert frame[50, 50, 1] == 200

    def test_person_area_tinted(self, setup_frames):
        """사람 영역(마스크 안)은 SILHOUETTE_TINT_BGR(40,25,55), 배경은 bg."""
        import theme
        camera, bg, mask = setup_frames
        frame = camera.copy()
        ui.apply_silhouette(frame, person_mask=mask, eye_points=[], hand_bbox=None, bg_image=bg)
        # 사람 영역 중앙 (50,50): tint
        assert frame[50, 50, 0] == theme.SILHOUETTE_TINT_BGR[0]
        assert frame[50, 50, 1] == theme.SILHOUETTE_TINT_BGR[1]
        assert frame[50, 50, 2] == theme.SILHOUETTE_TINT_BGR[2]
        # 배경 영역 (10,10): bg 색
        assert frame[10, 10, 1] == 200
        assert frame[10, 10, 2] == 200
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ui_renderer.py::TestApplySilhouetteBasic -v`
Expected: FAIL — `AttributeError: module 'ui_renderer' has no attribute 'apply_silhouette'`

- [ ] **Step 3: apply_silhouette() 기본 구현 (사람 + 배경만, reveal은 다음 Task)**

`ui_renderer.py`의 mask 헬퍼 다음 줄에 추가. import 블록 상단에 이미 있어야 하지만, 없으면 추가:

```python
def apply_silhouette(frame, person_mask, eye_points, hand_bbox, bg_image):
    """배경(원색) + 사람(자주톤 그림자) + 손/눈 reveal 합성 (in-place).

    person_mask: uint8 (H, W) 0~255. None이면 frame 전체를 bg_image로 채움.
    eye_points: [(x, y), ...] 최대 2개. 빈 리스트면 눈 reveal 생략.
    hand_bbox: (x1, y1, x2, y2). None이면 손 reveal 생략.
    bg_image: BGR (H, W, 3), frame과 동일 크기.
    """
    if bg_image is None or bg_image.shape != frame.shape:
        return  # 안전 폴백: 합성 안 함

    camera_pixels = frame.copy()

    # 1) 배경 영역 채우기
    if person_mask is None:
        frame[:] = bg_image
        return

    if person_mask.shape != frame.shape[:2]:
        person_mask = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

    mask_f = (person_mask.astype(np.float32) / 255.0)[..., None]  # (H, W, 1)

    blended = camera_pixels.astype(np.float32) * mask_f + bg_image.astype(np.float32) * (1.0 - mask_f)

    # 2) 사람 영역을 자주톤 평면으로 덮기
    tint = np.full_like(camera_pixels, theme.SILHOUETTE_TINT_BGR, dtype=np.uint8)
    person_alpha = mask_f * theme.SILHOUETTE_ALPHA
    blended = tint.astype(np.float32) * person_alpha + blended * (1.0 - person_alpha)

    frame[:] = np.clip(blended, 0, 255).astype(np.uint8)

    # 3) 손 영역 reveal (다음 Task에서 추가)
    # 4) 눈 영역 reveal (다음 Task에서 추가)
```

**필수 확인**: 파일 상단에 `from . import theme as theme_mod` 등으로 theme이 import되어 있어야 함. 기존 ui_renderer.py에서 theme을 어떻게 import하는지 확인 후 동일 패턴 사용:

Run: `grep -n "import theme" games/01_cafe_ninja/src/ui_renderer.py`

Expected 출력에 `from . import theme` 또는 `import theme` 형태가 있어야 함. 없으면 dual-import 블록(파일 상단)에 추가.

`np` (numpy)는 이미 ui_renderer 상단에 import됨. `cv2`도 마찬가지.

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ui_renderer.py -v`
Expected: PASS (7 cases — 헬퍼 5 + apply_silhouette 기본 2)

- [ ] **Step 5: Commit**

```bash
git add games/01_cafe_ninja/src/ui_renderer.py games/01_cafe_ninja/tests/test_ui_renderer.py
git commit -m "feat(cafe-ninja): apply_silhouette basic (bg + person tint)"
```

---

## Task 7: apply_silhouette — 손/눈 reveal + person_mask 클립

**Files:**
- Modify: `games/01_cafe_ninja/src/ui_renderer.py`
- Modify: `games/01_cafe_ninja/tests/test_ui_renderer.py`

- [ ] **Step 1: 실패 테스트 작성 — reveal 동작**

`test_ui_renderer.py` 하단에 추가:

```python
class TestApplySilhouetteReveal:
    @pytest.fixture
    def setup_frames(self):
        h, w = 200, 200
        camera = np.full((h, w, 3), 128, dtype=np.uint8)
        camera[100, 100] = [200, 100, 50]  # 마커 픽셀
        camera[40, 100] = [10, 20, 30]     # 눈 위치 마커
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:, :, 1] = 200
        # 사람 마스크: 거의 전체 (마진 5px 제외)
        mask = np.full((h, w), 255, dtype=np.uint8)
        mask[:5, :] = 0
        mask[-5:, :] = 0
        mask[:, :5] = 0
        mask[:, -5:] = 0
        return camera, bg, mask

    def test_hand_reveal_inside_person(self, setup_frames):
        """손 bbox가 사람 영역 안 → 손 중앙은 카메라 픽셀."""
        camera, bg, mask = setup_frames
        frame = camera.copy()
        hand_bbox = (80, 80, 120, 120)  # 중심 (100, 100), 사람 영역 안
        ui.apply_silhouette(frame, person_mask=mask, eye_points=[],
                            hand_bbox=hand_bbox, bg_image=bg)
        # 손 중심 (100, 100)은 카메라 픽셀 [200, 100, 50]에 가깝게 복원
        assert frame[100, 100, 0] > 150  # 카메라 픽셀 우세
        assert frame[100, 100, 1] < 130

    def test_eye_reveal_inside_person(self, setup_frames):
        """눈 좌표가 사람 영역 안 → 작은 원 안 픽셀이 카메라값."""
        import theme
        camera, bg, mask = setup_frames
        frame = camera.copy()
        ui.apply_silhouette(frame, person_mask=mask, eye_points=[(100, 40)],
                            hand_bbox=None, bg_image=bg)
        # 눈 중심 (100, 40): 카메라 픽셀 [10, 20, 30]에 가까움
        assert frame[40, 100, 0] < 40
        assert frame[40, 100, 1] < 50

    def test_eye_reveal_outside_person_no_effect(self, setup_frames):
        """눈 좌표가 배경 영역 → reveal 없음 (배경 그대로)."""
        camera, bg, mask = setup_frames
        frame = camera.copy()
        # (2, 2)는 마스크 외부 (배경)
        ui.apply_silhouette(frame, person_mask=mask, eye_points=[(2, 2)],
                            hand_bbox=None, bg_image=bg)
        # 배경 색 (0, 200, 0)
        assert frame[2, 2, 1] == 200

    def test_hand_bbox_none_no_reveal(self, setup_frames):
        """hand_bbox=None → 손 reveal 생략, 사람 영역은 tint 그대로."""
        import theme
        camera, bg, mask = setup_frames
        frame = camera.copy()
        ui.apply_silhouette(frame, person_mask=mask, eye_points=[],
                            hand_bbox=None, bg_image=bg)
        # 사람 영역 임의 픽셀: tint
        assert frame[100, 100, 0] == theme.SILHOUETTE_TINT_BGR[0]
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ui_renderer.py::TestApplySilhouetteReveal -v`
Expected: FAIL — hand/eye reveal 미구현 (frame[100,100]이 tint 그대로)

- [ ] **Step 3: apply_silhouette에 reveal 단계 추가**

Task 6에서 작성한 `apply_silhouette` 함수의 두 주석 `# 3) 손 영역 reveal (다음 Task에서 추가)` 와 `# 4) 눈 영역 reveal (다음 Task에서 추가)` 를 다음 코드로 통째 교체. Task 6의 `frame[:] = np.clip(blended, 0, 255).astype(np.uint8)` 줄은 **삭제** — 이번 블록의 마지막 줄이 최종 쓰기 담당:

```python
    # frame은 아직 tint 적용 전 (Task 6의 frame[:] 줄을 이번 단계 마지막으로 이동)
    # 위 blended 변수가 tint 적용 후 상태 — 이걸 기준으로 reveal 합성
    cam_f = camera_pixels.astype(np.float32)
    mask_2d = mask_f[..., 0]

    # 3) 손 영역 reveal (camera_pixels 복원, person_mask 외부는 클립)
    if hand_bbox is not None:
        hand_mask = _build_circular_mask(frame.shape[:2], hand_bbox, feather=theme.HAND_REVEAL_FEATHER)
        hand_mask = hand_mask * mask_2d  # 배경 영역엔 노출 X
        hm = hand_mask[..., None]
        blended = cam_f * hm + blended * (1.0 - hm)

    # 4) 눈 영역 reveal (양 눈 각각 작은 원)
    for (ex, ey) in eye_points:
        eye_mask = _build_disk_mask(frame.shape[:2], (ex, ey),
                                    radius=theme.EYE_REVEAL_RADIUS,
                                    feather=theme.EYE_REVEAL_FEATHER)
        eye_mask = eye_mask * mask_2d
        em = eye_mask[..., None]
        blended = cam_f * em + blended * (1.0 - em)

    frame[:] = np.clip(blended, 0, 255).astype(np.uint8)
```

**주의**: Task 6의 코드에서 `frame[:] = np.clip(blended, 0, 255).astype(np.uint8)` 줄을 반드시 삭제. 위 블록의 동명 줄이 단일 frame 쓰기 담당. `blended` 변수는 Task 6 단계의 tint 적용 결과를 그대로 사용 (선언 줄은 Task 6에 그대로 유지).

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/test_ui_renderer.py -v`
Expected: PASS (11 cases — 헬퍼 5 + 기본 2 + reveal 4)

- [ ] **Step 5: Commit**

```bash
git add games/01_cafe_ninja/src/ui_renderer.py games/01_cafe_ninja/tests/test_ui_renderer.py
git commit -m "feat(cafe-ninja): apply_silhouette hand/eye reveal with mask clip"
```

---

## Task 8: game.py 통합 — silhouette init + run + render 분기

**Files:**
- Modify: `games/01_cafe_ninja/src/game.py`

- [ ] **Step 1: dual-import 블록에 ninja_silhouette 추가**

`game.py`의 dual-import 블록 (line 40~75 부근). `try:` 블록에 추가:

```python
    from .ninja_silhouette import NinjaSilhouette
```

(기존 `from .cafe_ninja_sprite import ...` 바로 다음 줄에 추가)

`except ImportError:` 블록에 추가:

```python
    from ninja_silhouette import NinjaSilhouette
```

(기존 `from cafe_ninja_sprite import ...` 바로 다음 줄에 추가)

- [ ] **Step 2: `__init__`에 silhouette 인스턴스 추가**

`game.py`의 `CafeNinjaGame.__init__` (또는 클래스명 — `grep -n "def __init__" games/01_cafe_ninja/src/game.py`로 확인) 내부, `self.background = None` 다음 줄에 추가:

```python
        self.silhouette = NinjaSilhouette()
        if self.silhouette.available:
            print("🥷 Silhouette 활성 (Selfie Seg + Face Detection)")
        else:
            print("⚠️ Silhouette 비활성 — spotlight 폴백")
```

메모리 규칙 `feedback-init-must-define-all-attrs`에 따라 `self.silhouette`은 반드시 `__init__`에서 초기화 — 위 한 줄이 default 보장.

- [ ] **Step 3: run 루프에서 silhouette.process() 호출**

`game.py`의 run 루프 (메인 while loop, `cv2.imshow` 이전). 기존 `tip_px = ...` 또는 `hand_bbox = ...` 계산 직후에 추가:

```python
            if self.phase == GamePhase.PLAYING and self.silhouette.available:
                sil_result = self.silhouette.process(frame)
                person_mask = sil_result.person_mask
                eye_points = sil_result.eye_points
            else:
                person_mask = None
                eye_points = []
```

PLAYING 페이즈에서만 추론 — 선택/READY/GAME_OVER에서는 안내문 노출 필요하므로 silhouette 비활성. CPU 비용도 절약.

- [ ] **Step 4: render 분기 — silhouette 활성 시 apply_silhouette, 아니면 폴백**

`game.py`의 `render()` 메서드. 기존:

```python
        if self.background is not None:
            ui.draw_background(frame, self.background)
```

위 패턴이 있는 곳(line 379~380 부근). 그 다음 또는 비슷한 곳에 있는 `ui.apply_hand_spotlight(frame, hand_bbox)` 호출을 다음과 같이 통합 교체:

```python
        # PLAYING 페이즈에서만 silhouette 합성 (선택/READY/GAME_OVER 화면은 안내문 보여야 함)
        if (self.phase == GamePhase.PLAYING
                and self.silhouette.available
                and self.background is not None):
            ui.apply_silhouette(frame, person_mask, eye_points, hand_bbox, self.background)
        else:
            # 폴백: 기존 배경 블렌딩 + spotlight
            if self.background is not None:
                ui.draw_background(frame, self.background)
            if self.phase == GamePhase.PLAYING:
                ui.apply_hand_spotlight(frame, hand_bbox)
```

`render()` 메서드 시그니처에 `person_mask`, `eye_points` 추가:

```python
    def render(self, frame, tip_px, hand_bbox, person_mask=None, eye_points=None):
        if eye_points is None:
            eye_points = []
        # ... 기존 body ...
```

run 루프에서 `self.render(...)` 호출 부분도 인자 추가:

```python
            self.render(frame, tip_px, hand_bbox, person_mask, eye_points)
```

- [ ] **Step 5: 기존 unit test 통합 회귀로 회귀 없음 확인**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/ -q`
Expected: 모든 케이스 (기존 + Task 2~7 신규) PASS, 회귀 0건

- [ ] **Step 6: Commit**

```bash
git add games/01_cafe_ninja/src/game.py
git commit -m "feat(cafe-ninja): integrate ninja silhouette in game loop"
```

---

## Task 9: 두 모드 smoke 테스트 + 통합 회귀 + 최종 검증

**Files:** (수정 없음 — 검증만)

- [ ] **Step 1: 패키지 모드 import smoke**

Run:
```bash
.venv/bin/python -c "import importlib; importlib.import_module('games.01_cafe_ninja.src.game'); print('OK -m import')"
```
Expected: 출력 마지막 줄에 `OK -m import`. mediapipe / pygame 초기 로그는 정상.

- [ ] **Step 2: 직접 실행 import smoke**

Run:
```bash
.venv/bin/python -c "import sys; sys.path.insert(0, '/home/gjkong/PlayWait/games/01_cafe_ninja/src'); import game; print('OK direct import')"
```

Expected: `OK direct import` 출력. mediapipe / pygame 로드 로그는 정상.

이 두 모드는 메모리 규칙 [[feedback-no-lazy-imports]] 핵심 — 둘 다 통과해야 함. import 단계만 검증하며 게임 루프는 진입하지 않음 (카메라 점유 회피).

- [ ] **Step 3: 카페닌자 단위 테스트 회귀**

Run: `.venv/bin/pytest games/01_cafe_ninja/tests/ -q`
Expected: 기존 cases + 신규 cases (silhouette ~10 + ui_renderer ~11) 모두 PASS.

- [ ] **Step 4: 전체 통합 회귀**

Run: `.venv/bin/pytest games/ -q`
Expected: 678+ cases 모두 PASS, 회귀 0건.

- [ ] **Step 5: 카메라 실기 검증 (사용자 진행)**

Run: `.venv/bin/python -m games.01_cafe_ninja.src.game`

체크 항목:
- 시작 시 `🥷 Silhouette 활성` 로그 출력
- 사람 등장 전: 배경(town.jpg)만 보임 (그림자 X)
- 사람 정면: 자주톤 평면 실루엣 + 양 눈 작은 원 카메라 노출 + 손 영역 카메라 노출
- 사람 측면: 눈 reveal 사라짐, 사람 그림자만 (자연스러움)
- 손 영역에서 떨어지는 메뉴/Kunai 정확히 슬라이스 가능
- 30 FPS 유지 (저하 5 FPS 미만)

크래시 시: 시작 시점·페이즈·재현 절차 기록 → 다음 세션 수정.

- [ ] **Step 6: 일별 로그에 작업 기록 추가**

`docs/daily_logs/폴리싱1차_개발로그_2026-05-18.md`에 섹션 추가:

```markdown
## X. Cafe Ninja Silhouette (그림자 + 눈 복면) — silhouette design 구현

[spec 링크, 구현 결과, 카메라 실기 결과, 다음 작업]
```

- [ ] **Step 7: 최종 commit + 푸시 (사용자 확인 후)**

```bash
git add docs/daily_logs/폴리싱1차_개발로그_2026-05-18.md
git commit -m "docs(cafe-ninja): log silhouette implementation results"
```

푸시는 사용자 명시 승인 후 진행.

---

## 검증 완료 기준 체크리스트

- [ ] 신규 단위 테스트 모두 통과 (test_ninja_silhouette.py ~10 + test_ui_renderer.py ~11)
- [ ] 기존 카페닌자 회귀 0건
- [ ] 전체 통합 회귀 (`pytest games/ -q`) 0건
- [ ] 두 모드 smoke (`-m` + 직접) 통과
- [ ] 카메라 실기: 사람 정면/측면/미등장 3 시나리오 모두 정상
- [ ] 30 FPS 유지

## 메모리 규칙 체크리스트

- [ ] **feedback-no-lazy-imports**: ninja_silhouette는 game.py dual-import 블록에 추가됨, lazy import 0건
- [ ] **feedback-module-name-prefix**: `ninja_silhouette.py` (게임 prefix 적용)
- [ ] **feedback-init-must-define-all-attrs**: `self.silhouette`이 `__init__`에서 1회 정의됨
- [ ] **feedback-ui-renderer-anchor-fix**: 이번 작업은 anchor 영역 손대지 않음 (영향 X)
