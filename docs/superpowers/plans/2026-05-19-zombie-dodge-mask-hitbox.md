# Zombie Dodge — 실루엣 Hitbox 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 08_zombie_dodge의 사각형 AABB hitbox를 MediaPipe SelfieSegmentation 마스크 기반 픽셀 교집합 판정으로 교체하고 시각화도 마스크 외곽선으로 변경.

**Architecture:** `core/vision/person_mask.py` 신규 (재활용 가능 코어) → `08/src/body_tracker.py`의 DodgeBox/compute_dodge_box를 PersonMask/compute_person_mask로 교체 → `08/src/dodge_judge.py`의 aabb_circle_collide를 mask_circle_collide로 교체 → `08/src/ui_renderer.py`의 박스 그리기를 contour outline으로 교체.

**Tech Stack:** Python 3.12, MediaPipe SelfieSegmentation (model_selection=0), OpenCV (cv2.findContours/drawContours/resize), NumPy, pytest.

**참고:**
- Spec: `docs/superpowers/specs/2026-05-19-zombie-dodge-mask-hitbox-design.md`
- 메모리: [[feedback-no-lazy-imports]], [[feedback-init-must-define-all-attrs]], [[feedback-module-name-prefix]]
- PlayWait는 git 미초기화 상태 → "Commit" 대신 "체크포인트(파일 저장 + 테스트 통과)"로 진행.

---

## File Structure

| 파일 | 변경 | 책임 |
|---|---|---|
| `core/vision/person_mask.py` | 신규 | SelfieSegmentation 래퍼. 다른 게임에서도 재활용 |
| `tests/core/vision/test_person_mask.py` | 신규 | 코어 모듈 단위 테스트 |
| `games/08_zombie_dodge/src/body_tracker.py` | 교체 | DodgeBox/compute_dodge_box 삭제 → PersonMask/compute_person_mask |
| `games/08_zombie_dodge/tests/test_body_tracker.py` | 교체 | 마스크 기반 케이스로 마이그레이션 |
| `games/08_zombie_dodge/src/dodge_judge.py` | 수정 | aabb_circle_collide 삭제 → mask_circle_collide. evaluate_frame 시그니처 변경 |
| `games/08_zombie_dodge/tests/test_dodge_judge.py` | 교체 | 합성 mask 기반 케이스 |
| `games/08_zombie_dodge/src/ui_renderer.py` | 수정 | draw_dodge_box 삭제 → draw_person_outline |
| `games/08_zombie_dodge/src/game.py` | 수정 | PersonMaskDetector 초기화 + 매 frame 호출 + 렌더링 호출 |

`core/vision/__init__.py`는 기존 파일 그대로(import 추가 없음 — feedback-no-lazy-imports 정신).

---

## Task 1: `core/vision/person_mask.py` — PersonMaskDetector

**Files:**
- Create: `core/vision/person_mask.py`
- Create: `tests/core/vision/test_person_mask.py`
- Create: `tests/core/vision/__init__.py` (빈 파일, pytest discovery)
- Create: `tests/core/__init__.py` (없으면)
- Create: `tests/__init__.py` (없으면)

- [ ] **Step 1: 테스트 디렉토리 빈 `__init__.py` 생성**

확인 후 없으면 빈 파일 생성:
```bash
[ -f tests/__init__.py ] || touch tests/__init__.py
[ -f tests/core/__init__.py ] || touch tests/core/__init__.py
[ -f tests/core/vision/__init__.py ] || touch tests/core/vision/__init__.py
```

- [ ] **Step 2: failing 테스트 작성 — `tests/core/vision/test_person_mask.py`**

```python
"""PersonMaskDetector 단위 테스트.

실제 MediaPipe는 mock — 다운샘플/upscale/threshold 로직만 검증.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.vision.person_mask import PersonMaskDetector


def _make_soft_mask(h, w, value=0.8):
    """SelfieSegmentation이 반환하는 soft mask (float, 0~1) 모사."""
    return np.full((h, w), value, dtype=np.float32)


@patch("core.vision.person_mask.mp_solutions")
def test_process_returns_binary_mask(mock_mp):
    """soft mask threshold 0.5 → binary 0/1 uint8."""
    fake_result = MagicMock()
    fake_result.segmentation_mask = _make_soft_mask(180, 240, value=0.8)
    fake_seg = MagicMock()
    fake_seg.process.return_value = fake_result
    mock_mp.selfie_segmentation.SelfieSegmentation.return_value = fake_seg

    det = PersonMaskDetector(model_selection=0, downsample_height=180)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = det.process(frame)

    assert mask.dtype == np.uint8
    assert mask.shape == (720, 1280)
    assert mask.max() == 1
    assert mask.min() == 1   # 전 영역 0.8 > 0.5
    det.close()


@patch("core.vision.person_mask.mp_solutions")
def test_process_thresholds_at_half(mock_mp):
    """soft 0.3은 0, soft 0.7은 1."""
    soft = np.zeros((180, 240), dtype=np.float32)
    soft[:90, :] = 0.7   # 상단 1
    soft[90:, :] = 0.3   # 하단 0
    fake_result = MagicMock()
    fake_result.segmentation_mask = soft
    fake_seg = MagicMock()
    fake_seg.process.return_value = fake_result
    mock_mp.selfie_segmentation.SelfieSegmentation.return_value = fake_seg

    det = PersonMaskDetector(downsample_height=180)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = det.process(frame)

    assert mask[100, 100] == 1   # 상단 (upscale 후에도)
    assert mask[600, 100] == 0   # 하단
    det.close()


@patch("core.vision.person_mask.mp_solutions")
def test_context_manager(mock_mp):
    """with 블록 종료 시 close() 호출."""
    fake_seg = MagicMock()
    mock_mp.selfie_segmentation.SelfieSegmentation.return_value = fake_seg

    with PersonMaskDetector() as det:
        assert det is not None
    fake_seg.close.assert_called_once()
```

- [ ] **Step 3: 테스트 실행 — fail 확인**

Run: `.venv/bin/pytest tests/core/vision/test_person_mask.py -v`
Expected: `ModuleNotFoundError: No module named 'core.vision.person_mask'` 또는 ImportError

- [ ] **Step 4: 구현 — `core/vision/person_mask.py`**

```python
"""person_mask.py — MediaPipe SelfieSegmentation 픽셀 마스크 래퍼.

다른 게임에서도 재활용 가능한 코어 vision 모듈.

사용 예
-------
>>> from core.vision.person_mask import PersonMaskDetector
>>> with PersonMaskDetector(model_selection=0) as det:
...     mask = det.process(frame_bgr)   # ndarray (H, W) uint8, 1=사람

설계
----
- model_selection=0 일반 모델 (1보다 가벼움, 매장 카메라 거리에서 충분)
- 내부 처리는 다운샘플 (디폴트 180px 높이) → 원본 크기로 nearest upscale
- soft mask threshold 0.5로 binary 변환

Author: Stephen (gjkong)
Date: 2026-05-19
"""

import cv2
import numpy as np
import mediapipe as mp

# alias — 테스트에서 patch하기 편함
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
        # 다운샘플
        ds_w = int(w * self.downsample_height / h)
        small = cv2.resize(frame_bgr, (ds_w, self.downsample_height),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self._seg.process(rgb)
        soft = result.segmentation_mask   # float (h_ds, w_ds), 0~1
        binary_small = (soft > self.threshold).astype(np.uint8)
        # 원본 크기로 upscale (nearest)
        mask = cv2.resize(binary_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def close(self):
        self._seg.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
```

- [ ] **Step 5: 테스트 재실행 — pass 확인**

Run: `.venv/bin/pytest tests/core/vision/test_person_mask.py -v`
Expected: 3 passed

- [ ] **Step 6: 통합 회귀 — 전체 테스트 무결성**

Run: `.venv/bin/pytest -q 2>&1 | tail -5`
Expected: 전체 테스트 통과 (기존 678+ + 신규 3 = 681+). 회귀 없음.

- [ ] **Step 7: 체크포인트**

생성 파일 확인:
```bash
ls -la core/vision/person_mask.py tests/core/vision/test_person_mask.py
```

---

## Task 2: `body_tracker.py` 교체 — PersonMask / compute_person_mask

**Files:**
- Modify (전체 교체): `games/08_zombie_dodge/src/body_tracker.py`
- Modify (교체): `games/08_zombie_dodge/tests/test_body_tracker.py`

- [ ] **Step 1: 신규 테스트 작성 — `test_body_tracker.py` 전체 교체**

```python
"""body_tracker — PersonMask 추출 단위 테스트."""
import numpy as np
import pytest

from games.zombie_dodge.src.body_tracker import (
    PersonMask,
    compute_person_mask,
    DEFAULT_MIN_PIXEL_RATIO,
)
# Note: import path — 게임 폴더 모듈은 sys.path 처리에 따라
# `games.08_zombie_dodge.src.body_tracker` 형태일 수 있음.
# 기존 test_dodge_judge.py와 동일한 import 패턴 따라가기.


def _disk_mask(h, w, cx, cy, r):
    """원형 disk mask 합성 (사람 모양 대체)."""
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.uint8)


def test_compute_person_mask_valid():
    """충분히 큰 마스크 → PersonMask 반환."""
    raw = _disk_mask(720, 1280, 640, 360, 200)   # ~125k px ≫ 2%×720×1280=18432
    person = compute_person_mask(raw)
    assert person is not None
    assert person.mask.shape == (720, 1280)
    # bbox는 contour 외곽 사각형
    x1, y1, x2, y2 = person.bbox
    assert x1 < 640 < x2
    assert y1 < 360 < y2


def test_compute_person_mask_too_small_returns_none():
    """임계값 미만 마스크 → None."""
    raw = _disk_mask(720, 1280, 640, 360, 30)   # ~2800 px < 18432
    assert compute_person_mask(raw) is None


def test_compute_person_mask_empty_returns_none():
    raw = np.zeros((720, 1280), dtype=np.uint8)
    assert compute_person_mask(raw) is None


def test_default_min_pixel_ratio_is_two_percent():
    assert DEFAULT_MIN_PIXEL_RATIO == pytest.approx(0.02)


def test_person_mask_center():
    raw = _disk_mask(720, 1280, 640, 360, 200)
    person = compute_person_mask(raw)
    cx, cy = person.center
    # bbox 중앙은 disk 중심 근처
    assert abs(cx - 640) < 50
    assert abs(cy - 360) < 50
```

- [ ] **Step 2: 기존 import 경로 패턴 확인**

Run: `grep -nE "from games\.|import games\." games/08_zombie_dodge/tests/test_dodge_judge.py games/08_zombie_dodge/tests/conftest.py | head`
필요시 위 테스트의 import path를 실제 패턴에 맞춰 조정 (`from games.zombie_dodge.src.body_tracker` 또는 `from src.body_tracker` 등). conftest.py가 sys.path를 어떻게 손대는지 확인.

- [ ] **Step 3: 테스트 실행 — fail 확인**

Run: `.venv/bin/pytest games/08_zombie_dodge/tests/test_body_tracker.py -v`
Expected: ImportError (`PersonMask`, `compute_person_mask`, `DEFAULT_MIN_PIXEL_RATIO` 미정의)

- [ ] **Step 4: 구현 — `body_tracker.py` 전체 교체**

```python
"""body_tracker.py — SelfieSegmentation 마스크 → 회피 영역 (PersonMask).

순수 함수 모듈. MediaPipe 의존성 없음 — np.ndarray만 다룸.

이전 버전(2026-05-12)의 DodgeBox/compute_dodge_box는 픽셀 마스크 기반으로 교체.

Author: Stephen (gjkong)
Date: 2026-05-19 (W7 hitbox v2)
"""

from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================
# 1. 파라미터
# ============================================================
# 사람으로 인정하는 최소 픽셀 비율 (전체 frame 대비)
DEFAULT_MIN_PIXEL_RATIO = 0.02


# ============================================================
# 2. PersonMask
# ============================================================
@dataclass(frozen=True)
class PersonMask:
    """사람 픽셀 마스크 + 외곽 bbox.

    Attributes:
        mask: H×W uint8 0/1. 1=사람.
        bbox: (x1, y1, x2, y2) — contour 외곽 사각형. 렌더 영역 limit용.
    """
    mask: np.ndarray
    bbox: tuple

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


# ============================================================
# 3. 추출
# ============================================================
def compute_person_mask(
    raw_mask: np.ndarray,
    min_pixel_ratio: float = DEFAULT_MIN_PIXEL_RATIO,
) -> "PersonMask | None":
    """binary mask → PersonMask 또는 None.

    Args:
        raw_mask: H×W uint8 0/1. PersonMaskDetector.process() 결과.
        min_pixel_ratio: 사람 픽셀 수 / 전체 픽셀 수 임계값 (디폴트 0.02).

    Returns:
        PersonMask 또는 None (사람 미감지).
    """
    if raw_mask is None or raw_mask.size == 0:
        return None

    h, w = raw_mask.shape[:2]
    person_pixels = int(raw_mask.sum())
    if person_pixels < int(h * w * min_pixel_ratio):
        return None

    # contour 외곽 bbox
    ys, xs = np.where(raw_mask > 0)
    if ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    return PersonMask(mask=raw_mask, bbox=(x1, y1, x2, y2))
```

- [ ] **Step 5: 테스트 재실행 — pass 확인**

Run: `.venv/bin/pytest games/08_zombie_dodge/tests/test_body_tracker.py -v`
Expected: 5 passed

- [ ] **Step 6: 회귀 — 08 전체**

Run: `.venv/bin/pytest games/08_zombie_dodge/tests/ -q 2>&1 | tail -5`
Expected: dodge_judge / dodge_state / zombie / zombie_spawner는 그대로 통과. test_dodge_judge는 다음 task에서 마이그레이션될 예정 — 만약 여기서 이미 깨진다면 그 task의 의존성을 task 2가 만들었다는 뜻이니 OK (task 3에서 fix).

---

## Task 3: `dodge_judge.py` 충돌 함수 교체 — mask_circle_collide

**Files:**
- Modify: `games/08_zombie_dodge/src/dodge_judge.py`
- Modify (교체): `games/08_zombie_dodge/tests/test_dodge_judge.py`

- [ ] **Step 1: 신규 테스트 작성 — `test_dodge_judge.py` 전체 교체**

```python
"""dodge_judge — mask × circle 충돌 판정 단위 테스트."""
import numpy as np
import pytest

from games.zombie_dodge.src.body_tracker import PersonMask, compute_person_mask
from games.zombie_dodge.src.dodge_judge import (
    mask_circle_collide,
    evaluate_frame,
    FrameJudgement,
)
from games.zombie_dodge.src.zombie import Zombie   # 기존 zombie.py 사용


def _person(h, w, cx, cy, r):
    yy, xx = np.ogrid[:h, :w]
    raw = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.uint8)
    return compute_person_mask(raw)


def test_circle_inside_mask_collides():
    person = _person(720, 1280, 640, 360, 200)
    # 좀비 원 (40px) 가 마스크 한가운데
    assert mask_circle_collide(person.mask, 640, 360, 40) is True


def test_circle_outside_mask_passes():
    person = _person(720, 1280, 640, 360, 200)
    # 좀비 원이 마스크 밖
    assert mask_circle_collide(person.mask, 100, 100, 40) is False


def test_circle_edge_touch_below_threshold_passes():
    """원이 마스크 경계에 살짝만 걸치면 통과 (overlap < 15%)."""
    person = _person(720, 1280, 640, 360, 200)
    # 마스크 중심에서 r=240 거리 → 원 둘레만 살짝 닿음
    assert mask_circle_collide(
        person.mask, 640 + 235, 360, 40, overlap_ratio=0.15
    ) is False


def test_evaluate_frame_with_none_person_skips_collision():
    zombie = Zombie(x=100, y=200, radius=20, vy=300, kind="basic")
    judgement = evaluate_frame(zombies=[zombie], person=None, frame_h=720)
    assert judgement.collisions == []


def test_evaluate_frame_collision_removes_zombie():
    person = _person(720, 1280, 640, 360, 200)
    zombie = Zombie(x=640, y=360, radius=20, vy=300, kind="basic")
    judgement = evaluate_frame(zombies=[zombie], person=person, frame_h=720)
    assert zombie in judgement.collisions


def test_evaluate_frame_pass_when_below_floor():
    """좀비가 화면 아래로 나가면 passes에 들어감."""
    person = _person(720, 1280, 640, 360, 200)
    zombie = Zombie(x=100, y=750, radius=20, vy=300, kind="basic")
    judgement = evaluate_frame(zombies=[zombie], person=person, frame_h=720)
    assert zombie in judgement.passes
```

- [ ] **Step 2: 테스트 실행 — fail 확인**

Run: `.venv/bin/pytest games/08_zombie_dodge/tests/test_dodge_judge.py -v`
Expected: ImportError (`mask_circle_collide` 미정의) 또는 시그니처 mismatch

- [ ] **Step 3: `dodge_judge.py` 수정 — aabb_circle_collide 제거, mask_circle_collide 추가, evaluate_frame 시그니처 변경**

기존 `aabb_circle_collide()` 함수 전체 삭제. 새로:

```python
def mask_circle_collide(
    mask: np.ndarray,
    cx: int, cy: int, radius: int,
    overlap_ratio: float = 0.15,
) -> bool:
    """좀비 원과 사람 마스크의 픽셀 교집합 ≥ π·r²·overlap_ratio 면 True."""
    h, w = mask.shape[:2]
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius + 1)
    y2 = min(h, cy + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return False

    # 좀비 원의 disk 마스크 (crop 영역과 동일 크기)
    crop_h = y2 - y1
    crop_w = x2 - x1
    yy, xx = np.ogrid[y1:y2, x1:x2]
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius)

    overlap = int(np.count_nonzero(mask[y1:y2, x1:x2] & disk.astype(np.uint8)))
    threshold = int(np.pi * radius * radius * overlap_ratio)
    return overlap >= threshold
```

`evaluate_frame(zombies, box, frame_h)` 시그니처를 `evaluate_frame(zombies, person, frame_h)`로 변경. 내부:
```python
def evaluate_frame(zombies, person, frame_h):
    collisions = []
    passes = []
    survived = []
    for z in zombies:
        if z.y >= frame_h:
            passes.append(z)
            continue
        if person is not None and mask_circle_collide(
            person.mask, int(z.x), int(z.y), int(z.radius)
        ):
            collisions.append(z)
            continue
        survived.append(z)
    return FrameJudgement(collisions=collisions, passes=passes, survived=survived)
```

(`survived`는 기존에도 있었음 — 기존 코드 보존, `box` → `person`만 교체. 기존 `FrameJudgement` dataclass는 그대로 유지.)

- [ ] **Step 4: 테스트 재실행 — pass 확인**

Run: `.venv/bin/pytest games/08_zombie_dodge/tests/test_dodge_judge.py -v`
Expected: 6 passed

- [ ] **Step 5: 08 전체 회귀**

Run: `.venv/bin/pytest games/08_zombie_dodge/tests/ -q 2>&1 | tail -5`
Expected: 통과 (test_body_tracker 5 + test_dodge_judge 6 + state/zombie/spawner 기존)

---

## Task 4: `ui_renderer.py` — draw_person_outline

**Files:**
- Modify: `games/08_zombie_dodge/src/ui_renderer.py`

- [ ] **Step 1: 기존 `draw_dodge_box` 호출처 grep**

Run: `grep -n "draw_dodge_box\|DodgeBox" games/08_zombie_dodge/src/*.py`
모든 호출처/임포트 위치 파악.

- [ ] **Step 2: 신규 함수 추가**

`ui_renderer.py`에 추가:

```python
def draw_person_outline(
    frame_bgr,
    mask,
    color=(255, 255, 255),
    thickness=3,
):
    """마스크 외곽선을 frame 위에 그림.

    Args:
        frame_bgr: H×W×3 BGR 이미지 (in-place 그리기)
        mask: H×W uint8 0/1
        color: BGR 튜플
        thickness: 외곽선 두께 (px)
    """
    import cv2
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return
    cv2.drawContours(frame_bgr, contours, -1, color, thickness)
```

(cv2는 파일 상단에 이미 import되어 있음 — 함수 안 import는 [[feedback-no-lazy-imports]] 위반. 상단 import 확인.)

- [ ] **Step 3: 기존 `draw_dodge_box` 함수 삭제 + 호출처를 `draw_person_outline`으로 교체**

호출처마다 인자 형태 변경:
- 이전: `draw_dodge_box(frame, box, color=...)` → 이후: `draw_person_outline(frame, person.mask, color=...)`
- 호출 전 `if person is not None:` 가드 추가

- [ ] **Step 4: 함수 안 import 점검 (feedback-no-lazy-imports)**

Run: `grep -nE "^\s+(import|from) " games/08_zombie_dodge/src/ui_renderer.py`
함수 내부 import가 없는지 확인. Step 2의 `import cv2`는 함수 안에 적었다면 상단으로 옮길 것.

- [ ] **Step 5: 회귀**

Run: `.venv/bin/pytest games/08_zombie_dodge/tests/ -q 2>&1 | tail -5`
Expected: 기존 그대로 통과 (ui_renderer 직접 테스트는 없음 — game.py 통합에서 검증).

---

## Task 5: `game.py` 통합 + smoke 두 모드

**Files:**
- Modify: `games/08_zombie_dodge/src/game.py`

- [ ] **Step 1: 상단 import 추가 (lazy 금지)**

`game.py` 상단 import 블록에:
```python
from core.vision.person_mask import PersonMaskDetector
from games.zombie_dodge.src.body_tracker import compute_person_mask
```
(import path는 기존 모듈 import 스타일과 일치시킬 것.)

- [ ] **Step 2: `__init__`에 detector 속성 default 정의 ([[feedback-init-must-define-all-attrs]])**

Game 클래스 `__init__`에 추가:
```python
self.person_detector = PersonMaskDetector(model_selection=0)
self.current_person = None   # PersonMask | None — 매 프레임 갱신
```
기존 `pose` detector는 다른 용도 없으면 그대로 두되 (스코프 외) 정리는 별도 작업.

- [ ] **Step 3: 메인 루프 매 프레임 처리 추가**

기존 Pose landmark → compute_dodge_box 호출부를:
```python
raw_mask = self.person_detector.process(frame_bgr)
self.current_person = compute_person_mask(raw_mask)
```
로 교체.

`dodge_judge.evaluate_frame` 호출부:
```python
judgement = evaluate_frame(zombies, person=self.current_person, frame_h=frame_h)
```

렌더:
```python
if self.current_person is not None:
    color = (0, 0, 255) if hit_flash else (255, 255, 255)
    draw_person_outline(out_frame, self.current_person.mask, color=color, thickness=3)
```

- [ ] **Step 4: cleanup 시 detector close**

게임 종료 시 `self.person_detector.close()` 호출 (기존 cleanup 블록에 추가).

- [ ] **Step 5: smoke 두 모드 ([[feedback-no-lazy-imports]])**

Run: `.venv/bin/python -c "import games.08_zombie_dodge.src.game"`
Expected: import 성공 (실행은 X). 만약 import 자체에서 카메라 열려고 하면 module-level 호출이 잘못된 것 — fix.

Run: `.venv/bin/python -m games.08_zombie_dodge.src.game --help` 또는 잠깐 띄웠다가 Q로 종료. (카메라 + GUI 필요 — manual)

직접 실행 모드:
Run: `.venv/bin/python games/08_zombie_dodge/src/game.py --help` 또는 manual.

둘 다 module 초기화 + sound + detector 로딩 통과 확인.

- [ ] **Step 6: 전체 회귀**

Run: `.venv/bin/pytest -q 2>&1 | tail -5`
Expected: 통과. 신규 person_mask 3 + body_tracker 5 + dodge_judge 6 + 나머지 = 누적 ~690+.

---

## Task 6: 성능 sanity & 잔여 정리

**Files:**
- (Optional) Create: `tests/perf/test_person_mask_perf.py`

- [ ] **Step 1: 1프레임 처리 시간 측정 (선택)**

```python
"""성능 sanity — 카메라 없이 random frame으로 1프레임 시간 측정."""
import time
import numpy as np
import pytest

from core.vision.person_mask import PersonMaskDetector


@pytest.mark.slow
def test_one_frame_under_15ms():
    with PersonMaskDetector(model_selection=0, downsample_height=180) as det:
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        det.process(frame)   # warmup
        t0 = time.perf_counter()
        for _ in range(30):
            det.process(frame)
        avg_ms = (time.perf_counter() - t0) * 1000 / 30
        print(f"\navg = {avg_ms:.2f}ms/frame")
        assert avg_ms < 15.0
```

Run: `.venv/bin/pytest tests/perf/test_person_mask_perf.py -v -s`
Expected: 1프레임 < 15ms (W7 30fps 기준 충분).

- [ ] **Step 2: 사용 안 하는 import 정리**

기존 `body_tracker.py`에 있던 NOSE/LEFT_SHOULDER/RIGHT_SHOULDER 등 상수가 다른 곳에서 import되는지 확인:
Run: `grep -rn "NOSE\|LEFT_SHOULDER\|DodgeBox\|compute_dodge_box\|aabb_circle_collide" games/08_zombie_dodge/`
모두 없으면 OK. 있으면 호출처 fix.

- [ ] **Step 3: 최종 회귀**

Run: `.venv/bin/pytest -q --ignore=tests/perf 2>&1 | tail -5`
Expected: 전체 통과.

- [ ] **Step 4: 메모리에 변경 사항 기록**

`memory/project_playwait_polish_plan.md`에 W7 hitbox v2 폴리싱 완료 노트 추가:
```
- W7 hitbox v2 (2026-05-19): SelfieSegmentation 마스크 기반 충돌 + 외곽선 렌더로 사각형 박스 교체. core/vision/person_mask.py 신규 (재활용 가능).
```

- [ ] **Step 5: 체크포인트 (git 없음)**

변경/생성 파일 목록 확인:
```bash
ls -la core/vision/person_mask.py \
  games/08_zombie_dodge/src/{body_tracker,dodge_judge,ui_renderer,game}.py \
  tests/core/vision/test_person_mask.py
```
사용자에게 실기 플레이 요청 — 카메라 + GUI 필수.

---

## 자체 점검 결과

**Spec coverage:** spec §2.1~2.5 모두 task 1~5에 매핑. §3 Fallback은 task 2 (person=None 케이스) / task 5 (가드)에서 처리. §4 data flow는 task 5 통합. §5 테스트는 task 1·2·3 (5.1·5.2)과 task 6 (5.3 성능). §6 prefix·§7 smoke·§8 비스코프 모두 task에 반영.

**Placeholder scan:** TBD/TODO 없음. 코드 블록 모두 실제 코드. import path만 conftest 패턴에 맞춰 조정하라는 "exact 패턴 grep" step을 task 2 step 2에 명시.

**Type 일관성:** `PersonMask` (Task 2), `mask: np.ndarray, bbox: tuple`, `compute_person_mask(raw_mask) → PersonMask | None` 시그니처가 Task 3·5에서 동일하게 사용됨. `evaluate_frame(zombies, person, frame_h)` 시그니처 Task 3·5 일치. `mask_circle_collide(mask, cx, cy, radius, overlap_ratio=0.15)` Task 3 통일.
