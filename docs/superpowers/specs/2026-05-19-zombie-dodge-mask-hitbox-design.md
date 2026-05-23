# Zombie Dodge — 실루엣 Hitbox 개선 설계

- **게임**: W7 `games/08_zombie_dodge`
- **목표**: 사각형 AABB hitbox → MediaPipe SelfieSegmentation 픽셀 마스크 기반 판정
- **스코프**: hitbox 로직 + 화면 시각화. 좀비 그래픽·사운드·배경은 그대로.
- **작성일**: 2026-05-19

---

## 1. 배경

현재 `body_tracker.compute_dodge_box()`는 Pose landmark 3개(코·좌어깨·우어깨)로 어깨~허리 AABB 사각형을 계산하고, `dodge_judge.aabb_circle_collide()`로 좀비 원과 충돌 판정한다.

**문제점**
- 머리 위·팔·다리는 박스 밖 → 좀비가 머리에 정확히 떨어져도 통과 처리
- 박스가 사람 윤곽과 동떨어져 보임 (시각적 어색함)
- 옆으로 팔을 뻗는 회피 동작이 hitbox에 반영 안 됨

---

## 2. 변경 사항

### 2.1 신규 모듈 `core/vision/person_mask.py`

다른 게임에서도 재활용 가능한 코어 모듈로 분리.

```python
class PersonMaskDetector:
    def __init__(self, model_selection: int = 0, downsample_height: int = 180): ...
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame_bgr(H×W×3) → mask(H×W, uint8). 1=사람, 0=배경.
        내부 처리는 다운샘플 해상도, 결과는 원본 크기로 nearest upscale.
        """
```

- MediaPipe `selfie_segmentation.SelfieSegmentation(model_selection=0)` 일반 모델 사용
- 다운샘플(240×180 수준) → CPU 절약 (~3–5ms/frame 목표)
- threshold 0.5로 raw soft mask → binary mask 변환
- `with PersonMaskDetector() as det:` 컨텍스트 매니저 지원

### 2.2 `08/src/body_tracker.py` 교체

기존 `DodgeBox` / `compute_dodge_box()` 제거하고:

```python
@dataclass(frozen=True)
class PersonMask:
    mask: np.ndarray   # H×W uint8
    bbox: tuple        # (x1, y1, x2, y2) — 마스크 contour 외곽
    @property
    def center(self) -> tuple: ...

def compute_person_mask(detector, frame_bgr) -> PersonMask | None:
    """마스크 추출. 픽셀 합 < min_pixels면 None."""
```

- `bbox`는 렌더링 영역 limit용으로 contour 외곽 사각형 캐싱
- 사람 픽셀 수 임계값 미만이면 None (사람 미감지). 임계값 디폴트: 원본 픽셀 기준 `frame_w × frame_h × 0.02` (전체 화면의 2% 이상이면 valid).

### 2.3 `08/src/dodge_judge.py` 충돌 함수 교체

```python
def mask_circle_collide(
    mask: np.ndarray, cx: int, cy: int, radius: int,
    overlap_ratio: float = 0.15,
) -> bool:
    """좀비 원 영역과 마스크의 겹친 픽셀 수가
    π·r²·overlap_ratio 이상이면 True.
    """
```

- 좀비 원의 bounding box(`cy-r:cy+r`, `cx-r:cx+r`) crop → 미리 만든 원형 disk(`np.ones (2r+1, 2r+1)`을 원형 0/1 마스크) AND → `np.count_nonzero` 합산
- threshold: 원 면적(πr²) × 0.15 (튜닝 가능)
- `evaluate_frame()`에서 `aabb_circle_collide` 호출부를 `mask_circle_collide`로 교체
- `box: DodgeBox | None` 인자를 `person: PersonMask | None`으로 변경 — `person is None`이면 충돌 판정 skip (좀비는 통과)

### 2.4 `08/src/ui_renderer.py` 시각화 교체

`draw_dodge_box()` 제거, 신규 함수 추가:

```python
def draw_person_outline(
    frame_bgr: np.ndarray, mask: np.ndarray,
    color: tuple = (255, 255, 255), thickness: int = 3,
): ...
```

- `cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` → `cv2.drawContours(...)` 굵기 3px
- 색상: 기본 흰색, hit 직후 N프레임 빨강 깜빡임 (기존 박스 깜빡임 로직 재활용)

### 2.5 `08/src/game.py` 통합

- Pose detector + `PersonMaskDetector` 둘 다 초기화 (Pose 자체는 다른 용도 없으면 제거 후보지만 이번 작업에서는 손대지 않음 — 추후 정리)
- 매 frame: BGR → `mask = detector.process(frame_bgr)` → `person = compute_person_mask(...)`
- `dodge_judge.evaluate_frame(zombies, person)` 호출
- 렌더 시 `ui_renderer.draw_person_outline(out_frame, person.mask, ...)` 호출

---

## 3. Fallback / 엣지

| 상황 | 처리 |
|---|---|
| 사람 미감지 (마스크 픽셀 < 임계값) | 외곽선 미렌더 + 충돌 판정 skip — 좀비는 통과 |
| MediaPipe 초기화 실패 | 모듈 import 시점 예외 전파. `game.py`에서 try 없이 그대로 sys.exit 또는 stderr 출력 후 종료 (다른 게임의 Pose/Hands 초기화와 동일 톤) |
| 좀비 원 일부가 화면 밖 | `np.clip`으로 crop 범위 보정 후 계산 |

---

## 4. Data flow

```
frame_bgr ─→ PersonMaskDetector.process() ─→ mask
                                              ├─→ draw_person_outline() (시각화)
                                              ├─→ compute_person_mask() → PersonMask
                                              └─→ mask_circle_collide(zombie) ─→ FrameJudgement
```

---

## 5. 테스트

### 5.1 신규
- `tests/test_person_mask.py`
  - PersonMaskDetector mock (process가 고정 mask 반환) 컨텍스트 매니저 동작
  - 다운샘플 → upscale 결과 크기 = 원본 크기
- `tests/test_dodge_judge_mask.py`
  - 합성 mask (numpy로 사람 모양 disk 만들기) + 좀비 원 위치 케이스
    - 좀비가 마스크 한가운데 → hit
    - 좀비가 마스크 밖 → 통과(pass)
    - 좀비가 마스크 가장자리 살짝 겹침(겹친 픽셀 < threshold) → 통과
    - `person is None` → 모든 좀비 통과
  - `evaluate_frame()` 통합 — `passes`/`collisions`/`survived` 분기

### 5.2 기존 호환
- 기존 `tests/test_body_tracker.py`의 AABB 케이스는 마이그레이션:
  - DodgeBox 빌더 → 동일 영역의 사각형 mask로 변환해서 동일 의미의 테스트 유지
- 통합 회귀: 카메라 없이 mock detector + mock mask로 game 1-frame step

### 5.3 성능 sanity
- 1280×720 입력, 다운샘플 240×180, 좀비 10마리 기준 한 프레임 처리 < 10ms 확인 (개발 환경)

---

## 6. 모듈 prefix / 명명

[[feedback-module-name-prefix]]에 따라 게임 prefix 유지:
- `body_tracker.py` 그대로 (기존명 — 게임 prefix 없지만 코어와 분리되어 있음)
- 신규 코어 모듈은 `core/vision/person_mask.py` — 게임 외부라 일반명 OK
- 신규 테스트: `test_person_mask.py`, `test_dodge_judge_mask.py`

---

## 7. Smoke 모드

[[feedback-no-lazy-imports]]에 따라 두 모드 확인:
- `.venv/bin/python -m games.08_zombie_dodge.src.game`
- `.venv/bin/python games/08_zombie_dodge/src/game.py`

둘 다 import 단계 성공 + 게임 부팅 직전까지 진입 확인 (카메라 자동 종료 옵션 없으므로 manual).

---

## 8. 비스코프

다음 작업이 아님 — 추후 별도 작업:
- 좀비 그래픽 폴리싱 (피·머리카락 디테일)
- 배경 변경
- 사운드 추가/교체
- Pose detector 제거 (마스크가 Pose 대체할 수 있는지 확인 후 별도 작업)
