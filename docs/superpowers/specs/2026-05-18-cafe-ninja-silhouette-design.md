# 카페 닌자 — 그림자 실루엣 + 눈 복면 디자인 (2026-05-18)

폴리싱 1차 후속. 기존 배경 어둠 처리(`BACKGROUND_ALPHA=0.40` / `BACKGROUND_BRIGHTNESS=0.25`)에서 한 단계 더 나아가, **배경은 원색을 유지하고 사람 영역만 닌자 그림자 실루엣**으로 표현. 손은 게임 플레이를 위해 카메라 노출, 양 눈은 작은 원형으로 노출하여 **복면 닌자**의 시각적 정체성을 부여한다.

## 1. 목적

- **프라이버시**: 매장 환경에서 얼굴/체형 노출 최소화 (정체성: 눈만 빛나는 그림자)
- **컨셉 일치**: "카페 닌자" 정체성 강화 — 사용자 자신이 닌자가 된 듯한 몰입
- **배경 살리기**: 폴리싱 1차에서 도입한 중국 거리 배경(town.jpg)이 어둠 처리로 묻혔던 문제 해소

## 2. 비주얼 구성

```
┌──────────────────────────────────────────────────────┐
│ town.jpg 원색 배경 (사람 영역 외 100% 노출)          │
│                                                       │
│                   ◐ ◐                                 │
│              ░░░░░░░░░░░░    ← 사람: 자주톤(40,25,55)│
│             ░░░░░░░░░░░░░░     평면 그림자           │
│           ░░░░░░░░░░░░░░░░░                          │
│         ░░░░░░░░░░░░░░░░░░░░                         │
│        ░░░░░░  [손]   ░░░░░░  ← 손: 카메라 노출      │
│       ░░░░░░░░░░░░░░░░░░░░░░     (게임 플레이용)     │
│      ░░░░░░░░░░░░░░░░░░░░░░░░                        │
│                                                       │
│  [닌자 마스코트]                  [점수 패널]         │
└──────────────────────────────────────────────────────┘
```

- **배경 영역**: town.jpg 100% (블렌딩 X, 어둠 처리 X)
- **사람 영역**: 자주톤(BGR 40, 25, 55) 평면 채움 — 닌자 의상 톤
- **손 영역**: 카메라 원본 픽셀 노출 (페더 30px Gaussian)
- **양 눈 영역**: 작은 원(반지름 12px, 페더 6px)으로 카메라 원본 노출

## 3. 아키텍처

### 3.1 모듈 구조

| 모듈 | 책임 | 신규/변경 |
|---|---|---|
| `src/ninja_silhouette.py` | MediaPipe Selfie Segmentation + Face Detection 래퍼 | **신규** |
| `src/ui_renderer.py` | `apply_silhouette()` 추가, 기존 `draw_background`·`apply_hand_spotlight` 호출은 silhouette 활성 시 우회 | 변경 |
| `src/theme.py` | 실루엣 파라미터 5종 추가 | 변경 |
| `src/game.py` | run 루프에서 person_mask·eye_points 계산 → render 전달 | 변경 |
| `src/finger_tracker.py` | `get_hand_bbox()` 그대로 재사용 | 변경 없음 |

**모듈명 prefix**: 메모리 규칙(`feedback-module-name-prefix`)에 따라 `silhouette.py`가 아닌 **`ninja_silhouette.py`** 사용 — 다른 게임에서 동명 모듈을 만들어도 통합 회귀 충돌 회피.

### 3.2 ninja_silhouette.py 인터페이스

```python
class NinjaSilhouette:
    def __init__(self, model_selection: int = 1, min_face_confidence: float = 0.5):
        """
        model_selection: 0=일반(가벼움), 1=풍경/전신(권장)
        Selfie Segmentation + Face Detection 두 모델 초기화.
        실패 시 self.available = False 폴백.
        """

    def process(self, frame_bgr: np.ndarray) -> SilhouetteResult:
        """프레임 1회 추론. 합성에 필요한 마스크/좌표 반환."""

    def close(self) -> None:
        """MediaPipe 인스턴스 해제."""


@dataclass
class SilhouetteResult:
    person_mask: np.ndarray | None  # uint8, shape (H, W), 0~255 (사람=255)
    eye_points: list[tuple[int, int]]  # 최대 2개 (오른눈/왼눈 픽셀 좌표)
    detected: bool  # 사람·얼굴 둘 다 검출됐는가
```

### 3.3 ui_renderer.apply_silhouette() 인터페이스

```python
def apply_silhouette(
    frame: np.ndarray,           # 카메라 원본 (BGR, in-place 수정됨)
    person_mask: np.ndarray | None,
    eye_points: list[tuple[int, int]],
    hand_bbox: tuple[int, int, int, int] | None,
    bg_image: np.ndarray,        # town.jpg, frame과 동일 크기
) -> None:
    """배경 원색 + 사람 자주톤 + 손/눈 카메라 reveal 합성. frame in-place.

    person_mask=None → 사람 미검출: 배경 100% 적용, 카메라 미노출.
    eye_points=[] → 얼굴 미검출: 눈 reveal 생략.
    hand_bbox=None → 손 미검출: 손 reveal 생략.
    """
```

## 4. 합성 알고리즘 (apply_silhouette)

```
camera_pixels = frame.copy()  # 원본 보존 (reveal용)

# 1) 배경 영역에 town.jpg 채우기 (사람 영역에서는 카메라 원본 유지)
if person_mask is None:
    frame[:] = bg_image
    return
mask_f = (person_mask / 255).astype(np.float32)[..., None]  # (H,W,1)
# 사람 영역=카메라 픽셀, 배경 영역=town.jpg
frame[:] = (camera_pixels * mask_f + bg_image * (1 - mask_f)).astype(np.uint8)

# 2) 사람 영역을 자주톤 평면으로 덮기
person_alpha = mask_f * SILHOUETTE_ALPHA
tint = np.full_like(frame, SILHOUETTE_TINT_BGR, dtype=np.uint8)
frame[:] = (tint * person_alpha + frame * (1 - person_alpha)).astype(np.uint8)
# SILHOUETTE_ALPHA=1.0이면 사람 영역 완전 평면 그림자

# 3) 손 영역 reveal (camera_pixels 복원, person_mask 외부는 클립)
if hand_bbox:
    hand_mask = build_circular_mask(frame.shape[:2], hand_bbox, feather=HAND_REVEAL_FEATHER)
    # hand_mask: float32 (H,W) 0~1, 손 중앙=1, 외곽=Gaussian 감쇠
    hand_mask = hand_mask * mask_f[..., 0]   # 배경 영역엔 노출 X
    hm = hand_mask[..., None]
    frame[:] = (camera_pixels * hm + frame * (1 - hm)).astype(np.uint8)

# 4) 눈 영역 reveal (양 눈 각각 작은 원)
for (ex, ey) in eye_points:
    eye_mask = build_disk_mask(frame.shape[:2], (ex, ey), EYE_REVEAL_RADIUS, EYE_REVEAL_FEATHER)
    eye_mask = eye_mask * mask_f[..., 0]
    em = eye_mask[..., None]
    frame[:] = (camera_pixels * em + frame * (1 - em)).astype(np.uint8)
```

**연산 비용 추정** (640×480 기준):
- Selfie Seg 추론: 약 5~8ms (CPU, model_selection=1)
- Face Detection 추론: 약 2~4ms
- Mask 합성 4단계: 약 1~2ms
- **총 추가 비용: ~10~14ms/frame** → 30 FPS 목표 시 여유 ~19ms 확보

## 5. theme.py 신규 파라미터

```python
# 9. W3 폴리싱 - 실루엣 (2026-05-18 silhouette design)
SILHOUETTE_TINT_BGR = (40, 25, 55)   # 닌자 의상 자주톤 (기존 SPOTLIGHT_TINT와 동일)
SILHOUETTE_ALPHA = 1.0               # 1.0=평면 그림자, 0.85=카메라 약간 비침
EYE_REVEAL_RADIUS = 12               # px (얼굴 식별 불가 수준)
EYE_REVEAL_FEATHER = 6               # Gaussian sigma, 부드러운 경계
HAND_REVEAL_FEATHER = 30             # px, 손 외곽 페더
PERSON_MASK_THRESHOLD = 0.5          # Selfie Seg 확률 → 이진화 임계
```

기존 `BACKGROUND_ALPHA` / `BACKGROUND_BRIGHTNESS`는 **삭제 또는 비활성** — 신규 합성에서 사용 X. 다른 게임에서 참조하지 않음(확인 완료).

## 6. game.py 변경

### 6.1 init

```python
from .ninja_silhouette import NinjaSilhouette  # dual-import 블록에 추가
# ...
self.silhouette = NinjaSilhouette()  # 실패 시 self.silhouette.available = False
```

### 6.2 run 루프 (매 프레임)

```python
sil_result = self.silhouette.process(frame) if self.silhouette.available else None
person_mask = sil_result.person_mask if sil_result else None
eye_points = sil_result.eye_points if sil_result else []
# 손 bbox는 기존 finger_tracker 결과 활용
self.render(frame, tip_px, hand_bbox, person_mask, eye_points)
```

### 6.3 render

```python
if self.phase == "PLAYING" and self.silhouette.available:
    ui.apply_silhouette(frame, person_mask, eye_points, hand_bbox, self.background)
else:
    ui.draw_background(frame, self.background)  # 폴백
    ui.apply_hand_spotlight(frame, hand_bbox)
```

폴백: `silhouette.available=False` (모델 init 실패) 또는 PLAYING 외 페이즈 → 기존 spotlight 방식 유지.

## 7. 에러 핸들링

| 상황 | 동작 |
|---|---|
| `mediapipe.solutions.selfie_segmentation` 없음 (이전 mediapipe 버전) | `NinjaSilhouette.available = False` → 기존 spotlight 폴백 |
| Face Detection init 실패 | Selfie Seg만 사용, `eye_points = []` (사람만 그림자, 눈 reveal X) |
| 사람 미검출 (mask 빈 영역) | 배경 100% 노출, 카메라 미노출 ("사람 등장 대기" 자연 연출) |
| 얼굴 측면/뒤 (눈 keypoint 신뢰도 낮음) | `eye_points = []` (그림자만, 자연스러움) |
| mask shape mismatch | `cv2.resize`로 frame 크기 강제 일치 |
| 손/눈 좌표 화면 밖 | `build_disk_mask` 내부에서 클립 (현재 spotlight 로직과 동일) |

## 8. 테스트 전략

### 8.1 단위 테스트 (`tests/test_ninja_silhouette.py`)

| 케이스 | 검증 |
|---|---|
| `process()` 합성 frame | mask shape=(H, W), dtype=uint8, 값 범위 0~255 |
| `eye_points` 좌표 범위 | 모두 (0~W, 0~H) 안 |
| mediapipe init 실패 monkeypatch | `available=False` + `process()` 호출 시 None 반환 |
| 사람 미검출 (mask 전부 0) | `detected=False`, `eye_points=[]` |
| 얼굴 미검출 단독 | `person_mask` 정상, `eye_points=[]` |

**MediaPipe 인스턴스는 monkeypatch로 mock** — 네트워크/모델 로드 X. W9 `client_factory` 패턴 차용.

### 8.2 단위 테스트 (`tests/test_ui_renderer.py` 추가)

| 케이스 | 검증 |
|---|---|
| `apply_silhouette` 배경 영역 픽셀 | bg_image 값과 동일 |
| 사람 영역 픽셀 (손/눈 외) | `SILHOUETTE_TINT_BGR`과 동일 (ALPHA=1.0 가정) |
| 손 영역 중심 픽셀 | 카메라 원본 값과 동일 |
| 눈 영역 중심 픽셀 | 카메라 원본 값과 동일 |
| 눈 영역이 배경에 걸친 경우 | reveal X (person_mask 클립) |
| `person_mask=None` | frame 전체가 bg_image |

### 8.3 두 모드 smoke (메모리 규칙 `feedback-no-lazy-imports`)

```bash
.venv/bin/python -m games.01_cafe_ninja.src.game           # 패키지 모드
.venv/bin/python games/01_cafe_ninja/src/game.py            # 직접 실행
```

둘 다 카메라 초기화 + PLAYING 페이즈 진입까지 크래시 없음 확인.

### 8.4 통합 회귀

```bash
.venv/bin/pytest games/01_cafe_ninja/tests/ -q
.venv/bin/pytest games/ -q   # 678+ 통합 회귀 깨지지 않는지
```

## 9. 작업 산출물

### 신규
- `games/01_cafe_ninja/src/ninja_silhouette.py` (~150줄)
- `games/01_cafe_ninja/tests/test_ninja_silhouette.py` (~120줄, 12~15 cases)
- 본 spec 문서

### 변경
- `games/01_cafe_ninja/src/theme.py` (파라미터 5+1종 추가, BACKGROUND_* 정리)
- `games/01_cafe_ninja/src/ui_renderer.py` (`apply_silhouette()` 추가, 폴백 분기 유지)
- `games/01_cafe_ninja/src/game.py` (silhouette init + run 루프 + render 분기, dual-import 블록 갱신)
- `games/01_cafe_ninja/tests/test_ui_renderer.py` (apply_silhouette 6+ 케이스)

## 10. 검증 완료 기준

1. `pytest games/01_cafe_ninja/tests/ -q` 신규 케이스 포함 100% 통과
2. 두 모드 smoke 통과 (`-m` + 직접 실행)
3. 카메라 실기:
   - 사람 등장 전: 배경만 보임 (그림자 X)
   - 사람 정면: 자주톤 실루엣 + 양 눈 작은 원 + 손 영역 카메라 노출
   - 사람 측면: 자주톤 실루엣 + 눈 reveal 없음 (자연스러움)
   - 30 FPS 유지 (저하 5 FPS 미만)
4. 통합 회귀(`pytest games/ -q`) 깨지지 않음

## 11. 비범위 (이번 작업 제외)

- 다른 게임(02~11)에 silhouette 적용 — **베스트 3 선정 후 별도 진행**
- Selfie Seg 모델을 yolov8-seg 등으로 교체 — 성능 부족 확인 시 추후
- 눈 모양 stylize (반짝임 애니메이션, ★ 모양 등) — 폴리싱 2차로 보류
- 그림자 외곽선 stroke 추가 — 기본 평면이 닌자 그림자 의도 충족 → 보류

## 12. 메모리 규칙 체크리스트

- [x] **`feedback-no-lazy-imports`**: 모든 import는 `game.py` 상단 dual-import 블록 1곳에 집중
- [x] **`feedback-module-name-prefix`**: `ninja_silhouette.py` (게임 prefix)
- [x] **`feedback-init-must-define-all-attrs`**: `self.silhouette` `__init__`에서 1회 정의, `process` 결과 속성도 default 초기화
- [x] **`feedback-ui-renderer-anchor-fix`**: 이번 작업은 anchor 영역 손대지 않음 — 기존 일반화 패턴 유지
- [x] **두 모드 smoke**: 검증 완료 기준에 포함
