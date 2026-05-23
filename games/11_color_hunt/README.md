# 컬러 헌트 (Color Hunt) — W5

YOLO + HSV 색 분류로 매장 사물에서 미션 색 N개를 시간 안에 찾는 게임.
**세련된 8색 팔레트** (버건디 · 테라코타 · 머스타드 · 세이지 · 네이비 · 모브 · 크림 · 차콜).

상세 설계: [W5_컬러헌트_기획서.md](./W5_컬러헌트_기획서.md)

## 빠른 시작

```bash
cd ~/PlayWait

# 단위 테스트 (102 cases, ~0.25s — 카메라/YOLO 무관)
.venv/bin/pytest games/11_color_hunt/tests/ -q

# 라이브 데모 (게임 UI 없이 YOLO + 색 분류만)
.venv/bin/python games/11_color_hunt/run_demo.py

# 정식 게임 (페이즈 + 미션 + 스코어 HUD)
.venv/bin/python -m games.11_color_hunt.src.game
```

## 조작 (정식 게임)

| 키 | 동작 |
|---|---|
| `1` / `2` / `3` | 난이도 (쉬움 90s·2개 / 보통 60s·3개 / 어려움 45s·4개) |
| `SPACE` | 시작 |
| `R` | 재시작 (게임 종료 후) |
| `Q` / `ESC` | 종료 |
| `F` / `+` / `-` / `0` | 전체화면 / 확대 / 축소 / 1.5× 리셋 |

## 페이즈

```
DIFFICULTY_SELECT → READY (미션 카드)
        ↓                ↓ SPACE
        └──────── PLAYING (YOLO + 분류 + 카운트)
                         ↓
              WIN(collected≥goal) | TIMEOUT(remaining≤0)
                         ↓
                     GAME_OVER (둥근 카드 + 우승 메달)
```

## 모듈

| 파일 | 역할 | LOC |
|------|------|---:|
| `src/color_classifier.py` | HSV → 8색 분류 + 매칭 점수 (profile + min 집계, 순수함수) | 207 |
| `src/mission_generator.py` | 난이도 → Mission(color/goal/time/gate). 직전 회피 + RNG 격리 | 163 |
| `src/hunt_tracker.py` | 라운드 상태 — 수집/dedup(5s window)/종료 사유 (순수, time_provider 주입) | 233 |
| `src/object_pipeline.py` | YOLO duck-typed detector + ROI 중앙 60% + median HSV + 분류 통합 | 156 |
| `src/game.py` | 메인 게임 (4페이즈 + YOLO + 친근 UI + 안전 종료) | 530 |
| `src/theme.py` | PinkLAB + W5 8색 토큰 (한글/BGR/라벨) | (W3+α) |
| `src/ui_renderer.py` | 한글 PIL 배치 + **둥근 도형 카드** + 큰 색 원·진행 점·메달 | (W3+α) |
| `src/sound_manager.py` | pygame.mixer 7종 (W3 그대로) | 158 |
| `run_demo.py` | 게임 페이즈 없이 YOLO + 분류 실시간 데모 | 156 |
| `core/vision/yolo_engine.py` | ultralytics 래퍼 (lazy load) | 111 |

## 친근한 UI (W5 디자인 특징)

- **둥근 카드 패널** (모서리 22~24px) + 부드러운 그림자
- **큰 둥근 색 원** + 좌상단 흰 광택 — 미션 색을 친근하게 시각화
- **진행 점**: collected는 채워진 윈 색 원, remaining은 빈 회색 원 — 직관적
- **둥근 객체 박스** (cv2.rectangle 대신 직선 + ellipse arc 합성)
- **우승 메달**: 큰 둥근 메달 + 광택, 우승 카드 외곽이 핑크
- 모든 텍스트는 NanumGothic (이모지 X 박스 회피)

## 난이도 / 미션

| 난이도 | 시간 | 목표 | 신뢰도 게이트 | 풀 특성 |
|---|---|---|---|---|
| 쉬움 | 90s | 2개 | 60% | 흔한 색 (크림·차콜·네이비·머스타드) |
| 보통 | 60s | 3개 | 65% | 8색 균등 |
| 어려움 | 45s | 4개 | 75% | 톤 비슷한 그룹 (버건디·모브 / 테라코타·머스타드 / 세이지·네이비) |

## 중복 방지 (dedup)

- 같은 yolo_class + 박스 중심 80px 이내 + 5초 이내 = 같은 객체로 보고 카운트 1번만
- 다른 위치 또는 다른 클래스(예: cup → bottle)는 새 객체로 인정
- → 같은 컵을 여러 각도에서 비춰도 부정 카운트 방지

## 단위 테스트 (총 102 cases, ~0.25s)

| 파일 | 케이스 | 검증 |
|---|---:|---|
| `tests/test_color_classifier.py` | 32 | 8색 분류 정확성, NEUTRAL 폴백, fall-off 거동 |
| `tests/test_object_pipeline.py` | 28 | ROI 중앙 추출, median HSV, FakeDetector 통합 |
| `tests/test_mission_generator.py` | 19 | 난이도 풀·단조성, 직전 회피, 시드 격리 |
| `tests/test_hunt_tracker.py` | 23 | 수집·dedup window·종료 우선순위·time_provider |

```bash
.venv/bin/pytest games/11_color_hunt/tests/ -q
# ........... 102 passed in 0.25s
```

## 환경 (W1~W4 그대로)

- Python 3.12, MediaPipe 0.10.14, NumPy 1.26.4, OpenCV 4.8.1
- ultralytics (YOLO) — `pip install ultralytics` 필요. 첫 실행 시 `yolov8n.pt` (~6.5MB) 자동 다운로드 (~3초)
- 카메라 640×480, DISPLAY_SCALE=1.5
- 한글 폰트: NanumGothic (자동 탐색)
