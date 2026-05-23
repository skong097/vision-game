# AR 보물찾기 (Hidden Menu Hunt) — W10 (마지막 게임)

매장의 평범한 사물(컵·책·휴대폰...)이 Doby의 비밀 보물! YOLO + AR overlay로 카메라를 비추면 객체를 검출하고, **화면 중앙 ROI에 1초 유지**하면 발견. 60초 안에 목표 개수 발견 시 승리.

상세 설계: [W10_AR보물찾기_기획서.md](./W10_AR보물찾기_기획서.md)

## 빠른 시작

```bash
cd ~/PlayWait
# 단위 테스트 (69 cases, ~0.10s)
.venv/bin/pytest games/03_hidden_menu_hunt/tests/ -q

# 게임 실행 (카메라 + ultralytics 필요)
.venv/bin/python -m games.03_hidden_menu_hunt.src.game
```

## 조작

| 키 | 동작 |
|---|---|
| `1` / `2` / `3` | 난이도 (쉬움 3개·0.7s / 보통 5개·1.0s / 어려움 7개·1.5s) |
| `SPACE` | 게임 시작 |
| `R` | 재시작 |
| `Q` / `ESC` | 종료 |
| `F` / `+` / `-` / `0` | 전체 화면 / 확대 / 축소 / 리셋 |

## 페이즈 (W7 패턴 — 실시간 시뮬레이션)

```
DIFFICULTY_SELECT → READY ─(SPACE)─→ PLAYING (60초)
                                          ↓
                            WIN(목표 달성) | TIMEOUT
                                          ↓
                                    GAME_OVER
```

## 게임 메카닉

1. Doby가 미션 보물 출제 (12개 풀, 직전 회피)
2. 화면 중앙 **둥근 ROI** 표시 (30% 영역, 핑크)
3. 사용자가 카메라로 매장을 비추면 YOLO가 객체 검출 + AR 박스
4. 미션 객체가 ROI 안에 들어옴 → 진행 게이지 차오름
5. **유지 시간(난이도별 0.7~1.5s) 도달** → 발견! +20점 + 다음 보물 자동 출제
6. 60초 안에 목표 개수 발견 → WIN + **잔여 시간 보너스** (남은 초 × 1점)

## 보물 풀 (COCO 12종)

| 클래스 | 한글 | AR 힌트 | 풀 |
|---|---|---|---|
| cup | 컵 | 카페의 단짝! | E·N·H |
| bottle | 병 | 음료 한 잔 | E·N·H |
| book | 책 | 조용한 시간 | E·N·H |
| cell phone | 휴대폰 | SNS 공유 좋아요 | E·N·H |
| chair | 의자 | 편안한 자리 | E·N·H |
| laptop | 노트북 | 카공족 친구 | N·H |
| keyboard | 키보드 | 타이핑 박자 | N·H |
| vase | 화병 | 인테리어 포인트 | N·H |
| scissors | 가위 | 작은 도구 | H |
| mouse | 마우스 | 노트북 친구 | H |
| fork | 포크 | 디저트 시간 | H |
| spoon | 스푼 | 커피와 함께 | H |

## 모듈

| 파일 | 역할 | LOC |
|---|---|---:|
| `src/treasure_clues.py` | 12개 보물 풀 + 한글/힌트 + 출제기 (순수) | 106 |
| `src/treasure_state.py` | 60s 진행·ROI 유지 시간·발견 카운트·종료 (순수, time_provider) | 242 |
| `src/ar_overlay.py` | 중앙 ROI 계산 + 매칭 검출 (순수 함수) | 115 |
| `src/game.py` | 메인 게임 (4페이즈 + YoloEngine + AR + 안전 종료, lazy import 0건) | 552 |
| `src/theme.py` | W9 + W10 토큰 | (W9+α) |
| `src/ui_renderer.py` | W9 친근 도형 + **중앙 ROI 타겟 + AR 박스 + 발견 플래시** | (W9+α) |
| `src/sound_manager.py` | W9 그대로 | 158 |

W5의 `core/vision/yolo_engine.py` **재활용** — 추가 의존성 X.

## AR 컴포넌트 (W10 신규)

- **중앙 타겟**: 화면 30% ROI 외곽 둥근 사각형 + 4 모서리 점 + 중앙 십자선 + 진행 게이지
- **AR 박스**: 검출된 모든 객체에 둥근 박스 + 라벨. 미션 객체는 핑크 강조 (다른 객체는 회색)
- **발견 플래시**: 화면 중앙 큰 팝업 ("발견! [한글] +20점") 1초 지속
- **미션 카드**: 좌하단 작은 둥근 카드 — 현재 미션 + AR 힌트

## 단위 테스트 (총 69 cases, ~0.10s)

| 파일 | 케이스 | 검증 |
|---|---:|---|
| `tests/test_treasure_clues.py` | 22 | 12개 풀, pool 단조성, 직전 회피, 시드 격리 |
| `tests/test_ar_overlay.py` | 21 | CenterROI API, ROI 계산, find_target_in_roi (가장 높은 신뢰도, ROI 외부 무시) |
| `tests/test_treasure_state.py` | 26 | hold 누적/리셋, 발견 카운트, 시간 보너스, 종료 우선순위, time_provider |

```bash
.venv/bin/pytest games/03_hidden_menu_hunt/tests/ -q
# 69 passed in 0.10s
```

## 환경

- Python 3.12, OpenCV 4.8.1, NumPy 1.26.4
- **`ultralytics` 필요**: `pip install ultralytics` (YOLO 모델 자동 다운로드 ~6.5MB)
- 카메라 640×480, DISPLAY_SCALE=1.5
- 첫 실행 시 모델 다운로드 ~3초

## PlayWait 10/10 완료

W10이 10주 로드맵의 **마지막 게임**. 누적 회귀 678 cases, 모든 게임에서 친근 도형 UI + 안전 종료 + lazy import 0건 + 모듈명 prefix 패턴 일관.
