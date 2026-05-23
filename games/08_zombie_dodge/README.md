# 좀비 피하기 (Zombie Dodge) — W7 (할로윈 시즌)

위에서 떨어지는 좀비를 몸으로 피하는 액션 게임.
**60초 / 생명 3 / 목표 점수 달성 시 승리.**

상세 설계: [W7_좀비피하기_기획서.md](./W7_좀비피하기_기획서.md)

## 빠른 시작

```bash
cd ~/PlayWait
# 단위 테스트 (83 cases, ~0.13s — 카메라/Pose 무관)
.venv/bin/pytest games/08_zombie_dodge/tests/ -q

# 게임 실행 (카메라 필요, 상체가 화면에 보여야 정확)
.venv/bin/python -m games.08_zombie_dodge.src.game
```

## 조작

| 키 | 동작 |
|---|---|
| `1` / `2` / `3` | 난이도 (쉬움 100점 / 보통 200점 / 어려움 350점) |
| `SPACE` | 게임 시작 |
| `R` | 재시작 |
| `Q` / `ESC` | 종료 |
| `F` / `+` / `-` / `0` | 전체 화면 / 확대 / 축소 / 리셋 |

## 페이즈 (W3 패턴 — 실시간 시뮬레이션)

```
DIFFICULTY_SELECT → READY ─(SPACE)─→ PLAYING (60초)
                                          ↓
                            WIN(점수≥목표) | LIVES_OUT(생명0) | TIMEOUT
                                          ↓
                                    GAME_OVER
```

## 게임 메카닉

- 좀비가 화면 위에서 떨어짐 (난이도별 spawn rate)
- Pose로 손님의 머리·어깨 **회피 박스 (AABB)** 자동 추적
- 좀비가 박스를 빠져나가 화면 하단 통과 → **+10점**
- 좀비가 박스와 충돌 → **생명 -1** (좀비 즉시 제거)
- 목표 점수 도달 / 생명 0 / 60초 경과 시 종료

### 좀비 종류

| 종류 | 한글 | 특징 |
|---|---|---|
| `normal` | 좀비 | 그린, 보통 속도 (모든 난이도 등장) |
| `fast` | 스피드 좀비 | 보라, 1.6x 속도, 작음 (Normal/Hard) |
| `big` | 거대 좀비 | 차콜, 0.8x 속도, 크기 큼 |

## 모듈

| 파일 | 역할 | LOC |
|---|---|---:|
| `src/body_tracker.py` | Pose → 회피 박스 AABB (어깨 너비 정규화, 순수) | 153 |
| `src/zombie.py` | Zombie dataclass + step·위치 (순수) | 97 |
| `src/zombie_spawner.py` | 난이도별 spawn (RNG 격리, 순수) | 176 |
| `src/dodge_judge.py` | AABB-원 충돌 + 통과 판정 + 이중 카운트 방지 | 97 |
| `src/dodge_state.py` | 점수·생명·시간·종료 (W3 score_state 패턴, time_provider 주입) | 178 |
| `src/game.py` | 메인 게임 (4페이즈 + Pose + 안전 종료, lazy import 0건) | 550 |
| `src/theme.py` | W6 + 좀비 토큰 | 331 |
| `src/ui_renderer.py` | W6 친근 도형 + 좀비 도형 + 회피 박스 시각화 + 화면 4종 | 2173 |
| `src/sound_manager.py` | W6 그대로 | 157 |

## 회피 박스 정의

```python
# body_tracker.compute_dodge_box():
- 좌우: 어깨 좌우 끝 ± shoulder_pad (어깨너비 × 0.20)
- 상단: 코(NOSE) y - head_radius (어깨너비 × 0.45)
- 하단: 어깨 평균 y + body_height (어깨너비 × 1.3)
```

→ **scale-invariant**: 카메라 거리·신장에 무관하게 박스가 사람 비율에 맞춤.

## 친근 UI (W5/W6 헬퍼 재활용 + 좀비 도형)

- **둥근 카드** + 부드러운 그림자 (W5)
- **둥근 색 원 + 광택** (W5) — 좀비 본체에도 적용
- **진행 점** (W5) — 생명을 채워진 핑크 원 / 잃은 회색 원으로 표시
- **우승 메달** (W5) — 우승 시 큰 둥근 메달
- **좀비 도형 (W7 신규)**: 둥근 본체 + 흰 흰자/빨강 동공 + 입가 흰 이빨 세로선 3개
- **회피 박스 (W7 신규)**: 핑크 둥근 외곽선 + 4 모서리 핑크 점. 충돌 임박 시 노란색

## 단위 테스트 (총 83 cases, ~0.13s)

| 파일 | 케이스 | 검증 |
|---|---:|---|
| `tests/test_body_tracker.py` | 14 | 박스 좌표·scale-invariance·clipping·None edge |
| `tests/test_zombie.py` | 14 | step / position / 종류별 속성 |
| `tests/test_zombie_spawner.py` | 13 | spawn timing·시드 격리·종류 분포·난이도 단조성 |
| `tests/test_dodge_judge.py` | 17 | AABB-원 충돌·이미 마킹 좀비 skip·box None·우선순위 |
| `tests/test_dodge_state.py` | 25 | 점수·생명·종료 사유 우선순위·time_provider·캐시 |

## 환경

- Python 3.12, MediaPipe 0.10.14, NumPy 1.26.4, OpenCV 4.8.1
- 카메라 640×480, DISPLAY_SCALE=1.5
- **상체가 화면에 보여야 정확**: Pose 코+양 어깨가 잡혀야 회피 박스 형성
