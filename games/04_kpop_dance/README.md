# 💃 K-Pop 랜덤 댄스 (KPop Random Dance) — W6

> Doby가 시범 보이는 5종 포즈(T자/Y자/왼손위/오른손위/박수)를 따라 추는 게임.
> 5라운드 / 4정답 이상 = 승리.

상세 설계: [W6_KPop댄스_기획서.md](./W6_KPop댄스_기획서.md)

## 빠른 시작

```bash
cd ~/PlayWait
# 단위 테스트 (73 cases, ~0.10s — 카메라/Pose 무관)
.venv/bin/pytest games/04_kpop_dance/tests/ -q

# 게임 실행 (카메라 필요, 전신이 화면에 보여야 함)
.venv/bin/python -m games.04_kpop_dance.src.game
```

## 조작

| 키 | 동작 |
|---|---|
| `1` / `2` / `3` | 난이도 (쉬움 3.5s·60% / 보통 2.5s·70% / 어려움 1.5s·80%) |
| `SPACE` | 게임 시작 |
| `R` | 재시작 |
| `Q` / `ESC` | 종료 |
| `F` / `+` / `-` / `0` | 전체 화면 / 확대 / 축소 / 리셋 |

## 페이즈 (W4 동일 구조)

```
DIFFICULTY_SELECT → READY ─(SPACE)─→ SHOW_DOBY (1.5s)
                                          ↓
                                    COUNTDOWN (3-2-1)
                                          ↓
                                    MEASURE (1.5~3.5s, 난이도별)
                                          ↓
                                    ROUND_RESULT (2s)
                                          ↓
                            ┌─ 5라운드 미달 → SHOW_DOBY
                            └─ 5라운드 완료 → GAME_OVER
```

## 5종 포즈

| 포즈 | 한글 | 핵심 |
|---|---|---|
| `t_pose` | T자 | 양팔 좌우 수평 |
| `y_pose` | Y자 | 양손 위로 V자 |
| `left_up` | 왼손 위 | 왼손은 머리 위, 오른팔은 옆 |
| `right_up` | 오른손 위 | 오른손 위, 왼팔 옆 (대칭) |
| `clap` | 박수 | 양손을 가슴 앞에 모음 |

## 모듈

| 파일 | 역할 | LOC |
|---|---|---:|
| `src/pose_features.py` | Pose 33 landmark → 각도·비율 dict (어깨 너비 정규화, 순수) | 126 |
| `src/pose_classifier.py` | 5종 포즈 profile + similarity (min 집계, 순수) | 145 |
| `src/dance_state.py` | 5라운드 출제·점수·종료 (RNG 격리, 순수) | 202 |
| `src/game.py` | 메인 게임 (7페이즈 + Pose + 안전 종료, lazy import 0건) | 603 |
| `src/theme.py` | W5 + 포즈 토큰 (한글/색상/Doby stick figure 색상) | (W5+α) |
| `src/ui_renderer.py` | W5 친근 도형 + **Doby stick figure 5종** + W6 화면 7종 | (W5+α) |
| `src/sound_manager.py` | W5 그대로 | 158 |

## 난이도

| 난이도 | 측정 시간 | 정답 임계 | 부분 임계 |
|---|---|---|---|
| 쉬움 | 3.5초 | 60% | 30% |
| 보통 ⭐ | 2.5초 | 70% | 40% |
| 어려움 | 1.5초 | 80% | 50% |

W4 표정 미러링보다 한 단계씩 시간 ↑ — 신체 동작 반응이 표정보다 느림.

## 점수

- 라운드별: 정답 100 / 부분 50 / 실패 0
- 만점 500, 4정답 이상 = WIN, 그 이하 = FINISHED

## 친근한 UI (W5 헬퍼 재활용 + W6 stick figure)

- **둥근 카드 패널** + 부드러운 그림자 (W5 그대로)
- **Doby stick figure**: 머리(둥근 광택 원) + 척추·어깨·다리(선) + 5종 포즈별 팔 위치
- **진행 점**: 정답 수를 채워진 윈 색 원 + 빈 회색 원으로 표시
- **우승 메달** (W5 그대로)

## 단위 테스트 (총 73 cases, ~0.10s)

| 파일 | 케이스 | 검증 |
|---|---:|---|
| `tests/test_pose_features.py` | 16 | 각도·정규화·scale-invariance·edge |
| `tests/test_pose_classifier.py` | 29 | 5종 profile perfect match + cross-pose 분리 |
| `tests/test_dance_state.py` | 28 | 출제 회피·시드 격리·점수·종료 사유·난이도 |

```bash
.venv/bin/pytest games/04_kpop_dance/tests/ -q
# 73 passed in 0.10s
```

## 환경 (W1~W5 그대로)

- Python 3.12, MediaPipe 0.10.14, NumPy 1.26.4, OpenCV 4.8.1
- 카메라 640×480, DISPLAY_SCALE=1.5
- 한글 폰트: NanumGothic
- **전신이 화면에 보여야 정확**: Pose는 어깨·팔꿈치·손목·골반·무릎·발목이 모두 캡쳐돼야 정확도 ↑
