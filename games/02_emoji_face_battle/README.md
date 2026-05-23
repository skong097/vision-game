# 표정 미러링 챌린지 (Face Mirror Challenge) — W4

> Doby가 보여주는 표정(웃음/슬픔/놀람/화남)을 따라하는 미러링 게임.
> 5라운드 / 4정답 이상 = 승리.

상세 설계: [W4_표정미러링_기획서.md](./W4_표정미러링_기획서.md)

## 빠른 시작

```bash
cd ~/PlayWait
# 단위 테스트 (74 cases, ~0.1초)
.venv/bin/pytest games/02_emoji_face_battle/tests/ -q

# 게임 실행 (카메라 필요)
.venv/bin/python -m games.02_emoji_face_battle.src.game
```

## 조작

| 키 | 동작 |
|---|---|
| `1` / `2` / `3` | 난이도 선택 (쉬움 / 보통 / 어려움) |
| `SPACE` | 게임 시작 |
| `R` | 재시작 (게임 종료 후) |
| `Q` / `ESC` | 종료 |
| `F` | 전체 화면 토글 |
| `+` / `-` / `0` | 화면 확대 / 축소 / 리셋 |

## 페이즈

```
DIFFICULTY_SELECT → READY → SHOW_DOBY (1.5s) → COUNTDOWN (3s)
                              ↑                       ↓
                              │                   MEASURE (난이도별 1.2~3.0s)
                              │                       ↓
                              │                   ROUND_RESULT (2s)
                              │                       ↓
                              └── (5라운드 미달) ──→ GAME_OVER
```

## 모듈

| 파일 | 역할 | LOC |
|---|---|---:|
| `src/face_features.py` | Face Mesh landmark → 분류용 비율 feature dict (순수 함수) | 142 |
| `src/expression_classifier.py` | 4종 표정 profile + 유사도 / 분류 (순수 함수) | 155 |
| `src/round_state.py` | 5라운드 출제·점수·종료 조건 (순수 클래스, RNG 격리) | 260 |
| `src/game.py` | 메인 게임 (7페이즈 + Face Mesh + 안전 종료) | 618 |
| `src/theme.py` | PinkLAB 디자인 토큰 + W4 표정 색상·한글 라벨 | (W3+α) |
| `src/ui_renderer.py` | 한글 PIL 배치 + Doby 표정·게이지·HUD 컴포넌트 | (W3+α) |
| `src/sound_manager.py` | pygame.mixer 7종 (W3 그대로) | 158 |
| `src/face_mesh_test.py` | Face Mesh 단독 진단 (Step 2 검증용) | 159 |

## 난이도

| 난이도 | 측정 시간 | 정답 임계 | 부분 임계 |
|---|---|---|---|
| 쉬움 | 3.0초 | 60% | 30% |
| 보통 | 2.0초 | 70% | 40% |
| 어려움 | 1.2초 | 80% | 50% |

## 점수 / 종료

- 라운드마다: 정답 100 / 부분 정답 50 / 실패 0
- 만점 500점 (5라운드 × 100)
- **승리**: 5라운드 완료 + 정답 4회 이상 → `END_WIN`
- **도전 부족**: 5라운드 완료 + 정답 3회 이하 → `END_FINISHED`

## 단위 테스트 (총 74 cases, ~0.1초)

| 파일 | 케이스 | 검증 |
|---|---:|---|
| `tests/test_face_features.py` | 22 | feature 추출 정확성, scale-invariance |
| `tests/test_expression_classifier.py` | 19 | 4종 분류, 유사도 경계, NEUTRAL fallback |
| `tests/test_round_state.py` | 33 | 출제 회피·시드 격리, 점수·종료 사유·우선순위, 난이도별 임계 |

```bash
.venv/bin/pytest games/02_emoji_face_battle/tests/ -q
# ........... 74 passed in 0.10s
```

## 환경 (W1~W3 그대로)

- Python 3.12, MediaPipe 0.10.14, NumPy 1.26.4, OpenCV 4.8.1
- 카메라 640×480, DISPLAY_SCALE=1.5
- 한글 폰트: NanumGothic (자동 탐색)
