# 💕 커플 싱크 (Couple Sync) — W8 (발렌타인 시즌)

두 사람이 화면 좌·우에 자리잡고, **같은 미션 포즈를 동시에 일정 시간 유지하면 점수**. 60초 안에 목표 점수 달성 시 둘 다 우승.

상세 설계: [W8_커플싱크_기획서.md](./W8_커플싱크_기획서.md)

## 빠른 시작

```bash
cd ~/PlayWait
# 단위 테스트 (62 cases, ~0.08s — 카메라/Pose 무관)
.venv/bin/pytest games/10_couple_sync/tests/ -q

# 게임 실행 (카메라 필요, 두 사람이 좌·우에 위치)
.venv/bin/python -m games.10_couple_sync.src.game
```

## 조작

| 키 | 동작 |
|---|---|
| `1` / `2` / `3` | 난이도 (쉬움 80점·유지 0.5s / 보통 120점·0.7s / 어려움 200점·1.0s) |
| `SPACE` | 게임 시작 |
| `R` | 재시작 |
| `Q` / `ESC` | 종료 |
| `F` / `+` / `-` / `0` | 전체 화면 / 확대 / 축소 / 리셋 |

## 페이즈 (W3·W7 패턴 — 실시간 시뮬레이션 + 미션 회전)

```
DIFFICULTY_SELECT → READY ─(SPACE)─→ PLAYING (60초)
                                          ↓
                            WIN(점수≥목표) | TIMEOUT
                                          ↓
                                    GAME_OVER
```

## 게임 메카닉

- 카메라 frame을 좌·우 분할 → **각 영역에 MediaPipe Pose 1번씩 추론** (총 2번)
- 각 사람의 `pose_features` 추출 → 미션 포즈에 대한 `pose_similarity` 계산
- 두 사람의 **min(similarity) ≥ 임계** + 이 상태를 **유지 시간** 이상 지속 = 미션 완료
- 미션 완료: **+20점** (+ 둘 다 0.80 이상이면 보너스 +5)
- 새 미션 자동 출제 (직전 회피)
- 매칭 끊김 시 유지 시간 0으로 리셋
- 종료: 점수 ≥ 목표 → WIN, 60초 경과 → TIMEOUT

## 미션 포즈 (W6 그대로 재활용)

T자 / Y자 / 왼손 위 / 오른손 위 / 박수 — 5종.

## 모듈

| 파일 | 역할 | LOC |
|---|---|---:|
| `src/pose_features.py` | Pose landmark → 각도/비율 dict (W6 복사) | 126 |
| `src/pose_classifier.py` | 5종 포즈 profile + similarity (W6 복사) | 145 |
| `src/couple_detector.py` | 좌/우 분할 + ROI landmark 좌표 → 전체 frame 좌표 변환 (순수) | 100 |
| `src/sync_judge.py` | 두 features → 매칭 + min 집계 + 보너스 (순수) | 108 |
| `src/sync_state.py` | 미션 회전 + 유지 시간 누적 + 점수·종료 (순수, time_provider) | 245 |
| `src/game.py` | 메인 게임 (4페이즈 + Pose × 2 + 안전 종료, lazy import 0건) | 633 |
| `src/theme.py` | W7 + 발렌타인 토큰 (하트, 좌/우 컬러) | (W7+α) |
| `src/ui_renderer.py` | W7 친근 도형 + **하트 도형 + 듀얼 박스** + 화면 4종 | (W7+α) |
| `src/sound_manager.py` | W7 그대로 | 157 |

## 멀티 인물 처리 (V1)

MediaPipe Pose는 single-person 솔루션. V1은 다음 방식:

1. 카메라 frame을 좌(0~width/2) / 우(width/2~width) 두 영역으로 잘라
2. **각 영역에 별도 Pose 인스턴스로 추론** (총 2번)
3. 좌·우에서 잡힌 landmark의 x 좌표를 전체 frame 좌표계로 변환 (`couple_detector.remap_landmarks`)
4. 각 사람의 features → similarity → `sync_judge`

→ FPS는 다소 줄지만 추가 의존성 없이 멀티 인물 지원. Face Mesh는 V2 검토.

## 친근 UI (W5/W6/W7 헬퍼 재활용 + W8 신규)

- **둥근 카드** + 부드러운 그림자 (W5)
- **하트 도형** (W8 신규): 두 원 + 삼각형 합성 + 좌상단 광택. 싱크 상태에 따라:
  - 둘 다 미매칭: 회색 빈 하트
  - 둘 다 매칭: 핑크 채워진 하트, **유지 비율로 1.0 → 1.5x 크기**
  - 보너스 (둘 다 0.80 이상): **레드** 하트
- **듀얼 박스** (W8 신규): 화면 좌/우 영역 라벨 + Pose 잡힘/미잡힘 색 차별 + 분류된 포즈 한글 표시
- **우승 화면**: 큰 레드 하트 + "둘 다 쿠폰" 메시지

## 단위 테스트 (총 62 cases, ~0.08s)

| 파일 | 케이스 | 검증 |
|---|---:|---|
| `tests/test_couple_detector.py` | 17 | 좌/우 좌표 변환·경계·None edge·튜플 호환 |
| `tests/test_sync_judge.py` | 13 | 둘 다 매칭/한쪽만/둘 다 X·min 집계·보너스 임계·None |
| `tests/test_sync_state.py` | 32 | 미션 회전·유지 시간 누적·끊김 리셋·종료 우선순위·time_provider |

```bash
.venv/bin/pytest games/10_couple_sync/tests/ -q
# 62 passed in 0.08s
```

## 환경

- Python 3.12, MediaPipe 0.10.14, NumPy 1.26.4, OpenCV 4.8.1
- 카메라 640×480, DISPLAY_SCALE=1.5
- **Pose 두 인스턴스** → FPS 부담 ↑. model_complexity=0 사용 (정확도보다 FPS 우선)
- 두 사람이 화면 좌·우에 충분히 떨어져 자리잡아야 정확
