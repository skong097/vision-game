# 💃 K-Pop 랜덤 댄스 (KPop Random Dance) — W6

> **PlayWait W6 — 여섯 번째 게임 / 첫 MediaPipe Pose 도입**
> **개발 시작**: 2026-05-12 (W5 마무리 직후 같은 날)
> **개발 기간**: 1주 (W6)
> **난이도**: ⭐⭐⭐⭐
> **기술**: MediaPipe Pose (33 landmark)

---

## 📌 게임 개요

**Doby가 시범을 보이는 K-Pop 안무 포즈를 따라 추는 게임.** 5라운드 동안 5종 포즈 중 1개씩 출제 → 손님이 정해진 시간 안에 같은 포즈를 취하면 점수.

W4(표정 미러링)의 "Doby 시범 → 카운트다운 → 측정" 흐름을 신체 포즈로 확장.

### 핵심 가치 (B2B2C)

| 이해관계자 | 효과 |
|---|---|
| **손님 (C)** | 몸을 쓰는 활성 게임 → 단순 셀카보다 흥겨움. K-Pop 곡 BGM 미래 확장 가능 |
| **매장 (B)** | 시각적 활기 → 다른 손님의 호기심 유발, 입소문. SNS 영상 공유 자연스러움 |
| **플랫폼 (B)** | Pose 데이터 → 매장 별 인기 동작 분포 분석 |

### 핵심 특징

- **5종 포즈**: T자(양팔 좌우) / Y자(양팔 위) / 왼손 위(오른손 옆) / 오른손 위(왼손 옆) / 박수(양손 모음)
- **각도·비율 기반 분류**: 양팔 각도(어깨-팔꿈치-손목), 손 높이(어깨 대비), 손 사이 거리
- **5라운드 60초 한 판**: 4정답 이상 = 승리
- **정확도 % 표시**: 단순 정답/오답 X — W4와 동일 gradient (0~100%)

---

## 🎮 게임 규칙

### 5종 포즈 정의

| 포즈 | 핵심 특징 | 주요 feature |
|---|---|---|
| **T자 (t_pose)** | 양팔 좌우로 수평 펴기 | left_arm_angle ≈ 180°, right_arm_angle ≈ 180°, hand_y_offset ≈ 0 |
| **Y자 (y_pose)** | 양팔 위로 V자 | hand_y_above_shoulder > 임계, arm_angles 펴짐 |
| **왼손 위 (left_up)** | 왼손만 머리 위, 오른팔 옆 | left_hand_above >>, right_hand_at_shoulder |
| **오른손 위 (right_up)** | 오른손만 머리 위, 왼팔 옆 | right_hand_above >>, left_hand_at_shoulder |
| **박수 (clap)** | 양손을 가슴 앞으로 모음 | hand_distance < 임계, hand_y ≈ 어깨 |

### 게임 진행 (W4 페이즈 그대로)

```
DIFFICULTY_SELECT → READY ─(SPACE)─→ SHOW_DOBY (Doby 포즈 1.5s)
                                          ↓
                                    COUNTDOWN (3s)
                                          ↓
                                    MEASURE (난이도별 1.5~3.5s)
                                          ↓
                                    ROUND_RESULT (2s)
                                          ↓
                            ┌─ 5라운드 미달 → SHOW_DOBY
                            └─ 5라운드 완료 → GAME_OVER
```

### 정확도 계산 (W4 패턴 그대로)

각 라운드 측정 시간 동안 매 프레임 `pose_similarity(target, features)` 계산 → **최고값을 최종 정확도**로 채택 (W4 결정). EMA로 게이지 부드럽게 보간.

### 난이도

| 난이도 | 측정 시간 | 정답 임계 | 부분 임계 |
|---|---|---|---|
| Easy | 3.5초 | 60% | 30% |
| Normal ⭐ | 2.5초 | 70% | 40% |
| Hard | 1.5초 | 80% | 50% |

W4 표정보다 한 단계씩 시간 늘림 — 신체 동작 반응이 표정보다 느림.

### 점수

- 라운드별: 정답 100 / 부분 50 / 실패 0
- 만점 500 (5라운드 × 100)
- 4정답 이상 = WIN, 그 이하 = FINISHED (도전 부족) — W4와 동일

---

## 🛠️ 기술 설계

### MediaPipe Pose

```python
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
```

33 landmark, 핵심 인덱스:
- 어깨: 11(좌)·12(우)
- 팔꿈치: 13(좌)·14(우)
- 손목: 15(좌)·16(우)

### Feature 추출

```python
def extract_pose_features(landmarks) -> dict:
    """Pose 33 landmark → 분류용 feature dict
    Returns:
        {
            "left_arm_angle":  float,   # 어깨-팔꿈치-손목 각도 (deg)
            "right_arm_angle": float,
            "left_hand_y_off": float,   # 손 y - 어깨 y (음수=위) / 어깨너비
            "right_hand_y_off": float,
            "hand_distance":   float,   # 양 손 사이 거리 / 어깨너비
            "hand_y_avg":      float,   # 양 손 평균 y / 어깨너비
        }
    """
```

scale-invariant — 어깨 너비로 정규화. y는 OpenCV 좌표라 위로 갈수록 작음 → "위" = 음수.

### 분류 / 유사도

```python
def pose_similarity(target: str, features: dict) -> float:
    """target 포즈 profile에 features가 얼마나 부합하는지 0~1.
    집계: min(per-feature score) — W4 expression과 동일 strict."""
```

---

## 🎨 UI/UX

W5의 **친근 도형 헬퍼**(둥근 카드, 큰 색 점, 진행 점, 우승 메달) 100% 재활용.
W6 신규는 **Doby Stick Figure**: 어깨·팔꿈치·손목을 작은 원으로, 선으로 연결 → 5종 포즈 시범.

### 화면 구성

```
┌──────────────────────────────────────────┐
│  R 3/5 | 점수 200 | 정확도 78%            │ 둥근 HUD
├──────────────────────────────────────────┤
│      [Doby Stick Figure — T자]           │
│            "T자!"                          │
│      "양팔을 옆으로 펴세요"                  │
├──────────────────────────────────────────┤
│      [정확도 게이지 ████░░ 70%]            │
│      [남은 시간 2.1s / 2.5s]                │
└──────────────────────────────────────────┘
```

---

## 📂 파일 구조

```
games/04_kpop_dance/
├── README.md                  # 마무리 시 작성
├── W6_KPop댄스_기획서.md      # ✅ 본 문서
├── src/
│   ├── __init__.py
│   ├── game.py                # 메인 게임 (7페이즈)
│   ├── pose_features.py       # Pose landmark → feature (순수)
│   ├── pose_classifier.py     # 5종 분류 + similarity (순수)
│   ├── dance_state.py         # 5라운드 진행 (순수, prefix 적용)
│   ├── theme.py               # W5 + 포즈 토큰
│   ├── ui_renderer.py         # W5 친근 도형 + Doby stick figure
│   └── sound_manager.py       # W5 그대로
├── assets/sounds/             # W5에서 복사 (7종)
└── tests/                     # pytest
    ├── conftest.py
    ├── test_pose_features.py
    ├── test_pose_classifier.py
    └── test_dance_state.py
```

(`tests/__init__.py`는 만들지 않음 — W3 트러블 #12)

---

## 🔄 W5 → W6 재활용 모듈

| 모듈 | 재활용도 | 비고 |
|---|---|---|
| `theme.py` | 95% | 포즈 5종 한글/색상 토큰 추가 |
| `ui_renderer.py` | 100% + W6 stick figure | 친근 도형 헬퍼 그대로 |
| `sound_manager.py` | 100% | 그대로 |
| `assets/sounds/` | 100% | 7종 복사 |
| 안전 종료 패턴 | 100% | try-finally + signal + atexit + pose.close() |
| 페이즈 상태머신 | W4와 동일 7페이즈 |

---

## ✅ Definition of Done

- [ ] Pose 단독 import 정상 (refine_landmarks=False 기본)
- [ ] 5종 포즈 본인 몸으로 80%+ 인식률 (실기 캘리브레이션 후)
- [ ] 7페이즈 상태머신 정확 동작
- [ ] 단위 테스트 ≥ 50건 (features + classifier + state)
- [ ] 친근 UI (둥근 카드 + 스틱 피겨 + 진행 점) 한글 표시
- [ ] 안전 종료 (try-finally + signal + atexit + pose.close)
- [ ] **lazy import 0건** (W4 트러블 #16 강제 규칙)
- [ ] **모듈명 prefix 적용** (W5 트러블 #17 학습): pose_features / pose_classifier / dance_state
- [ ] 두 실행 모드 smoke 통과: `-m games.04_kpop_dance.src.game` + 직접 실행

---

## 🤔 결정 (기본값, 사용자 확인)

1. **포즈 5종**: T자/Y자/왼손위/오른손위/박수 — 화면 안에서 양손 다 보이는 동작 위주
2. **출제 방식**: 랜덤 5종 중 1, 직전 회피
3. **Doby 표시**: 작은 stick figure (어깨-팔꿈치-손목 점·선) — OpenCV 도형
4. **AI 점수 없음**: 손님 단독 도전형 (W4와 동일)
5. **BGM**: 본 회차에서는 미도입 — 기존 사운드 7종으로 충분. 향후 K-Pop 라이선스 곡 통합 시 별도 트랙

---

**문서 버전**: v1.0
**최종 수정**: 2026-05-12
