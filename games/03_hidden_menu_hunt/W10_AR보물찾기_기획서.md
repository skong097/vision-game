# 🔍 AR 보물찾기 (Hidden Menu Hunt) — W10 (마지막 게임)

> **PlayWait W10 — 열 번째 게임 (로드맵 마지막)**
> **개발 시작**: 2026-05-12
> **난이도**: ⭐⭐⭐⭐⭐
> **기술**: YOLO (객체 검출) + AR overlay (cv2 도형 텍스트)

---

## 📌 게임 개요

**매장의 평범한 사물(컵, 책, 휴대폰...)이 Doby의 비밀 보물!** Doby가 미션 사물을 출제하면, 손님은 매장을 돌아다니며 카메라로 해당 사물을 화면 가운데에 1초 유지 → 발견 완료 + AR 힌트 표시 → 다음 보물. 60초 안에 N개 발견 시 승리.

### 핵심 가치 (B2B2C)

| 이해관계자 | 효과 |
|---|---|
| **손님 (C)** | 매장을 능동적으로 탐험하는 재미. AR 힌트로 매장 정보·메뉴 정보 자연 노출 |
| **매장 (B)** | 사물 = 메뉴/굿즈/소품 → 자연 노출. AR 힌트에 매장-특화 메시지 삽입 가능 |
| **플랫폼 (B)** | 매장별 객체 분포 데이터 — W5 컬러 헌트와 결합하면 풍부한 카탈로그 |

### W5 컬러 헌트와 차별점

| | W5 컬러 헌트 | W10 AR 보물찾기 |
|---|---|---|
| 매칭 기준 | 색(HSV) | 객체 클래스(YOLO COCO) |
| 발견 조건 | 화면 어디서든 객체가 색 매칭 | **화면 중앙 ROI에 1초 유지** (정조준 요구) |
| UI | 색 swatch + 박스 | **AR 힌트 카드 + 중앙 타겟 + 발견 애니메이션** |
| 컨셉 | 색 미션 (시즌 메뉴 색 등) | 보물 (매장 활용도 ↑) |

---

## 🎮 게임 규칙

### 페이즈 (W3·W7 패턴)

```
DIFFICULTY_SELECT → READY ─(SPACE)─→ PLAYING (60초)
                                          ↓
                            WIN(N개 발견) | TIMEOUT
                                          ↓
                                    GAME_OVER
```

### 게임 메카닉

1. Doby가 보물 풀에서 1개 출제 (예: "cup")
2. 화면 중앙에 십자 타겟 + 보물 이름 + AR 힌트 표시
3. 사용자가 카메라로 매장을 돌아다님 — YOLO가 객체 검출
4. **중앙 ROI(화면 중앙 30% 영역)에 보물 클래스 객체 중심이 들어옴** → 유지 시간 누적
5. 1.0초(보통) 유지 → **발견!** AR 박스 깜빡임 + 사운드 + +20점 + 다음 보물
6. 60초 안에 목표 발견 수 도달 → WIN

### 난이도

| 난이도 | 발견 목표 | 유지 시간 | 보물 풀 |
|---|---|---|---|
| Easy | 3개 | 0.7s | 흔한 (cup, book, cell phone, bottle, chair) |
| Normal | 5개 | 1.0s | 8개 (위 + laptop, keyboard, vase) |
| Hard | 7개 | 1.5s | 12개 (위 + scissors, mouse, fork, spoon, banana) |

점수: 발견 1개 = 20점, 목표 도달 시 잔여 시간 보너스 (남은 초 × 1점).

---

## 🛠️ 모듈 설계

```
games/03_hidden_menu_hunt/
├── README.md (마무리)
├── W10_AR보물찾기_기획서.md  # ✅ 본 문서
├── src/
│   ├── game.py                 # 5페이즈 + YOLO + AR (lazy import 0)
│   ├── treasure_clues.py       # 보물 풀 + 출제기 (순수)
│   ├── treasure_state.py       # 60초·발견 카운트·유지 시간 (순수, time_provider)
│   ├── ar_overlay.py           # 중앙 ROI 판정 + AR 도형 (cv2 부분 분리)
│   ├── theme.py                # W9 + W10 토큰
│   ├── ui_renderer.py          # W9 친근 도형 + AR 컴포넌트
│   └── sound_manager.py        # W9 그대로
├── assets/sounds/              # W9 7종 복사
└── tests/
    ├── conftest.py
    ├── test_treasure_clues.py
    ├── test_treasure_state.py
    └── test_ar_overlay.py
```

모듈명 prefix: `treasure_*`, `ar_*`. W5의 `hunt_tracker`와 이름 충돌 회피 (W5 트러블 #17).

YoloEngine은 **`core/vision/yolo_engine.py` 재활용** (W5에서 동일 패턴 — duck-typed detector).

---

## 🎨 UI/UX

### AR 컴포넌트 (W10 신규)

- **중앙 타겟**: 화면 가운데 ROI 사각형 외곽 + 4 모서리 핑크 점 + 십자선
- **AR 보물 박스**: 발견된 객체 박스를 둥근 모서리 + 글로우 효과
- **AR 힌트 라벨**: 객체 상단에 "🎯 cup 발견 중..." (이모지 없이) + 진행 게이지
- **발견 플래시**: 발견 순간 화면 핑크 깜빡임 + 점수 팝업
- **HUD**: 좌측 보물 카드 (현재 미션 + 한글) / 중앙 진행 점 / 우측 시간

### 보물 카드

| 영문 클래스 | 한글 | AR 힌트 |
|---|---|---|
| cup | 컵 | 카페의 단짝! 라떼·아메리카노 |
| bottle | 병 | 물·음료 |
| book | 책 | 조용한 시간 |
| cell phone | 휴대폰 | SNS 공유 |
| chair | 의자 | 편안한 자리 |
| laptop | 노트북 | 카공족 친구 |
| keyboard | 키보드 | 타이핑 박자 |
| vase | 화병 | 인테리어 포인트 |
| scissors | 가위 | 작은 도구 |
| mouse | 마우스 | 노트북 친구 |
| fork | 포크 | 디저트 |
| spoon | 스푼 | 커피·차 |

(banana는 한국 카페 환경엔 흔치 않으나 COCO 클래스에 있어 추가 가능)

---

## ✅ Definition of Done

- [ ] treasure_clues: 보물 풀 + 출제기 단위 테스트 (직전 회피, 시드 격리)
- [ ] treasure_state: 60초·발견 카운트·유지 시간 누적·종료 사유 단위 테스트
- [ ] ar_overlay: 중앙 ROI 판정 (순수) 단위 테스트
- [ ] 5페이즈 상태머신 + YOLO + AR overlay
- [ ] **lazy import 0건** (W4 트러블 #16)
- [ ] **모듈명 prefix** treasure_/ar_ (W5 트러블 #17)
- [ ] 친근 도형 UI (W5~W9 헬퍼 재활용) + AR 박스·타겟 신규
- [ ] 두 실행 모드 smoke 통과
- [ ] 단위 테스트 ≥ 50건
- [ ] README + 일별 로그
- [ ] (Stretch) 실기 — YOLO 모델 + 카메라 환경에서 5라운드

---

**문서 버전**: v1.0
**최종 수정**: 2026-05-12
