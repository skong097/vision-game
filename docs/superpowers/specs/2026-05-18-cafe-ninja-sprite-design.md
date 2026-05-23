# Cafe Ninja 폴리싱 — 닌자 스프라이트 + 배경 통합

날짜: 2026-05-18
대상 게임: `games/01_cafe_ninja`

## 목표
10개 게임 평가 단계에서 W3 카페 닌자에 다음 비주얼 폴리싱을 1차 적용:
1. 좌상단 코너 닌자 마스코트 (idle/attack/throw 애니메이션)
2. 떨어지는 슬라이스 타겟에 Kunai(표창) 7번째 종류 추가 (B1 안)
3. 배경 이미지 합성 (중국 거리 톤, 카페+동양 무드)

## 에셋
- 캐릭터: OpenGameArt "Ninja Adventure" (pzUH, CC0). 232x439 PNG × 90 프레임
- 배경: `assets/images/old-chinese-town-street-summer/*.jpg` (6622x2416)
- 게임 로컬 경로:
  - `games/01_cafe_ninja/assets/images/ninja/{idle,attack,throw}/`
  - `games/01_cafe_ninja/assets/images/background/town.jpg`

## 코드 변경
- 신규: `cafe_ninja_sprite.py` — `NinjaSprite` 클래스 (프레임 로드+리사이즈 캐시+상태 머신)
- `falling_object.py`: `KIND_KUNAI = "kunai"`, MENU_KINDS에 포함
- `spawner.py`: 메뉴 가중치에 kunai 10 추가
- `theme.py`: KIND_COLORS / KIND_LABEL / KIND_KOREAN에 kunai 엔트리 + SPRITE_DIR/BG 경로 상수
- `slice_judge.py`: KIND_SCORES["kunai"] = 30 (최고 점수)
- `ui_renderer.py`: `draw_background()` (cv2.addWeighted 25%), `draw_ninja_mascot(frame, sprite)`, `draw_falling_object` kunai PNG 분기
- `game.py`: NinjaSprite 1개 + 배경 1장 로드, render 흐름 (배경→객체→마스코트→HUD), slice 이벤트 시 sprite.set_state()

## 통합 흐름
```
render():
  draw_background(frame, bg_image)        # 25% 블렌딩
  for obj: draw_falling_object(frame, obj) # kunai는 PNG, 나머지 원
  draw_slice_flashes, finger_trail, tip
  draw_ninja_mascot(frame, ninja_sprite)  # 좌상단 ~116x220
  draw_score_bar (기존)
```

## 테스트
- 단위: cafe_ninja_sprite 프레임 로드/캐시/상태 전이 (3 케이스)
- Smoke: `-m` + 직접 실행 두 모드 모두 PLAYING 페이즈 진입 + Kunai 1회 스폰까지 확인

## Scope-out
- 사운드 신규 추가 X (slice.wav 그대로)
- 다른 9개 게임 영향 X
- 배경 GIF 애니메이션 X (정적 JPG)
