# 고요 속의 외침 (Silent Charades) — W9 (LLM 도입)

Doby가 출제한 한국어 단어를 손짓·몸짓·표정으로 표현. 마지막 frame을 **Claude AI**가 보고 0~100점으로 평가.
5라운드 / 4정답 이상 = 승리.

상세 설계: [W9_고요속의외침_기획서.md](./W9_고요속의외침_기획서.md)

## 빠른 시작

```bash
cd ~/PlayWait
# 단위 테스트 (71 cases, ~0.11s — 카메라/API 무관, Anthropic SDK mock)
.venv/bin/pytest games/05_silent_charades/tests/ -q

# 게임 실행 — API 키 있으면 (권장)
ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python -m games.05_silent_charades.src.game

# API 키 없어도 동작 (fallback: 점수 50~70 랜덤)
.venv/bin/python -m games.05_silent_charades.src.game
```

## 조작

| 키 | 동작 |
|---|---|
| `1` / `2` / `3` | 난이도 (쉬움 20s·60% / 보통 15s·70% / 어려움 10s·80%) |
| `SPACE` | 게임 시작 |
| `R` | 재시작 |
| `Q` / `ESC` | 종료 |
| `F` / `+` / `-` / `0` | 전체 화면 / 확대 / 축소 / 리셋 |

## 페이즈 (W4 + EVALUATING)

```
DIFFICULTY_SELECT → READY ─(SPACE)─→ SHOW_WORD (2s, 단어 카드)
                                          ↓
                                    COUNTDOWN (3-2-1)
                                          ↓
                                    EXPRESS (10~20s, 손짓·몸짓 표현)
                                          ↓
                                    EVALUATING (Claude API 호출 ~2~3s)
                                          ↓
                                    ROUND_RESULT (3s, 점수 + 코멘트)
                                          ↓
                            ┌─ 5라운드 미달 → SHOW_WORD
                            └─ 5라운드 완료 → GAME_OVER
```

## 게임 메카닉

1. **Doby가 한국어 단어 출제** — 10개 풀에서 직전 회피 (강아지·비행기·농구·노래·잠자기·책 읽기·박수·춤추기·화남·놀람)
2. 사용자가 손짓·몸짓·표정으로 표현 (난이도별 10~20초)
3. 표현 종료 시점에 frame 1장을 캡처 → **Claude Haiku 4.5 (Vision)** 에 보내 평가
4. LLM이 0~100점 + 한국어 코멘트 반환
5. 임계: 정답(쉬움 60% / 보통 70% / 어려움 80%) → +100점, 부분 → +50점, 실패 → 0점
6. 5라운드 / 4정답 이상 → WIN

## LLM 통합

- **모델**: `claude-haiku-4-5` (저렴/빠름, vision + structured output 지원)
- **프롬프트 캐싱**: system 프롬프트(평가 가이드 + 단어 풀 hints)에 `cache_control: ephemeral` 적용 — Haiku 4.5는 최소 캐시 prefix 4096 토큰이라 silent miss될 수 있음 (비용 영향 미미)
- **Structured output**: `output_config.format` json_schema → JSON 응답 강제
- **Fallback**: `ANTHROPIC_API_KEY` 환경변수 없거나 호출 실패 시 → 50~70 랜덤 점수 + "API 키 없음" 코멘트 (게임 진행은 가능)

## 모듈

| 파일 | 역할 | LOC |
|---|---|---:|
| `src/silent_words.py` | 단어 풀 10개 + 출제기 (직전 회피, RNG 격리) | 92 |
| `src/silent_llm.py` | Claude Haiku 4.5 + Vision 평가 + Fallback (Anthropic SDK) | 280 |
| `src/silent_state.py` | 5라운드 진행·점수·종료 (W4 패턴, prefix) | 211 |
| `src/game.py` | 메인 게임 (8페이즈 + Pose + Hands + LLM, lazy import 0건) | 669 |
| `src/theme.py` | W8 + W9 토큰 (단어 카드 색, verdict 매핑) | (W8+α) |
| `src/ui_renderer.py` | W8 친근 도형 + 단어 카드 + 평가 스피너 + 결과 화면 | (W8+α) |
| `src/sound_manager.py` | W8 그대로 | 158 |

## 단위 테스트 (총 71 cases, ~0.11s)

| 파일 | 케이스 | 검증 |
|---|---:|---|
| `tests/test_silent_words.py` | 14 | 단어 풀 10개, 직전 회피, 시드 격리 |
| `tests/test_silent_llm.py` | 27 | 점수 clipping, 이미지 매직 바이트, fallback(no API key), Mock client API 성공/실패/잘못된 JSON, 캐시 친화 system 프롬프트 결정성 |
| `tests/test_silent_state.py` | 30 | outcome 변환, 라운드 누적, current_word 가드, 종료 우선순위, api_used_count |

**LLM 단위 테스트는 Anthropic SDK를 monkeypatch로 mock** — `client_factory` 파라미터로 가짜 client 주입. 네트워크/API 키 없이 검증 100%.

```bash
.venv/bin/pytest games/05_silent_charades/tests/ -q
# 71 passed in 0.11s
```

## API 비용

- 라운드당 1회 호출 (마지막 frame 1장 + 캐시된 system 프롬프트)
- 5라운드 = 5호출
- Haiku 4.5: $1.00 / 1M input + $5.00 / 1M output
- 1게임 약 **$0.01 미만** (예상치, vision 입력 포함)

## 환경

- Python 3.12, MediaPipe 0.10.14, OpenCV 4.8.1, NumPy 1.26.4
- **`anthropic` 패키지 필요**: `pip install anthropic` (없어도 fallback 모드는 동작)
- 카메라 640×480, DISPLAY_SCALE=1.5
- 상체 + 양손이 화면에 보이도록 자리잡으면 평가 정확도 ↑
