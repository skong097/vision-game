"""
silent_llm.py — Claude API 평가 + Fallback
===================================================

evaluate_expression(word, image_bytes) → EvalResult(score, comment, used_api).

ANTHROPIC_API_KEY 환경변수가 없으면 rule-based fallback로 동작 (게임은 진행 가능).
API 키가 있으면 Claude Haiku 4.5 + Vision으로 평가.

설계 결정
---------
- **모델**: claude-haiku-4-5 — 저렴/빠름, vision 지원, structured output 지원
- **system 프롬프트 캐싱**: `cache_control` ephemeral — 안정 (단어 풀 hints 포함).
  Haiku 4.5 최소 캐시 prefix는 4096 토큰이라 현재 시스템 프롬프트(~1500)는 캐시
  적중이 안 될 수 있음(silent miss). 비용 영향 미미.
- **structured output**: `output_config.format` json_schema → 응답 형식 강제
- **에러 분기**: API 호출 실패 시 fallback (게임 끊김 방지)

W4 트러블 #16 학습: lazy import 0건 — 단, anthropic 패키지는 fallback에서 필요 X.
API 사용 분기에서만 import (단위 테스트 시 패키지 없어도 fallback 검증 가능).

Author: Stephen (gjkong)
Date: 2026-05-12 (W9 Step 3)
"""

import base64
import json
import os
import random
from dataclasses import dataclass

try:
    from .silent_words import WORD_CATALOG, get_word_by_slug
except ImportError:
    from silent_words import WORD_CATALOG, get_word_by_slug


# ============================================================
# 1. 결과 dataclass
# ============================================================
@dataclass(frozen=True)
class EvalResult:
    """평가 결과.

    Attributes:
        score: 0~100 (정수, 클리핑됨)
        comment: 한국어 코멘트 (30자 이내 권장)
        used_api: True면 Claude API 결과, False면 fallback
    """
    score: int
    comment: str
    used_api: bool


# ============================================================
# 2. 상수
# ============================================================
MODEL_ID = "claude-haiku-4-5"
MAX_TOKENS = 1024            # 응답 형식이 짧음 (JSON ~50 토큰)
# JSON output 스키마
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "comment": {
            "type": "string",
            "maxLength": 60,
        },
    },
    "required": ["score", "comment"],
    "additionalProperties": False,
}

# Fallback 점수 범위
FALLBACK_SCORE_MIN = 50
FALLBACK_SCORE_MAX = 70
FALLBACK_COMMENT = "API 키 없음 — 자세 평가 생략"
API_ERROR_COMMENT = "평가 중 오류 발생 — fallback"


# ============================================================
# 3. system 프롬프트 (캐시 후보)
# ============================================================
def _build_system_prompt() -> str:
    """안정적인 system 프롬프트 — 평가 가이드 + 단어 풀 핵심 동작."""
    lines = [
        "당신은 사용자가 한국어 단어를 손짓·몸짓·표정으로 표현했는지 평가하는 친절한 심사위원이에요.",
        "사용자가 어떤 단어를 표현했는지 출제 단어가 함께 주어집니다.",
        "이미지에서 단어와 일치하는 핵심 동작이 보이면 높은 점수, 그렇지 않으면 낮은 점수.",
        "",
        "점수 가이드:",
        "- 핵심 동작 명확 + 자세 안정 → 80~100점",
        "- 핵심 동작 일부 + 자세 흔들림 → 40~70점",
        "- 단어와 무관한 자세 → 0~30점",
        "",
        "단어별 핵심 동작 (참고용):",
    ]
    for w in WORD_CATALOG:
        lines.append(f"- {w.ko}: {w.hint}")
    lines += [
        "",
        "응답 규칙:",
        "- JSON 스키마 {score: 0~100 정수, comment: 한국어 30자 이내}",
        "- 코멘트는 친절하고 격려하는 톤 (예: '귀가 잘 보였어요!')",
        "- 사용자가 무관한 자세를 취해도 무례하지 않게, 짧고 가볍게 코멘트.",
    ]
    return "\n".join(lines)


# 미리 빌드해두기 (인스턴스 사이에서 동일 — 캐시 친화)
SYSTEM_PROMPT = _build_system_prompt()


# ============================================================
# 4. 이미지 포맷 자동 감지
# ============================================================
def _detect_image_media_type(image_bytes: bytes) -> str:
    """매직 바이트로 JPEG/PNG/WebP 구별. 기본은 JPEG."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if (len(image_bytes) >= 12
            and image_bytes[:4] == b"RIFF"
            and image_bytes[8:12] == b"WEBP"):
        return "image/webp"
    return "image/jpeg"


# ============================================================
# 5. 점수 clipping
# ============================================================
def _clip_score(value) -> int:
    """LLM이 잘못된 타입을 줘도 0~100 안에 들어오게."""
    try:
        s = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, s))


# ============================================================
# 6. Fallback (API 키 없음 또는 호출 실패)
# ============================================================
def _fallback_eval(reason: str = FALLBACK_COMMENT, seed=None) -> EvalResult:
    """규칙 기반 fallback — 게임 진행만 가능하게 평균값 반환.

    Args:
        reason: 코멘트 (기본은 "API 키 없음").
        seed: 단위 테스트용. None이면 시스템 random.
    """
    rng = random.Random(seed)
    score = rng.randint(FALLBACK_SCORE_MIN, FALLBACK_SCORE_MAX)
    return EvalResult(score=score, comment=reason, used_api=False)


# ============================================================
# 7. Claude API 호출 (핵심 로직)
# ============================================================
def _call_claude_api(
    word: str,
    image_bytes: bytes,
    api_key: str,
    client_factory=None,
) -> EvalResult:
    """Claude Haiku 4.5 호출 — vision + structured output.

    Args:
        word: 출제 단어 (한국어)
        image_bytes: JPEG/PNG/WebP bytes
        api_key: API 키
        client_factory: 테스트용 — anthropic.Anthropic 대체

    Returns:
        EvalResult(used_api=True)

    Raises:
        모든 anthropic 예외는 호출측에서 fallback으로 변환.
    """
    if client_factory is None:
        import anthropic
        client_factory = anthropic.Anthropic

    client = client_factory(api_key=api_key)
    media_type = _detect_image_media_type(image_bytes)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        cache_control={"type": "ephemeral"},
        system=SYSTEM_PROMPT,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": OUTPUT_SCHEMA,
            },
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"출제 단어: {word}\n이 표현을 평가해주세요.",
                    },
                ],
            }
        ],
    )

    # 응답의 첫 text 블록 추출 → JSON 파싱
    text = ""
    for block in response.content:
        # 블록 타입 검사 — duck-typed (mock 호환)
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = block.text
            break

    if not text:
        raise ValueError("Claude API에서 텍스트 응답을 받지 못함")

    data = json.loads(text)
    score = _clip_score(data.get("score", 0))
    comment = str(data.get("comment", "")).strip()
    if not comment:
        comment = "평가 완료"
    return EvalResult(score=score, comment=comment, used_api=True)


# ============================================================
# 8. 공개 API
# ============================================================
def evaluate_expression(
    word: str,
    image_bytes: bytes,
    *,
    api_key: str = None,
    client_factory=None,
    fallback_seed=None,
) -> EvalResult:
    """단어 + 이미지로 사용자 표현 평가.

    Args:
        word: 출제 단어 (한국어, 예: "강아지")
        image_bytes: JPEG/PNG/WebP frame
        api_key: 명시적 키. None이면 ANTHROPIC_API_KEY 환경변수.
        client_factory: 테스트용 — anthropic.Anthropic 대체
        fallback_seed: 테스트용 — fallback 시 RNG 시드

    Returns:
        EvalResult. API 키 없거나 호출 실패 시 fallback.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _fallback_eval(seed=fallback_seed)

    try:
        return _call_claude_api(
            word=word,
            image_bytes=image_bytes,
            api_key=key,
            client_factory=client_factory,
        )
    except Exception:
        # 게임 끊김 방지 — 어떤 예외든 fallback
        return _fallback_eval(reason=API_ERROR_COMMENT, seed=fallback_seed)
