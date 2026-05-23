"""
test_silent_llm.py — silent_llm 단위 테스트
====================================================

검증:
- API 키 없으면 fallback (used_api=False, 점수 50~70)
- 이미지 매직 바이트 감지 (PNG/JPEG/WebP)
- 점수 clipping (음수/100초과/잘못된 타입)
- Mock client으로 정상 API 응답 파싱
- API 예외 → fallback
- JSON 파싱 실패 → fallback
- SYSTEM_PROMPT 안정성 (단어 풀 변경 없으면 동일 — 캐시 친화)
"""

import json
from dataclasses import dataclass

import pytest

from silent_llm import (
    API_ERROR_COMMENT,
    EvalResult,
    FALLBACK_COMMENT,
    FALLBACK_SCORE_MAX,
    FALLBACK_SCORE_MIN,
    MODEL_ID,
    SYSTEM_PROMPT,
    _build_system_prompt,
    _clip_score,
    _detect_image_media_type,
    _fallback_eval,
    evaluate_expression,
)


# ============================================================
# Mock 안ropic Client (테스트용)
# ============================================================
@dataclass
class _MockBlock:
    type: str
    text: str = ""


@dataclass
class _MockResponse:
    content: list


class _MockMessages:
    def __init__(self, response_text: str = None, raise_exc=None):
        self._response_text = response_text
        self._raise_exc = raise_exc
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return _MockResponse(content=[
            _MockBlock(type="text", text=self._response_text or "")
        ])


class _MockClient:
    def __init__(self, response_text: str = None, raise_exc=None):
        self.messages = _MockMessages(
            response_text=response_text, raise_exc=raise_exc,
        )


def _make_factory(response_text: str = None, raise_exc=None):
    """anthropic.Anthropic 대체 factory — api_key 받아도 무시."""
    def factory(api_key=None):
        return _MockClient(response_text=response_text, raise_exc=raise_exc)
    return factory


# ============================================================
# 1. _clip_score
# ============================================================
class TestClipScore:
    def test_in_range(self):
        assert _clip_score(50) == 50

    def test_above_max(self):
        assert _clip_score(150) == 100

    def test_below_zero(self):
        assert _clip_score(-10) == 0

    def test_float_truncates(self):
        assert _clip_score(75.7) == 75

    def test_string_digit(self):
        assert _clip_score("85") == 85

    def test_invalid_type_returns_zero(self):
        assert _clip_score(None) == 0
        assert _clip_score("not a number") == 0


# ============================================================
# 2. _detect_image_media_type
# ============================================================
class TestDetectMediaType:
    def test_png(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        assert _detect_image_media_type(png) == "image/png"

    def test_jpeg(self):
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 10
        assert _detect_image_media_type(jpeg) == "image/jpeg"

    def test_webp(self):
        # RIFF + 4 bytes size + WEBP
        webp = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 4
        assert _detect_image_media_type(webp) == "image/webp"

    def test_unknown_falls_back_jpeg(self):
        assert _detect_image_media_type(b"random bytes") == "image/jpeg"


# ============================================================
# 3. _fallback_eval
# ============================================================
class TestFallback:
    def test_returns_eval_result(self):
        r = _fallback_eval(seed=0)
        assert isinstance(r, EvalResult)
        assert r.used_api is False

    def test_score_in_fallback_range(self):
        for seed in range(20):
            r = _fallback_eval(seed=seed)
            assert FALLBACK_SCORE_MIN <= r.score <= FALLBACK_SCORE_MAX

    def test_default_comment(self):
        r = _fallback_eval(seed=0)
        assert r.comment == FALLBACK_COMMENT

    def test_custom_comment(self):
        r = _fallback_eval(reason="custom", seed=0)
        assert r.comment == "custom"


# ============================================================
# 4. evaluate_expression — no API key → fallback
# ============================================================
class TestNoApiKey:
    def test_no_env_no_arg_uses_fallback(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        r = evaluate_expression("강아지", b"\xff\xd8fake",
                                 fallback_seed=0)
        assert r.used_api is False
        assert FALLBACK_SCORE_MIN <= r.score <= FALLBACK_SCORE_MAX

    def test_empty_env_uses_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        r = evaluate_expression("강아지", b"\xff\xd8fake",
                                 fallback_seed=0)
        assert r.used_api is False


# ============================================================
# 5. evaluate_expression — Claude API 정상 응답
# ============================================================
class TestApiSuccess:
    def test_parses_json_response(self):
        factory = _make_factory(
            response_text='{"score": 85, "comment": "귀가 잘 보였어요!"}',
        )
        r = evaluate_expression(
            "강아지", b"\x89PNG\r\n\x1a\n" + b"\x00" * 10,
            api_key="test-key", client_factory=factory,
        )
        assert r.used_api is True
        assert r.score == 85
        assert "귀" in r.comment

    def test_clamps_out_of_range_score(self):
        factory = _make_factory(
            response_text='{"score": 150, "comment": "최고!"}',
        )
        r = evaluate_expression(
            "비행기", b"\xff\xd8fake",
            api_key="test-key", client_factory=factory,
        )
        assert r.score == 100
        assert r.used_api is True

    def test_empty_comment_default(self):
        factory = _make_factory(
            response_text='{"score": 50, "comment": ""}',
        )
        r = evaluate_expression(
            "박수", b"\xff\xd8fake",
            api_key="test-key", client_factory=factory,
        )
        assert r.comment == "평가 완료"

    def test_call_kwargs_include_model_and_cache(self):
        """API 호출 시 model + cache_control + system 포함 검증."""
        factory = _make_factory(
            response_text='{"score": 70, "comment": "좋아요"}',
        )
        # factory()를 한 번 호출해서 mock client을 잡아두고
        # 호출 후 그 kwargs를 검증할 수 있도록 closure 사용
        captured = {}

        def capturing_factory(api_key=None):
            client = factory(api_key=api_key)
            captured["client"] = client
            return client

        r = evaluate_expression(
            "노래", b"\xff\xd8fake",
            api_key="test-key", client_factory=capturing_factory,
        )
        assert r.used_api is True
        kwargs = captured["client"].messages.last_call_kwargs
        assert kwargs["model"] == MODEL_ID
        assert kwargs["cache_control"] == {"type": "ephemeral"}
        # output_config 포함 (structured output)
        assert "format" in kwargs["output_config"]
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        # system 프롬프트에 단어 풀의 한글 키워드들 포함
        system = kwargs["system"]
        assert "강아지" in system
        assert "비행기" in system


# ============================================================
# 6. evaluate_expression — 에러 → fallback
# ============================================================
class TestApiErrors:
    def test_api_exception_falls_back(self):
        factory = _make_factory(raise_exc=RuntimeError("network error"))
        r = evaluate_expression(
            "강아지", b"\xff\xd8fake",
            api_key="test-key", client_factory=factory,
            fallback_seed=0,
        )
        assert r.used_api is False
        assert r.comment == API_ERROR_COMMENT
        assert FALLBACK_SCORE_MIN <= r.score <= FALLBACK_SCORE_MAX

    def test_invalid_json_falls_back(self):
        factory = _make_factory(response_text="not valid json")
        r = evaluate_expression(
            "강아지", b"\xff\xd8fake",
            api_key="test-key", client_factory=factory,
            fallback_seed=0,
        )
        assert r.used_api is False
        assert r.comment == API_ERROR_COMMENT

    def test_no_text_block_falls_back(self):
        # 빈 content 응답
        @dataclass
        class _EmptyResp:
            content: list

        class _EmptyMessages:
            def create(self, **kwargs):
                return _EmptyResp(content=[])

        class _EmptyClient:
            def __init__(self):
                self.messages = _EmptyMessages()

        def factory(api_key=None):
            return _EmptyClient()

        r = evaluate_expression(
            "강아지", b"\xff\xd8fake",
            api_key="test-key", client_factory=factory,
            fallback_seed=0,
        )
        assert r.used_api is False


# ============================================================
# 7. SYSTEM_PROMPT 안정성 (캐시 친화)
# ============================================================
class TestSystemPrompt:
    def test_build_is_deterministic(self):
        """같은 단어 풀에서 두 번 빌드 → 동일 문자열 (캐시 가능)."""
        a = _build_system_prompt()
        b = _build_system_prompt()
        assert a == b

    def test_module_level_prompt_matches_build(self):
        assert SYSTEM_PROMPT == _build_system_prompt()

    def test_contains_score_guide(self):
        assert "80~100" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT

    def test_contains_all_word_hints(self):
        """단어 풀의 한글 단어가 모두 system 프롬프트에 포함."""
        from silent_words import WORD_CATALOG
        for w in WORD_CATALOG:
            assert w.ko in SYSTEM_PROMPT
