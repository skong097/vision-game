"""
color_classifier.py — 세련된 8색 팔레트 분류 + 매칭 점수
==========================================================

객체 ROI의 HSV 대표값을 입력으로 받아:
- classify_color(): 가장 가까운 색 + 신뢰도
- color_match_score(): 특정 색 대비 매칭도 0~1

순수 함수 모듈. OpenCV/YOLO/카메라 의존성 없음 → 풍부한 단위 테스트.

설계 방식
---------
- 각 색에 **profile** = {h_range, s_range, v_range} (3축 동시 제약)
- 한 축이라도 범위 밖이면 거리 비례 선형 감소 → 한 축이라도 크게 벗어나면
  전체 점수 떨어짐 (집계는 **min**)
- min 집계 이유: "세련된 색"은 단일 hue로 정의되지 않음. 예) 세이지는
  녹색 hue + **낮은 채도**가 동시에 충족되어야 사용자가 "세이지"로 인식.
  채도가 0.7인 녹색은 그냥 비비드 그린이지 세이지가 아님.

HSV convention
--------------
- H: [0, 360) — 표준 색상환 (OpenCV 0-179이 아닌 정규화 값)
- S: [0, 1]
- V: [0, 1]

호출측이 OpenCV BGR→HSV 변환 후 (H*2, S/255, V/255)로 정규화해서 전달.

Author: Stephen (gjkong)
Date: 2026-05-11 (W5 Step 2)
"""


# ============================================================
# 1. 색 상수 — 세련된 8색 팔레트
# ============================================================
COLOR_BURGUNDY = "burgundy"      # 깊은 와인 레드
COLOR_TERRACOTTA = "terracotta"  # 흙빛 주황-적
COLOR_MUSTARD = "mustard"        # 머스타드 옐로우
COLOR_SAGE = "sage"              # 차분한 회녹
COLOR_NAVY = "navy"              # 깊은 푸른색
COLOR_MAUVE = "mauve"            # 부드러운 자주-핑크
COLOR_CREAM = "cream"            # 따뜻한 오프화이트
COLOR_CHARCOAL = "charcoal"      # 부드러운 검정
COLOR_NEUTRAL = "neutral"        # 매칭 안 됨 (분류 결과로만)

ALL_COLORS = (
    COLOR_BURGUNDY, COLOR_TERRACOTTA, COLOR_MUSTARD, COLOR_SAGE,
    COLOR_NAVY, COLOR_MAUVE, COLOR_CREAM, COLOR_CHARCOAL,
)
MISSION_COLORS = ALL_COLORS  # 게임 미션으로 출제 가능한 색


# ============================================================
# 2. 색 profile — HSV 3축 범위
# ============================================================
# h_range: list of (low, high) sub-ranges (각 sub-range 내에서 low <= high).
#          색상환을 가로지르는 경우 두 sub-range로 분할 (예: 버건디).
#          None이면 색상 무관 (차콜처럼 무채색 계열).
# s_range, v_range: (low, high) — 한 구간.
COLOR_PROFILES = {
    # 와인레드: red hue, 진하고 깊은 톤 (V 낮음 + S 적당)
    COLOR_BURGUNDY: {
        "h_range": [(340, 360), (0, 10)],
        "s_range": (0.45, 0.85),
        "v_range": (0.20, 0.50),
    },
    # 테라코타: 흙빛 — 오렌지 hue + 중간 V + 중간 S (산뜻한 오렌지와 차별)
    COLOR_TERRACOTTA: {
        "h_range": [(8, 25)],
        "s_range": (0.40, 0.75),
        "v_range": (0.35, 0.65),
    },
    # 머스타드: 노랑-주황 사이 + 충분히 S — 그러나 V는 lemon보다 낮게
    COLOR_MUSTARD: {
        "h_range": [(35, 55)],
        "s_range": (0.55, 0.90),
        "v_range": (0.45, 0.75),
    },
    # 세이지: 녹색 hue + **낮은 S** (이게 핵심 — 비비드 그린과 분리)
    COLOR_SAGE: {
        "h_range": [(75, 135)],
        "s_range": (0.15, 0.40),
        "v_range": (0.45, 0.75),
    },
    # 네이비: 파란 hue + 매우 낮은 V (밝은 파랑과 분리)
    COLOR_NAVY: {
        "h_range": [(210, 235)],
        "s_range": (0.45, 0.85),
        "v_range": (0.15, 0.40),
    },
    # 모브: 핑크-퍼플 사이 + **낮은 S** (핫핑크와 분리)
    COLOR_MAUVE: {
        "h_range": [(295, 340)],
        "s_range": (0.15, 0.40),
        "v_range": (0.50, 0.75),
    },
    # 크림: 따뜻한 hue 힌트 + 낮은 S + **매우 높은 V** (순백과 차별)
    COLOR_CREAM: {
        "h_range": [(20, 60)],
        "s_range": (0.05, 0.25),
        "v_range": (0.82, 0.97),
    },
    # 차콜: 색상 무관 + 낮은 S + 낮은 V (순검정보다 살짝 들뜬 어둠)
    COLOR_CHARCOAL: {
        "h_range": None,
        "s_range": (0.00, 0.30),
        "v_range": (0.10, 0.28),
    },
}


# ============================================================
# 3. fall-off 폭 & 신뢰 임계
# ============================================================
# 한 축이 범위 밖이면 이 거리만큼 멀어졌을 때 score 0.
# 작을수록 strict, 클수록 관대. 매장 환경 캘리브레이션 후 튜닝 예정.
HUE_FALLOFF = 15.0   # degrees
SAT_FALLOFF = 0.12
VAL_FALLOFF = 0.12

# 어느 색에도 0.60 미만이면 NEUTRAL — 모호한 색 강제 분류 방지
NEUTRAL_THRESHOLD = 0.60


# ============================================================
# 4. 축별 점수 — 범위 안 1.0, 밖이면 거리 비례 선형 감소
# ============================================================
def _hue_score(h: float, ranges) -> float:
    """순환 hue 공간에서 sub-range 중 가장 가까운 곳까지의 거리로 점수.

    ranges=None이면 색상 제약 없음 → 1.0 (차콜용).
    """
    if ranges is None:
        return 1.0
    h = h % 360.0
    min_d = 360.0
    for (low, high) in ranges:
        if low <= h <= high:
            return 1.0
        # 양 끝점까지의 순환 거리 (0/360 경계 처리)
        d_low = min(abs(h - low), 360.0 - abs(h - low))
        d_high = min(abs(h - high), 360.0 - abs(h - high))
        d = min(d_low, d_high)
        if d < min_d:
            min_d = d
    return max(0.0, 1.0 - min_d / HUE_FALLOFF)


def _band_score(value: float, low: float, high: float, falloff: float) -> float:
    """value가 [low, high]에 있으면 1.0, 바깥이면 falloff 단위로 선형 감소."""
    if low <= value <= high:
        return 1.0
    d = (low - value) if value < low else (value - high)
    return max(0.0, 1.0 - d / falloff)


# ============================================================
# 5. 매칭 점수 (target 색 대비)
# ============================================================
def color_match_score(target: str, hsv: tuple) -> float:
    """target 색의 profile에 hsv가 얼마나 부합하는지 0~1.

    집계: **min(축별 score)** — 한 축이라도 크게 벗어나면 0에 수렴.
    "세련된 색"은 hue·sat·val 모두 동시 충족되어야 의미를 가짐.

    Args:
        target: ALL_COLORS 중 하나
        hsv: (H, S, V), H in [0, 360), S/V in [0, 1]
    Returns:
        0.0 ~ 1.0
    Raises:
        ValueError: 알 수 없는 target
    """
    if target not in COLOR_PROFILES:
        raise ValueError(f"알 수 없는 색: {target}")
    h, s, v = hsv
    p = COLOR_PROFILES[target]
    scores = (
        _hue_score(h, p["h_range"]),
        _band_score(s, p["s_range"][0], p["s_range"][1], SAT_FALLOFF),
        _band_score(v, p["v_range"][0], p["v_range"][1], VAL_FALLOFF),
    )
    return min(scores)


# ============================================================
# 6. 분류 (모든 색 중 가장 매칭되는 것)
# ============================================================
def classify_color(hsv: tuple) -> tuple:
    """hsv를 가장 잘 설명하는 색 + 신뢰도 반환.

    Returns:
        (color, confidence): color는 ALL_COLORS 중 하나 또는 COLOR_NEUTRAL
                              (모두 임계 미만 시), confidence는 0~1.
    """
    best_color = COLOR_NEUTRAL
    best_score = 0.0
    for color in ALL_COLORS:
        s = color_match_score(color, hsv)
        if s > best_score:
            best_score = s
            best_color = color

    if best_score < NEUTRAL_THRESHOLD:
        return (COLOR_NEUTRAL, best_score)
    return (best_color, best_score)
