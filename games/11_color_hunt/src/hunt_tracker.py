"""
hunt_tracker.py — 라운드 진행 상태 (수집 개수 + 남은 시간 + dedup)
========================================================================

W5 컬러 헌트의 한 라운드(1 mission) 상태.

핵심
----
- 같은 객체가 연속 프레임에서 잡히면 카운트 1번만 → **dedup window** 적용
- 객체 식별은 (yolo_class, bbox center 근접) 휴리스틱 사용. YOLO 트랙 ID가 있다면
  더 정확하지만 ultralytics 기본 predict는 트랙 미포함 → 위치 기반 dedup.
- 신뢰도 게이트(`min_color_confidence`) 미만은 카운트 X
- 종료 조건:
    1) collected ≥ goal → WIN (즉시)
    2) time_remaining ≤ 0 → TIMEOUT
- 시간 의존부는 `time_provider` 콜백으로 주입 가능 → 단위 테스트 시간 제어 OK

순수 클래스. cv2/YOLO/카메라 의존성 X.

Author: Stephen (gjkong)
Date: 2026-05-12 (W5 Step 3)
"""

import time

try:
    from .color_classifier import COLOR_NEUTRAL
except ImportError:
    from color_classifier import COLOR_NEUTRAL


# ============================================================
# 1. 종료 사유
# ============================================================
END_NONE = "none"
END_WIN = "win"
END_TIMEOUT = "timeout"


# ============================================================
# 2. 기본 상수
# ============================================================
# 같은 yolo_class + 동일 영역에서 5초 이내 재검출 = 중복 카운트 무시
DEFAULT_DEDUP_WINDOW = 5.0      # 초
# 두 객체의 box center가 이 픽셀 이내면 "같은 객체"로 본다
DEFAULT_PROXIMITY_PX = 80


# ============================================================
# 3. ScoreTracker
# ============================================================
class HuntTracker:
    """1 라운드 분량 상태.

    Attributes:
        mission: mission_generator.Mission
        collected: 매칭 카운트
        attempts: 모든 detection 시도 수 (debug/통계)
        rejected_by_color: 색 매칭 실패로 거부된 객체 수
        rejected_by_confidence: 게이트 미달
        rejected_by_dedup: dedup으로 거부
        start_time: 라운드 시작 epoch (start() 호출 시점)
        _end_reason: cache
        _recent_matches: dedup 윈도우용 리스트 [(class_name, cx, cy, t)]
    """

    def __init__(
        self,
        mission,
        dedup_window: float = DEFAULT_DEDUP_WINDOW,
        proximity_px: int = DEFAULT_PROXIMITY_PX,
        time_provider=None,
    ):
        """
        Args:
            mission: 적용할 Mission (target_color/goal_count/time_limit/confidence_gate)
            dedup_window: 같은 객체 재검출 차단 시간(초)
            proximity_px: 같은 객체로 보는 박스 중심 거리 임계
            time_provider: () -> float (epoch 시간). None이면 time.time. 테스트용 주입.
        """
        self.mission = mission
        self.dedup_window = dedup_window
        self.proximity_px = proximity_px
        self._now = time_provider if time_provider is not None else time.time
        self.reset()

    # ----------------------------------------
    # 라이프사이클
    # ----------------------------------------
    def reset(self):
        self.collected = 0
        self.attempts = 0
        self.rejected_by_color = 0
        self.rejected_by_confidence = 0
        self.rejected_by_dedup = 0
        self.start_time = None
        self._end_reason = END_NONE
        self._recent_matches = []  # [(class_name, cx, cy, t)]

    def start(self):
        self.start_time = self._now()

    # ----------------------------------------
    # 시간
    # ----------------------------------------
    def get_elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return self._now() - self.start_time

    def get_remaining(self) -> float:
        return max(0.0, self.mission.time_limit - self.get_elapsed())

    # ----------------------------------------
    # 객체 적용 — pipeline.ObjectMatch 리스트 처리
    # ----------------------------------------
    def apply_matches(self, matches) -> int:
        """이 프레임의 분석 결과를 상태에 반영.

        Args:
            matches: List[ObjectMatch] (object_pipeline.analyze_frame 결과)

        Returns:
            이 호출로 새로 카운트된 객체 수 (0 ~ len(matches))
        """
        if self.start_time is None:
            return 0
        if self._end_reason != END_NONE:
            return 0

        now_t = self._now()
        gained = 0

        for m in matches:
            self.attempts += 1
            if not m.is_target_match:
                self.rejected_by_color += 1
                continue
            if m.color_confidence < self.mission.confidence_gate:
                self.rejected_by_confidence += 1
                continue
            cx, cy = _bbox_center(m.bbox)
            if self._is_duplicate(m.yolo_class, cx, cy, now_t):
                self.rejected_by_dedup += 1
                continue
            # 카운트
            self.collected += 1
            gained += 1
            self._recent_matches.append((m.yolo_class, cx, cy, now_t))

        # 만료된 dedup 항목 정리 (성능: O(N) but recent list는 작음)
        cutoff = now_t - self.dedup_window
        self._recent_matches = [
            r for r in self._recent_matches if r[3] >= cutoff
        ]
        return gained

    def _is_duplicate(self, class_name: str, cx: int, cy: int,
                      now_t: float) -> bool:
        """최근 dedup_window 내에 같은 class·근접 위치 객체가 있었는가?"""
        cutoff = now_t - self.dedup_window
        for (cls, rx, ry, t) in self._recent_matches:
            if t < cutoff:
                continue
            if cls != class_name:
                continue
            dx, dy = (cx - rx), (cy - ry)
            if (dx * dx + dy * dy) ** 0.5 <= self.proximity_px:
                return True
        return False

    # ----------------------------------------
    # 종료 판정
    # ----------------------------------------
    def check_end(self) -> str:
        """게임 종료 사유.

        우선순위: WIN (즉시) > TIMEOUT
        """
        if self._end_reason != END_NONE:
            return self._end_reason

        if self.collected >= self.mission.goal_count:
            self._end_reason = END_WIN
            return END_WIN

        if self.start_time is not None and self.get_remaining() <= 0:
            self._end_reason = END_TIMEOUT
            return END_TIMEOUT

        return END_NONE

    def is_game_over(self) -> bool:
        return self.check_end() != END_NONE

    def is_win(self) -> bool:
        return self.check_end() == END_WIN

    # ----------------------------------------
    # UI / 로깅
    # ----------------------------------------
    def get_summary(self) -> dict:
        return {
            "target_color": self.mission.target_color,
            "goal_count": self.mission.goal_count,
            "collected": self.collected,
            "progress": (self.collected / self.mission.goal_count)
                        if self.mission.goal_count else 0.0,
            "time_limit": self.mission.time_limit,
            "elapsed": self.get_elapsed(),
            "remaining": self.get_remaining(),
            "end_reason": self.check_end(),
            "attempts": self.attempts,
            "rejected_by_color": self.rejected_by_color,
            "rejected_by_confidence": self.rejected_by_confidence,
            "rejected_by_dedup": self.rejected_by_dedup,
        }

    def __repr__(self):
        return (
            f"HuntTracker(collected={self.collected}/"
            f"{self.mission.goal_count}, "
            f"remaining={self.get_remaining():.1f}s, "
            f"end={self.check_end()})"
        )


# ============================================================
# 4. 유틸
# ============================================================
def _bbox_center(bbox) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)
