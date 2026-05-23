"""
pose_features.py — MediaPipe Pose landmark → 분류용 feature 추출
=================================================================

순수 함수 모듈. MediaPipe 의존성 없음 — landmark 객체의 .x, .y만 읽음.
테스트는 namedtuple/객체로 대체 가능.

Feature는 모두 **scale-invariant 비율** — 어깨 너비로 정규화.
y는 OpenCV/MediaPipe normalized coord (위→0, 아래→1) 기준.

핵심 feature
------------
- 양 팔 각도 (어깨-팔꿈치-손목, deg)
- 양 손 어깨 대비 y 오프셋 (음수=위)
- 양 손 사이 거리 (어깨 너비 단위)
- 양 손 평균 높이 (어깨 대비)

Author: Stephen (gjkong)
Date: 2026-05-12 (W6 Step 2)
"""

import math


# ============================================================
# 1. MediaPipe Pose 핵심 landmark 인덱스 (33 landmarks)
# ============================================================
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


# ============================================================
# 2. 좌표 헬퍼
# ============================================================
def _xy(landmark):
    """landmark에서 (x, y) 추출 — .x/.y 또는 [0]/[1]."""
    if hasattr(landmark, "x"):
        return (landmark.x, landmark.y)
    return (landmark[0], landmark[1])


def _dist(a, b) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.hypot(bx - ax, by - ay)


def _angle_deg(a, b, c) -> float:
    """세 점이 이루는 각도 (b가 꼭짓점), 도 단위.

    벡터 BA, BC의 내적으로 계산. 두 벡터 중 하나라도 0이면 0 반환.
    """
    ax, ay = _xy(a)
    bx, by = _xy(b)
    cx, cy = _xy(c)
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_t = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


# ============================================================
# 3. 메인 feature 추출
# ============================================================
def extract_pose_features(landmarks) -> dict:
    """Pose 33 landmark → 분류용 feature dict

    Args:
        landmarks: MediaPipe NormalizedLandmark 리스트 또는 (x, y) 튜플 리스트.
                   정규화 좌표 (0~1).

    Returns:
        다음 키를 가진 dict (모두 float):
            left_arm_angle:   왼팔 각도 (어깨-팔꿈치-손목), 0~180 deg
            right_arm_angle:  오른팔 각도
            left_hand_y_off:  (왼손 y - 왼 어깨 y) / 어깨너비
                              음수 = 손이 어깨보다 위
            right_hand_y_off: 오른쪽
            hand_distance:    양 손 사이 거리 / 어깨너비
            hand_y_avg:       양 손 평균 y_off (음수=양손 모두 위)

    Raises:
        IndexError: landmarks가 33개 미만일 때 핵심 인덱스 접근 실패
    """
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    le = landmarks[LEFT_ELBOW]
    re = landmarks[RIGHT_ELBOW]
    lw = landmarks[LEFT_WRIST]
    rw = landmarks[RIGHT_WRIST]

    shoulder_width = _dist(ls, rs)
    if shoulder_width < 1e-6:
        shoulder_width = 1e-6  # 0 division 방지

    left_arm_angle = _angle_deg(ls, le, lw)
    right_arm_angle = _angle_deg(rs, re, rw)

    ls_y = _xy(ls)[1]
    rs_y = _xy(rs)[1]
    lw_y = _xy(lw)[1]
    rw_y = _xy(rw)[1]

    left_hand_y_off = (lw_y - ls_y) / shoulder_width
    right_hand_y_off = (rw_y - rs_y) / shoulder_width

    hand_distance = _dist(lw, rw) / shoulder_width
    hand_y_avg = (left_hand_y_off + right_hand_y_off) / 2.0

    return {
        "left_arm_angle":   left_arm_angle,
        "right_arm_angle":  right_arm_angle,
        "left_hand_y_off":  left_hand_y_off,
        "right_hand_y_off": right_hand_y_off,
        "hand_distance":    hand_distance,
        "hand_y_avg":       hand_y_avg,
    }
