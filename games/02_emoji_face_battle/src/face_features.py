"""
face_features.py — Face Mesh landmark → 표정 분류용 feature 추출
================================================================

순수 함수 모듈. MediaPipe 의존성 없음 — landmark 객체의 .x, .y만 읽음.
테스트는 간단한 namedtuple/객체로 대체 가능.

Feature는 모두 **scale-invariant 비율**로 정규화 (얼굴 크기/카메라 거리 무관):
- 가로 크기 기준은 양 눈 바깥 끝 사이 거리 (interocular_outer)
- 세로 크기 기준은 눈에서 입까지 거리 (face_vertical)

Author: Stephen (gjkong)
Date: 2026-05-05 (W4 Step 2)
"""

import math


# ============================================================
# 1. MediaPipe Face Mesh 핵심 landmark 인덱스
# ============================================================
# 표준 468 + refined 10 = 478. 본 게임은 기본 468 인덱스만 사용.

# 입
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14

# 왼쪽 눈
LEFT_EYE_OUTER = 33     # 바깥쪽
LEFT_EYE_INNER = 133    # 안쪽 (코 쪽)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

# 오른쪽 눈
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# 눈썹
LEFT_BROW_INNER = 55     # 코 쪽
LEFT_BROW_OUTER = 105    # 바깥쪽 (정수리 쪽)
RIGHT_BROW_INNER = 285
RIGHT_BROW_OUTER = 334

# 코끝
NOSE_TIP = 1


# ============================================================
# 2. 좌표 헬퍼 (landmark 객체 또는 (x, y) 튜플 모두 지원)
# ============================================================
def _xy(landmark):
    """landmark에서 (x, y) 튜플 추출 — .x/.y 속성 또는 [0]/[1] 인덱싱"""
    if hasattr(landmark, "x"):
        return (landmark.x, landmark.y)
    return (landmark[0], landmark[1])


def _dist(a, b) -> float:
    """두 landmark 사이 유클리드 거리 (정규화 좌표 기준)"""
    ax, ay = _xy(a)
    bx, by = _xy(b)
    dx = bx - ax
    dy = by - ay
    return math.hypot(dx, dy)


def _y(landmark) -> float:
    return _xy(landmark)[1]


def _x(landmark) -> float:
    return _xy(landmark)[0]


# ============================================================
# 3. 메인 feature 추출 함수
# ============================================================
def extract_face_features(landmarks) -> dict:
    """Face Mesh landmark 리스트 → 표정 분류용 feature dict

    Args:
        landmarks: MediaPipe NormalizedLandmark 리스트 또는 인덱싱 가능한
                   (x, y) 튜플 리스트. 정규화 좌표(0~1).
    Returns:
        다음 키를 가진 dict (모두 float, scale-invariant):
            mouth_aspect_ratio: 입 가로/세로 (웃음 시 큼, 놀람 시 작음)
            mouth_open: 입 벌어짐 (높이 / interocular)
            mouth_corner_lift: 입꼬리 y - 아래입술 y (음수=위로=웃음)
            eye_open_avg: 양 눈 평균 개방률 (높이/너비)
            brow_height_avg: 눈썹 안쪽 - 눈 위쪽 (높을수록 눈썹 올라감)
            brow_furrow: 양 눈썹 안쪽 사이 거리 / interocular (작을수록 찌푸림)

    Raises:
        IndexError: landmarks가 478(또는 468)개 미만일 때 핵심 인덱스 접근 실패
    """
    # 정규화 기준: 양 눈 바깥 끝 사이 거리
    interocular = _dist(landmarks[LEFT_EYE_OUTER], landmarks[RIGHT_EYE_OUTER])
    if interocular < 1e-6:
        interocular = 1e-6  # 0 division 방지

    # 입 가로/세로
    mouth_w = _dist(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
    mouth_h = _dist(landmarks[UPPER_LIP_CENTER], landmarks[LOWER_LIP_CENTER])
    mouth_aspect_ratio = mouth_w / mouth_h if mouth_h > 1e-6 else 100.0
    mouth_open = mouth_h / interocular

    # 입꼬리 y - 아래입술 y (이미지 좌표: y는 아래로 증가)
    # 음수 = 입꼬리가 더 위 = 웃음, 양수 = 입꼬리가 더 아래 = 슬픔
    corner_avg_y = (_y(landmarks[MOUTH_LEFT]) + _y(landmarks[MOUTH_RIGHT])) / 2.0
    lower_lip_y = _y(landmarks[LOWER_LIP_CENTER])
    mouth_corner_lift = (corner_avg_y - lower_lip_y) / interocular

    # 눈 개방률 (양 눈 평균)
    left_eye_h = _dist(landmarks[LEFT_EYE_TOP], landmarks[LEFT_EYE_BOTTOM])
    left_eye_w = _dist(landmarks[LEFT_EYE_OUTER], landmarks[LEFT_EYE_INNER])
    right_eye_h = _dist(landmarks[RIGHT_EYE_TOP], landmarks[RIGHT_EYE_BOTTOM])
    right_eye_w = _dist(landmarks[RIGHT_EYE_OUTER], landmarks[RIGHT_EYE_INNER])
    left_eye_open = left_eye_h / left_eye_w if left_eye_w > 1e-6 else 0.0
    right_eye_open = right_eye_h / right_eye_w if right_eye_w > 1e-6 else 0.0
    eye_open_avg = (left_eye_open + right_eye_open) / 2.0

    # 눈썹 높이 (눈썹 안쪽 - 눈 위쪽; 음수 = 눈썹이 눈보다 위 = 정상)
    # 표정에 따라 절댓값이 변함. 분류엔 평균 값을 사용 (양수일수록 눈썹 처짐)
    brow_left_dy = (_y(landmarks[LEFT_BROW_INNER]) - _y(landmarks[LEFT_EYE_TOP])) / interocular
    brow_right_dy = (_y(landmarks[RIGHT_BROW_INNER]) - _y(landmarks[RIGHT_EYE_TOP])) / interocular
    brow_height_avg = (brow_left_dy + brow_right_dy) / 2.0

    # 눈썹 안쪽 사이 거리 (찌푸림 = 좁아짐)
    brow_furrow = _dist(landmarks[LEFT_BROW_INNER], landmarks[RIGHT_BROW_INNER]) / interocular

    return {
        "mouth_aspect_ratio": mouth_aspect_ratio,
        "mouth_open": mouth_open,
        "mouth_corner_lift": mouth_corner_lift,
        "eye_open_avg": eye_open_avg,
        "brow_height_avg": brow_height_avg,
        "brow_furrow": brow_furrow,
    }
