"""
W4 단위 테스트 conftest
=========================

폴더명이 숫자로 시작(`02_emoji_face_battle`)해서 일반 패키지 import 불가.
src/ 경로를 sys.path에 주입해 `from face_features import ...` 가능하게 함.
W2·W3와 동일 패턴.

NOTE: tests/__init__.py는 일부러 만들지 않음 (W3 트러블 #12 학습).
"""

import os
import sys

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
