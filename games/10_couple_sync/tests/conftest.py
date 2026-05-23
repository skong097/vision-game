"""
W8 단위 테스트 conftest
=========================

폴더명 숫자 시작(`10_couple_sync`)이라 일반 import 불가.
src/ 경로 sys.path 주입. W2~W7 패턴 동일.

NOTE: tests/__init__.py 만들지 않음 (W3 트러블 #12).
"""

import os
import sys

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
