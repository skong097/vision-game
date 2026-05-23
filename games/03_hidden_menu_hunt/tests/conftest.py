"""
W10 단위 테스트 conftest
==========================

폴더명 숫자 시작(`03_hidden_menu_hunt`)이라 일반 import 불가.
src/ 경로 sys.path 주입.
프로젝트 루트도 sys.path에 추가 — `core.vision.yolo_engine` 임포트용
(W5의 yolo_engine 재활용).

NOTE: tests/__init__.py 만들지 않음 (W3 트러블 #12).
"""

import os
import sys

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
for p in (SRC_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
