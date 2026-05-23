# PlayWait 디렉토리 구조

> **생성일**: 2026-04-30
> **총 디렉토리**: 60개
> **총 파일**: 61개

## 전체 트리

```
.
|-- assets
|   |-- images
|   |-- models
|   `-- sounds
|-- core
|   |-- game_base
|   |   |-- __init__.py
|   |   |-- base_game.py
|   |   |-- reward_system.py
|   |   `-- score_manager.py
|   |-- platform
|   |   |-- __init__.py
|   |   |-- analytics.py
|   |   |-- coupon_qr.py
|   |   `-- store_api.py
|   |-- ui
|   |   |-- __init__.py
|   |   |-- effects.py
|   |   `-- overlay_renderer.py
|   |-- vision
|   |   |-- __init__.py
|   |   |-- mediapipe_engine.py
|   |   `-- yolo_engine.py
|   `-- __init__.py
|-- dashboard
|   |-- backend
|   `-- frontend
|-- docs
|   |-- api
|   |-- daily_logs
|   `-- design
|       `-- directory_structure.md
|-- games
|   |-- 01_cafe_ninja
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   |-- 02_emoji_face_battle
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   |-- 03_hidden_menu_hunt
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   |-- 04_kpop_dance
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   |-- 05_silent_charades
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   |-- 06_rps_evolution
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   |-- 07_speed_counter
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   |-- 08_zombie_dodge
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   |-- 09_santa_catch
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   |-- 10_couple_sync
|   |   |-- assets
|   |   |-- src
|   |   |   |-- __init__.py
|   |   |   `-- game.py
|   |   |-- tests
|   |   |   `-- __init__.py
|   |   `-- README.md
|   `-- __init__.py
|-- scripts
|-- tests
|   `-- __init__.py
|-- .env.example
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## 주요 디렉토리 설명

### `core/` - 공통 코어 모듈
모든 게임이 공유하는 핵심 라이브러리. **W0 (Week 0)** 에 우선 구축.

- `vision/` - MediaPipe, YOLO 엔진 통합 래퍼
- `game_base/` - BaseGame 추상 클래스, 점수/타이머/보상 매니저
- `ui/` - 공통 오버레이 렌더링, 파티클/사운드 이펙트
- `platform/` - 매장 API 연동, 쿠폰 QR 발급, 데이터 분석

### `games/` - 게임별 모듈 (10종)
각 게임은 `BaseGame` 클래스를 상속받아 독립 모듈로 구현.

| 폴더 | 게임명 | 개발 주차 |
|------|--------|----------|
| 01_cafe_ninja | 카페 닌자 | W3 |
| 02_emoji_face_battle | 표정 미러링 챌린지 | W4 |
| 03_hidden_menu_hunt | AR 보물찾기 | W10 |
| 04_kpop_dance | K-Pop 랜덤 댄스 | W6 |
| 05_silent_charades | 고요 속의 외침 | W9 |
| 06_rps_evolution | 가위바위보 진화 | **W1 (시작)** |
| 07_speed_counter | 스피드 카운터 | W2 |
| 08_zombie_dodge | 좀비 피하기 | W7 |
| 09_santa_catch | 산타 선물 받기 | W5 |
| 10_couple_sync | 커플 싱크 | W8 |

각 게임 폴더 구조:
```
{game_name}/
├── src/           # 게임 로직 소스 코드
│   ├── __init__.py
│   └── game.py    # BaseGame 상속 클래스
├── assets/        # 게임 전용 리소스 (이미지, 사운드)
├── tests/         # 게임 단위 테스트
└── README.md      # 게임 기획서 + 사용법
```

### `dashboard/` - 매장 사장님용 웹 대시보드
- `backend/` - FastAPI 기반 관리 API
- `frontend/` - 매장 설정 UI (메뉴 등록, 보상 난이도, 통계)

### `docs/` - 문서
- `daily_logs/` - 일별 개발 로그 (.md)
- `design/` - 설계 문서 (아키텍처, 게임 기획서)
- `api/` - API 명세서

### `assets/` - 공통 리소스
- `images/` - 공통 UI 이미지
- `sounds/` - 공통 사운드 이펙트
- `models/` - 학습된 AI 모델 파일 (.pt, .onnx) — Git LFS 권장

### `tests/` - 통합 테스트
여러 게임/모듈을 가로지르는 통합 테스트.

### `scripts/` - 유틸 스크립트
데이터 마이그레이션, 모델 변환, 배포 스크립트 등.

## 최상위 파일

| 파일 | 용도 |
|------|------|
| `README.md` | 프로젝트 소개 |
| `requirements.txt` | Python 의존성 |
| `.gitignore` | Git 제외 파일 |
| `.env.example` | 환경 변수 템플릿 |

---

## 개발 진행 정책

- 게임은 **한 번에 하나씩** 개발 진행 (W1부터 W10까지 순차적)
- 게임별 `src/game.py` 와 `README.md` 는 **해당 주차 시작 시** 채움
- 매일 작업 종료 시 `docs/daily_logs/YYYY-MM-DD.md` 작성

**다음 작업**: W1 가위바위보 진화 (`games/06_rps_evolution/`) 개발 시작
