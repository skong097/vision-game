# PlayWait

> AI Vision 기반 매장 대기시간 게임 플랫폼
> YOLO + MediaPipe로 즐기는 식당/카페 미니게임 10종

## 프로젝트 소개

PlayWait는 식당/카페 대기시간을 활용한 **B2B2C AI 게임 플랫폼**입니다.
고객은 게임을 즐기며 대기하고, 승리 시 매장 할인 쿠폰을 받습니다.
매장은 체류시간 증대, 메뉴 노출, 고객 데이터를 확보합니다.

## 게임 라인업 (10종)

| # | 게임명 | 기술 | 난이도 |
|---|--------|------|--------|
| 1 | 카페 닌자 | MediaPipe Hands + YOLO | 보통 |
| 2 | 표정 미러링 챌린지 | MediaPipe Face Mesh | 보통 |
| 3 | AR 보물찾기 | YOLO + AR | 매우 어려움 |
| 4 | K-Pop 랜덤 댄스 | MediaPipe Pose | 어려움 |
| 5 | 고요 속의 외침 | Pose + Hands + LLM | 매우 어려움 |
| 6 | 가위바위보 진화 | MediaPipe Hands | 매우 쉬움 |
| 7 | 스피드 카운터 | MediaPipe Hands | 쉬움 |
| 8 | 좀비 피하기 (할로윈) | Pose + YOLO | 어려움 |
| 9 | 산타 선물 받기 (X-mas) | MediaPipe Hands | 보통 |
| 10 | 커플 싱크 (발렌타인) | Face + Pose 멀티 | 어려움 |

## 디렉토리 구조

```
PlayWait/
├── core/              # 공통 코어 (모든 게임 공유)
│   ├── vision/        # MediaPipe, YOLO 엔진 래퍼
│   ├── game_base/     # BaseGame 추상 클래스, 점수, 보상
│   ├── ui/            # 오버레이, 이펙트
│   └── platform/      # 매장 API, 쿠폰, 분석
├── games/             # 게임별 모듈 (10종)
├── dashboard/         # 매장 사장님용 웹 대시보드
├── docs/              # 문서 (일별 로그, 설계, API)
├── tests/             # 통합 테스트
├── scripts/           # 유틸 스크립트
└── assets/            # 공통 리소스 (이미지, 사운드, 모델)
```

## 빠른 시작

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 첫 게임 실행 (W1: 가위바위보 진화)
python -m games.06_rps_evolution.src.game
```

## 개발 로드맵

10주 계획. 매주 1게임씩 개발.
자세한 일정은 `docs/design/roadmap.md` 참고.

## 기술 스택

- **Python 3.12**
- **MediaPipe** (Hands, Pose, Face Mesh)
- **YOLO v8/v11** (ultralytics)
- **FastAPI** (백엔드 API)
- **OpenCV** (영상 처리)
- **PWA** (프론트엔드)

## 개발자

- **Stephen (gjkong)** - PinkLAB 로봇 엔지니어
- 1인 개발자 (Solo Developer)

## 라이선스

MIT License — 자세한 내용은 [LICENSE](LICENSE) 파일 참고.
