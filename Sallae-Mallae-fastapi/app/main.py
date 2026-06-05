from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import init_db
from app.services import florence
from app.api.v1.router import router as v1_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 이벤트"""
    # ── Startup ──
    logger.info("🚀 살래말래 API 서버 시작")

    # DB 테이블 생성
    await init_db()
    logger.info("✅ DB 초기화 완료")

    # Florence-2 모델 로드 (1회)
    florence.load_model(settings.florence_model_id)

    yield

    # ── Shutdown ──
    logger.info("🛑 서버 종료")


app = FastAPI(
    title=settings.app_name,
    description="""
## 살래말래? — AI 구매 판단 앱 백엔드

카메라로 상품을 촬영하고 맥락을 입력하면 AI가 **살래요 / 고민해요 / 말래요** 판단을 내려줍니다.

### 처리 흐름
1. Flutter 앱 → `POST /api/v1/analyze` (이미지 base64 + 맥락)
2. **Florence-2** (로컬) → 상품 캡션 생성
3. **Gemini Flash** → 캡션 + 맥락 → 판단 JSON
4. 결과 반환 + 히스토리 DB 저장
    """,
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# Flutter 앱에서 호출 가능하도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 Flutter 앱 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": settings.app_name,
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
    }
