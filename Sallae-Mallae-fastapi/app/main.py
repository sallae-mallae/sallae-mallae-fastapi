from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import init_db
from app.services import florence, labeling
from app.api.v1.router import router as v1_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("🚀 살래말래 API 서버 시작")

    await init_db()
    logger.info("✅ DB 초기화 완료")

    # Florence-2 로드 (캡션 생성용, 1회)
    florence.load_model(settings.florence_model_id)

    # YOLO 로드 (라벨링/파인튜닝용, 1회)
    labeling.load_yolo_model(settings.yolo_model)

    yield
    # ── Shutdown ──
    logger.info("🛑 서버 종료")


app = FastAPI(
    title=settings.app_name,
    description="""
## 살래말래? — AI 구매 판단 앱 백엔드

### 처리 흐름
1. Flutter → `POST /api/v1/analyze` (이미지 base64 + 맥락)
2. **Florence-2** (로컬) → 이미지 캡션 생성
3. **Supabase RAG** → 캡션으로 유사 상품/후회 사례 검색
4. **Gemini Flash** → 캡션 + RAG + 맥락 → 잔소리 판단 JSON
5. 결과 반환 + 히스토리 DB 저장

### Swagger 테스트
이미지 base64 변환 없이 바로 테스트하려면 `/api/v1/analyze/test` 사용
    """,
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "test_endpoint": f"{settings.api_v1_prefix}/analyze/test",
        "health": f"{settings.api_v1_prefix}/health",
    }
