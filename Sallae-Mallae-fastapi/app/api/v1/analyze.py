"""
/api/v1/analyze       — Flutter 연동 (base64 JSON)
/api/v1/analyze/test  — Swagger 테스트용 (파일 업로드, base64 불필요)
"""
import base64

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse, ContextInput
from app.services import florence, rag
from app.services import llm
from app.services.history_service import save_history

router = APIRouter()


async def _run_pipeline(
    request: AnalyzeRequest,
    ai_model: str | None,
    db: AsyncSession,
) -> AnalyzeResponse:
    """
    공통 파이프라인
    1. Florence-2  → 이미지 캡션
    2. Supabase RAG → 관련 상품 정보/후회 사례
    3. Gemini      → 캡션 + RAG + 맥락 → 판단
    4. DB 히스토리 저장
    """
    import logging
    logger = logging.getLogger(__name__)

    # 1. 이미지 유효성 검사 (base64 디코딩 가능 여부)
    try:
        base64.b64decode(request.image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=422, detail="image_base64가 유효한 base64 문자열이 아닙니다.")

    # 2. Florence-2: 이미지 → 캡션
    try:
        caption = florence.generate_caption(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지 캡션 생성 오류: {e}")

    # 3. Supabase RAG: 캡션 → 관련 정보 검색
    try:
        rag_context = await rag.search(caption, category=request.context.category)
    except Exception as e:
        logger.warning(f"RAG 검색 실패 (무시하고 진행): {e}")
        rag_context = ""

    # 4. Gemini: 캡션 + RAG + 맥락 → 판단
    try:
        result = await llm.judge(
            caption=caption,
            rag_context=rag_context,
            request=request,
            model=ai_model,
        )
        result.rag_used = bool(rag_context)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 판단 오류: {e}")

    # 5. 히스토리 저장
    try:
        record = await save_history(db, request, result)
        result.history_id = record.id
    except Exception as e:
        logger.error(f"히스토리 저장 실패: {e}")

    return result


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="상품 구매 판단 (Flutter 연동)",
    description="""
**Flutter 앱 전용** — image_base64를 JSON 바디로 전송합니다.

처리 흐름:
1. **Florence-2** → 이미지 캡션 생성
2. **Supabase RAG** → 캡션으로 유사 상품 정보/후회 사례 검색
3. **Gemini Flash** → 캡션 + RAG 족보 + 맥락 → 잔소리 판단 JSON

> Swagger에서 직접 테스트하려면 `/api/v1/analyze/test` 를 사용하세요 (파일 업로드 방식).
""",
)
async def analyze_product(
    request: AnalyzeRequest,
    ai_model: str | None = Query(None, description="Gemini 모델 override"),
    db: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    return await _run_pipeline(request, ai_model, db)


@router.post(
    "/analyze/test",
    response_model=AnalyzeResponse,
    summary="상품 구매 판단 — Swagger 테스트용 (파일 업로드)",
    description="""
**Swagger UI / Postman 테스트 전용** — 상품 이미지 파일만 업로드하면
사진만으로 살래/고민/말래 판단을 반환합니다.
""",
)
async def analyze_product_test(
    image: UploadFile = File(..., description="상품 이미지 파일 (JPEG)"),
    db: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    raw = await image.read()
    image_base64 = base64.b64encode(raw).decode()

    request = AnalyzeRequest(
        image_base64=image_base64,
        context=ContextInput(),
        save_image=False,
    )
    return await _run_pipeline(request, None, db)
