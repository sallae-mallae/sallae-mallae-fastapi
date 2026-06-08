"""
/api/v1/analyze
Flutter → 이미지 + 질문 + 맥락 → 살래/고민/말래 판단
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services import florence, llm
from app.services.history_service import save_history

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="상품 구매 판단",
    description="""
카메라로 촬영한 상품 이미지와 맥락 정보를 전송하면 AI가 살래/고민해요/말래요를 판단합니다.

**처리 흐름:**
1. Florence-2 → 이미지 캡션 생성
2. Gemini Flash → 캡션 + 맥락 → 판단 JSON
3. DB 히스토리 저장 (save_image 옵션)
""",
)
async def analyze_product(
    request: AnalyzeRequest,
    ai_model: str | None = Query(None, description="AI 모델 override (설정 화면 연동)"),
    db: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    # 1. Florence-2: 이미지 → 캡션
    try:
        caption = florence.generate_caption(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지 처리 오류: {e}")

    # 2. Gemini: 캡션 + 맥락 → 판단
    try:
        result = await llm.judge(caption=caption, request=request, model=ai_model)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 판단 오류: {e}")

    # 3. 히스토리 저장
    try:
        record = await save_history(db, request, result)
        result.history_id = record.id
    except Exception as e:
        # 저장 실패해도 결과는 반환
        import logging
        logging.getLogger(__name__).error(f"히스토리 저장 실패: {e}")

    return result
