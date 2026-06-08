"""
/api/v1/history
화면 13 - 구매 판단 기록 CRUD
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.history import HistoryListResponse, HistoryDetailResponse
from app.services.history_service import (
    get_history_list,
    get_history_detail,
    delete_history,
)

router = APIRouter()


@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="구매 판단 기록 목록",
    description="살래(buy) / 고민(maybe) / 말래(no) 필터링 지원. 최신순 정렬.",
)
async def list_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    verdict: str | None = Query(None, description="buy | maybe | no"),
    db: AsyncSession = Depends(get_db),
) -> HistoryListResponse:
    return await get_history_list(db, skip=skip, limit=limit, verdict_filter=verdict)


@router.get(
    "/history/{record_id}",
    response_model=HistoryDetailResponse,
    summary="구매 판단 기록 단건 조회",
    description="저장된 이미지(image_base64)를 포함하여 반환합니다.",
)
async def get_history(
    record_id: int,
    db: AsyncSession = Depends(get_db),
) -> HistoryDetailResponse:
    record = await get_history_detail(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")
    return record


@router.delete(
    "/history/{record_id}",
    summary="구매 판단 기록 삭제",
)
async def remove_history(
    record_id: int,
    db: AsyncSession = Depends(get_db),
):
    deleted = await delete_history(db, record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")
    return {"deleted": True, "id": record_id}
