"""히스토리 CRUD 서비스"""
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import HistoryRecord
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse, Verdict
from app.schemas.history import HistoryItem, HistoryListResponse, HistoryDetailResponse


VERDICT_LABEL = {"buy": "살래요", "maybe": "고민해요", "no": "말래요"}


async def save_history(
    db: AsyncSession,
    request: AnalyzeRequest,
    response: AnalyzeResponse,
) -> HistoryRecord:
    """분석 결과를 히스토리로 저장"""
    ctx = request.context
    record = HistoryRecord(
        verdict=response.verdict.value,
        question=request.question,
        caption=response.caption,
        reason=response.reason,
        caution=response.caution,
        recommendation=response.recommendation,
        category=ctx.category,
        price=ctx.price,
        purpose=ctx.purpose,
        condition=ctx.condition.value if ctx.condition else None,
        criteria=",".join(ctx.criteria) if ctx.criteria else None,
        image_base64=request.image_base64 if request.save_image else None,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


def _to_item(record: HistoryRecord) -> HistoryItem:
    return HistoryItem(
        id=record.id,
        verdict=Verdict(record.verdict),
        verdict_label=VERDICT_LABEL[record.verdict],
        question=record.question,
        reason=record.reason,
        caution=record.caution,
        recommendation=record.recommendation,
        category=record.category,
        price=record.price,
        purpose=record.purpose,
        condition=record.condition,
        criteria=record.criteria.split(",") if record.criteria else [],
        has_image=record.image_base64 is not None,
        created_at=record.created_at,
    )


async def get_history_list(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 20,
    verdict_filter: str | None = None,
) -> HistoryListResponse:
    query = select(HistoryRecord).order_by(HistoryRecord.created_at.desc())
    count_query = select(func.count()).select_from(HistoryRecord)

    if verdict_filter:
        query = query.where(HistoryRecord.verdict == verdict_filter)
        count_query = count_query.where(HistoryRecord.verdict == verdict_filter)

    total = (await db.execute(count_query)).scalar_one()
    rows = (await db.execute(query.offset(skip).limit(limit))).scalars().all()

    return HistoryListResponse(
        items=[_to_item(r) for r in rows],
        total=total,
    )


async def get_history_detail(db: AsyncSession, record_id: int) -> HistoryDetailResponse | None:
    record = await db.get(HistoryRecord, record_id)
    if not record:
        return None
    base = _to_item(record)
    return HistoryDetailResponse(
        **base.model_dump(),
        image_base64=record.image_base64,
    )


async def delete_history(db: AsyncSession, record_id: int) -> bool:
    record = await db.get(HistoryRecord, record_id)
    if not record:
        return False
    await db.delete(record)
    await db.commit()
    return True
