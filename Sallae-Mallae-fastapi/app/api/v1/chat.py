"""
/api/v1/chat — AI 채팅 대화 기록 저장/조회
세션(ChatSession) 하나에 여러 메시지(ChatMessage)가 쌓이는 구조
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import ChatSession, ChatMessage
from app.schemas.chat import (
    CreateSessionRequest,
    AddMessageRequest,
    SessionSummary,
    SessionListResponse,
    SessionDetailResponse,
    MessageItem,
)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/sessions", response_model=SessionDetailResponse, status_code=201,
             summary="대화 세션 생성 (초기 메시지 포함 가능)")
async def create_session(body: CreateSessionRequest, db: AsyncSession = Depends(get_db)):
    session = ChatSession(title=body.title, user_id=body.user_id)
    db.add(session)
    await db.flush()  # session.id 확보

    for m in body.messages:
        db.add(ChatMessage(session_id=session.id, role=m.role, content=m.content))

    await db.commit()
    return await _build_detail(db, session.id)


@router.post("/sessions/{session_id}/messages", response_model=SessionDetailResponse,
             summary="기존 세션에 메시지 추가")
async def add_messages(session_id: int, body: AddMessageRequest,
                       db: AsyncSession = Depends(get_db)):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    for m in body.messages:
        db.add(ChatMessage(session_id=session_id, role=m.role, content=m.content))

    await db.commit()
    return await _build_detail(db, session_id)


@router.get("/sessions", response_model=SessionListResponse,
            summary="대화 세션 목록")
async def list_sessions(
    user_id: int | None = Query(None, description="특정 사용자 세션만 필터"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    query = select(ChatSession).order_by(ChatSession.created_at.desc())
    count_query = select(func.count()).select_from(ChatSession)
    if user_id is not None:
        query = query.where(ChatSession.user_id == user_id)
        count_query = count_query.where(ChatSession.user_id == user_id)

    total = (await db.execute(count_query)).scalar_one()
    sessions = (await db.execute(query.offset(skip).limit(limit))).scalars().all()

    items = []
    for s in sessions:
        cnt = (await db.execute(
            select(func.count()).select_from(ChatMessage)
            .where(ChatMessage.session_id == s.id)
        )).scalar_one()
        items.append(SessionSummary(
            id=s.id, title=s.title, user_id=s.user_id,
            message_count=cnt, created_at=s.created_at,
        ))
    return SessionListResponse(items=items, total=total)


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse,
            summary="대화 세션 단건 조회 (메시지 전체 포함)")
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return await _build_detail(db, session_id)


@router.delete("/sessions/{session_id}", summary="대화 세션 삭제")
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    # 메시지 먼저 삭제 (SQLite는 ON DELETE CASCADE가 기본 비활성)
    msgs = (await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
    )).scalars().all()
    for m in msgs:
        await db.delete(m)
    await db.delete(session)
    await db.commit()
    return {"deleted": True, "id": session_id}


async def _build_detail(db: AsyncSession, session_id: int) -> SessionDetailResponse:
    session = await db.get(ChatSession, session_id)
    rows = (await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    )).scalars().all()
    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        user_id=session.user_id,
        created_at=session.created_at,
        messages=[MessageItem.model_validate(r) for r in rows],
    )
