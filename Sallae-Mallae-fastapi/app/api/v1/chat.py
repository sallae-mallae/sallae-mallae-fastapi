"""
/api/v1/chat — AI 채팅 대화 기록 저장/조회
세션(ChatSession) 하나에 여러 메시지(ChatMessage)가 쌓이는 구조
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import ChatSession, ChatMessage
from app.schemas.analyze import AnalyzeRequest
from app.schemas.chat import (
    CreateSessionRequest,
    AddMessageRequest,
    SessionSummary,
    SessionListResponse,
    SessionDetailResponse,
    MessageItem,
    ChatAnalyzeRequest,
    ChatAnalyzeResponse,
)
from app.services import florence, rag, llm

router = APIRouter(prefix="/chat", tags=["Chat"])


def _format_answer(r) -> str:
    """AI 분석 결과를 채팅 말풍선용 텍스트로 조합"""
    lines = [f"{r.verdict_label}! {r.reason}"]
    if r.pros:
        lines.append(f"\n👍 좋은 점: {r.pros}")
    if r.cons:
        lines.append(f"👎 아쉬운 점: {r.cons}")
    if r.caution:
        lines.append(f"⚠️ {r.caution}")
    if r.recommendation:
        lines.append(f"💡 {r.recommendation}")
    return "\n".join(lines)


@router.post("/analyze", response_model=ChatAnalyzeResponse,
             summary="채팅형 분석 (분석 + 대화 자동 저장 통합)")
async def chat_analyze(body: ChatAnalyzeRequest, db: AsyncSession = Depends(get_db)):
    """
    홈에서 사진+텍스트(또는 텍스트만) 전송 → 세션 확보 → 분석 → 대화 저장.
    - session_id 없으면 새 대화방 생성
    - image_base64 없으면 세션의 마지막 사진 재활용
    - pro_mode True면 Gemini에 이미지 직접 전송
    """
    # 1. 세션 확보
    if body.session_id is None:
        session = ChatSession(title=(body.question or "새 대화")[:50], user_id=body.user_id)
        db.add(session)
        await db.flush()  # session.id 확보
    else:
        session = await db.get(ChatSession, body.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # 2. 이미지 결정 (트리거 사진 or 이전 사진 재활용)
    image_reused = False
    if body.image_base64:
        image_b64 = body.image_base64
        session.last_image_base64 = image_b64   # 세션에 최신 사진 기억
    else:
        image_b64 = session.last_image_base64
        image_reused = True
        if not image_b64:
            raise HTTPException(status_code=422,
                                detail="분석할 사진이 없습니다. 사진을 먼저 보내주세요.")

    # 3. 분석 파이프라인 (Florence-2 → RAG → Gemini)
    try:
        caption = florence.generate_caption(image_b64)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지 캡션 생성 오류: {e}")

    try:
        rag_ctx = await rag.search(caption, category=body.context.category)
    except Exception:
        rag_ctx = ""

    analyze_req = AnalyzeRequest(
        image_base64=image_b64,
        question=body.question,
        context=body.context,
    )
    try:
        result = await llm.judge(caption, rag_ctx, analyze_req, send_image=body.pro_mode)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 판단 오류: {e}")

    # 4. 대화 저장 (사용자 질문 + AI 답변)
    db.add(ChatMessage(session_id=session.id, role="user", content=body.question))
    db.add(ChatMessage(session_id=session.id, role="assistant", content=_format_answer(result)))
    await db.commit()

    detail = await _build_detail(db, session.id)
    return ChatAnalyzeResponse(
        session_id=session.id,
        verdict=result.verdict,
        verdict_label=result.verdict_label,
        product_info=result.product_info,
        reason=result.reason,
        pros=result.pros,
        cons=result.cons,
        caution=result.caution,
        recommendation=result.recommendation,
        pro_mode=body.pro_mode,
        image_reused=image_reused,
        messages=detail.messages,
    )


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
