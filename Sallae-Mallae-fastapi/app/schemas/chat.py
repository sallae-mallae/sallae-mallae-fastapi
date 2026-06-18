"""AI 채팅 대화 기록 스키마"""
from datetime import datetime
from pydantic import BaseModel, Field

from app.schemas.analyze import ContextInput, Verdict


class MessageInput(BaseModel):
    role: str = Field(..., description="user | assistant", examples=["user"])
    content: str = Field(..., examples=["이 신발 살 만한가요?"])


class AnalysisData(BaseModel):
    """assistant 메시지에 담기는 구조화 분석 결과 (프론트가 파싱 없이 그대로 사용)"""
    verdict: str
    verdict_label: str
    product_info: str | None = None
    reason: str
    pros: str | None = None
    cons: str | None = None
    caution: str | None = None
    recommendation: str | None = None


class MessageItem(BaseModel):
    id: int
    role: str
    content: str                          # 말풍선용 텍스트
    data: AnalysisData | None = None      # 구조화 결과 (assistant 메시지에만, user는 null)
    created_at: datetime

    model_config = {"from_attributes": True}


class CreateSessionRequest(BaseModel):
    """세션 생성 (초기 메시지 함께 저장 가능)"""
    title: str | None = Field(None, examples=["나이키 신발 상담"])
    user_id: int | None = None
    messages: list[MessageInput] = Field(default_factory=list)


class AddMessageRequest(BaseModel):
    """기존 세션에 메시지 추가 (1개 또는 여러 개)"""
    messages: list[MessageInput]


class SessionSummary(BaseModel):
    id: int
    title: str | None
    user_id: int | None
    message_count: int
    created_at: datetime


class SessionListResponse(BaseModel):
    items: list[SessionSummary]
    total: int


class SessionDetailResponse(BaseModel):
    id: int
    title: str | None
    user_id: int | None
    created_at: datetime
    messages: list[MessageItem]


# ── 통합: 채팅형 분석 (분석 + 대화 저장) ──────────────────────────
class ChatAnalyzeRequest(BaseModel):
    session_id: int | None = Field(None, description="없으면 새 대화방 생성")
    user_id: int | None = None
    question: str = Field(..., examples=["이거 살까 말까?"])
    image_base64: str | None = Field(
        None, description="트리거(살까/말까 등) 시에만 전송. 없으면 세션의 마지막 사진 재활용"
    )
    pro_mode: bool = Field(False, description="True면 Gemini에 이미지 직접 전송(정밀·비용↑)")
    context: ContextInput = Field(default_factory=ContextInput)


class ChatAnalyzeResponse(BaseModel):
    session_id: int
    verdict: Verdict
    verdict_label: str
    product_info: str | None = None
    reason: str
    pros: str | None = None
    cons: str | None = None
    caution: str | None = None
    recommendation: str | None = None
    pro_mode: bool
    image_reused: bool   # 이전 사진 재활용 여부
    messages: list[MessageItem]
