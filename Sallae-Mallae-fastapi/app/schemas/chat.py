"""AI 채팅 대화 기록 스키마"""
from datetime import datetime
from pydantic import BaseModel, Field


class MessageInput(BaseModel):
    role: str = Field(..., description="user | assistant", examples=["user"])
    content: str = Field(..., examples=["이 신발 살 만한가요?"])


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
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
