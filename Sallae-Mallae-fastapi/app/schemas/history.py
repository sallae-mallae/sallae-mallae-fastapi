"""히스토리 스키마 (화면 13 - History)"""
from datetime import datetime
from pydantic import BaseModel
from app.schemas.analyze import Verdict


class HistoryItem(BaseModel):
    id: int
    verdict: Verdict
    verdict_label: str
    question: str | None
    reason: str
    caution: str | None
    recommendation: str | None
    category: str | None
    price: str | None
    purpose: str | None
    condition: str | None
    criteria: list[str]
    has_image: bool   # 이미지 저장 여부 (image_base64 노출 X)
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryListResponse(BaseModel):
    items: list[HistoryItem]
    total: int


class HistoryDetailResponse(HistoryItem):
    """단건 조회 시 이미지 포함"""
    image_base64: str | None = None
