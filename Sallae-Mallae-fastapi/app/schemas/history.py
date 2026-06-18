"""히스토리 스키마 (화면 13 - History)"""
from datetime import datetime
from pydantic import BaseModel
from app.schemas.analyze import Verdict


class HistoryItem(BaseModel):
    id: int
    verdict: Verdict
    verdict_label: str
    question: str | None
    product_info: str | None   # 한국어 제품 상세 설명 (화면 표시용)
    pros: str | None
    cons: str | None
    reason: str
    caution: str | None
    recommendation: str | None
    category: str | None
    price: str | None
    purpose: str | None
    condition: str | None
    criteria: list[str]
    has_image: bool                    # 이미지 저장 여부
    image_base64: str | None = None    # 업로드된 상품 이미지 (프론트 표시용)
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryListResponse(BaseModel):
    items: list[HistoryItem]
    total: int


class HistoryDetailResponse(HistoryItem):
    """단건 조회 시 이미지 포함"""
    image_base64: str | None = None
