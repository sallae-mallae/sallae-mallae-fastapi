"""분석 요청/응답 스키마 (Flutter ↔ FastAPI)"""
from enum import Enum
from pydantic import BaseModel, Field


class ProductCondition(str, Enum):
    """상품 상태 (화면 06)"""
    good = "good"       # 문음
    normal = "normal"   # 보통
    poor = "poor"       # 불량


class Verdict(str, Enum):
    """판단 결과"""
    buy = "buy"       # 살래요 🟢
    maybe = "maybe"   # 고민해요 🟡
    no = "no"         # 말래요 🔴


class ContextInput(BaseModel):
    """맥락 입력 (화면 06 - UX Context Input)"""
    category: str | None = Field(None, example="가방", description="상품 분류")
    price: str | None = Field(None, example="50,000원", description="가격")
    purpose: str | None = Field(None, example="매일 쓰는 가방이 필요해요", description="사용 목적")
    condition: ProductCondition | None = Field(None, description="상품 상태: good/normal/poor")
    criteria: list[str] = Field(default_factory=list, example=["가격", "상태"], description="중요 기준")


class AnalyzeRequest(BaseModel):
    """Flutter → 서버 분석 요청"""
    image_base64: str = Field(..., description="카메라 촬영 이미지 (base64)")
    question: str | None = Field(None, example="이 가방 살 만한가요?", description="사용자 질문")
    context: ContextInput = Field(default_factory=ContextInput, description="맥락 정보")
    save_image: bool = Field(False, description="이미지 서버 저장 여부 (설정 화면 연동)")


class AnalyzeResponse(BaseModel):
    """서버 → Flutter 분석 응답"""
    verdict: Verdict = Field(..., description="판단 결과: buy/maybe/no")
    verdict_label: str = Field(..., description="판단 레이블: 살래요/고민해요/말래요")
    reason: str = Field(..., description="판단 이유")
    caution: str | None = Field(None, description="주의사항 (buy/maybe 시)")
    recommendation: str | None = Field(None, description="추천 행동 (maybe/no 시)")
    caption: str = Field(..., description="Florence-2 상품 인식 캡션")
    history_id: int | None = Field(None, description="저장된 기록 ID")
