"""분석 요청/응답 스키마 (Flutter ↔ FastAPI)"""
from enum import Enum
from pydantic import BaseModel, Field


class ProductCondition(str, Enum):
    good = "good"       # 문음
    normal = "normal"   # 보통
    poor = "poor"       # 불량


class Verdict(str, Enum):
    buy = "buy"       # 살래요
    maybe = "maybe"   # 고민해요
    no = "no"         # 말래요


class ContextInput(BaseModel):
    category: str | None = Field(None, example="가방")
    price: str | None = Field(None, example="50,000원")
    purpose: str | None = Field(None, example="매일 쓰는 가방이 필요해요")
    condition: ProductCondition | None = Field(None, description="good/normal/poor")
    criteria: list[str] = Field(default_factory=list, example=["가격", "상태"])


# 현재 동환님의 schemas/analyze.py 상태
class AnalyzeRequest(BaseModel):
    """Flutter → 서버 분석 요청"""
    image_base64: str = Field(..., description="JPEG base64 인코딩 이미지")
    question: str | None = Field(None, example="이 가방 살 만한가요?")
    context: ContextInput = Field(default_factory=ContextInput)
    
    # 💡 [여기 추가!] 밑빠진 독을 막아주는 핵심 코드입니다.
    ocr_text: str | None = Field(None, description="프론트엔드 ML Kit 텍스트")
    
    save_image: bool = Field(False, description="이미지 서버 저장 여부")


class AnalyzeResponse(BaseModel):
    """서버 → Flutter 분석 응답"""
    verdict: Verdict
    verdict_label: str = Field(..., description="살래요/고민해요/말래요")
    reason: str
    caution: str | None = None
    recommendation: str | None = None
    caption: str = Field(..., description="Florence-2 캡션")
    rag_used: bool = Field(False, description="RAG 컨텍스트 사용 여부")
    history_id: int | None = None
