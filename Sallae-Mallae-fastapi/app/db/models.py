"""SQLAlchemy ORM 모델"""
from datetime import datetime
from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base


class HistoryRecord(Base):
    """구매 판단 기록 (화면 13 - History)"""
    __tablename__ = "history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # 판단 결과: "buy" | "maybe" | "no"
    verdict: Mapped[str] = mapped_column(String(10), nullable=False)

    # AI 분석 요약
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)   # Florence-2 캡션
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    caution: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 맥락 정보 (화면 06 - Context Input)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)   # 분류 (가방 등)
    price: Mapped[str | None] = mapped_column(String(50), nullable=True)       # 가격 (50,000원)
    purpose: Mapped[str | None] = mapped_column(Text, nullable=True)           # 사용 목적
    condition: Mapped[str | None] = mapped_column(String(10), nullable=True)   # 문음/보통/불량
    criteria: Mapped[str | None] = mapped_column(String(100), nullable=True)   # 중요 기준 (콤마 구분)

    # 이미지 (설정에서 사진 서버 저장 ON 시에만 저장)
    image_base64: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
