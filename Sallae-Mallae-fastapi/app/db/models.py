"""SQLAlchemy ORM 모델"""
from datetime import datetime
from sqlalchemy import String, Text, DateTime, Float, Boolean, Integer, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base


class ChatSession(Base):
    """AI 채팅 대화 세션"""
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)   # 로그인 사용자(선택)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class ChatMessage(Base):
    """채팅 메시지 (세션에 속함)"""
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class User(Base):
    """회원 테이블"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    nickname: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class HistoryRecord(Base):
    """구매 판단 기록 (화면 13 - History)"""
    __tablename__ = "history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # 판단 결과: "buy" | "maybe" | "no"
    verdict: Mapped[str] = mapped_column(String(10), nullable=False)

    # AI 분석 요약
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)        # Florence-2 영어 캡션 (내부용)
    product_info: Mapped[str | None] = mapped_column(Text, nullable=True)   # 한국어 제품 상세 설명 (화면 표시용)
    pros: Mapped[str | None] = mapped_column(Text, nullable=True)
    cons: Mapped[str | None] = mapped_column(Text, nullable=True)
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


class LabelRecord(Base):
    """파인튜닝 라벨링 기록"""
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # YOLO 감지 결과 (가장 신뢰도 높은 객체)
    primary_label: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # 전체 감지 결과 JSON (모든 바운딩박스 포함)
    detections_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 수동 검수
    corrected_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # 사용자 힌트 (맥락 06화면 category 값)
    category_hint: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # 저장된 파일 경로 (dataset 폴더 내)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    label_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # YOLO .txt 파일

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
