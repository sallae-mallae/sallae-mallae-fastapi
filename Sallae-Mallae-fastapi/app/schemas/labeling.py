"""라벨링 관련 스키마"""
from datetime import datetime
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """YOLO 감지 바운딩박스 (정규화 좌표 0~1)"""
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = Field(..., description="감지 신뢰도 (0~1)")


class DetectedObject(BaseModel):
    """감지된 객체 하나"""
    class_id: int
    class_name: str
    bbox: BoundingBox


class LabelRequest(BaseModel):
    """Flutter → 서버 라벨링 요청"""
    image_base64: str = Field(..., description="OpenCV 압축 이미지 base64")
    category_hint: str | None = Field(None, description="사용자 입력 분류 힌트 (맥락 06화면)")


class LabelResponse(BaseModel):
    """서버 → Flutter 라벨링 응답"""
    label_id: int
    detected_objects: list[DetectedObject]
    primary_label: str = Field(..., description="가장 신뢰도 높은 객체 클래스명")
    confidence: float
    annotated_image_base64: str = Field(..., description="바운딩박스 그려진 이미지 base64")
    saved_to_dataset: bool


class LabelCorrection(BaseModel):
    """수동 검수 시 라벨 수정 요청"""
    corrected_label: str = Field(..., description="수정된 정확한 클래스명")
    is_verified: bool = Field(True)


class LabelItem(BaseModel):
    """히스토리 목록용"""
    label_id: int
    primary_label: str
    corrected_label: str | None
    confidence: float
    is_verified: bool
    category_hint: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetStats(BaseModel):
    """파인튜닝 데이터셋 현황"""
    total_samples: int
    verified_samples: int
    unverified_samples: int
    label_distribution: dict[str, int]
    dataset_path: str
