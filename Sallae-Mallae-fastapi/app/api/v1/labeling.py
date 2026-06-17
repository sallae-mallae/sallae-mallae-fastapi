"""
/api/v1/label
이미지 라벨링 + 파인튜닝 데이터셋 관리 API
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import LabelRecord
from app.schemas.labeling import (
    LabelRequest,
    LabelResponse,
    LabelCorrection,
    LabelItem,
    DatasetStats,
)
from app.services import labeling

router = APIRouter()


@router.post(
    "/label",
    response_model=LabelResponse,
    summary="이미지 라벨링",
    description="""
Flutter에서 OpenCV 압축된 이미지를 받아 YOLO로 객체 감지 후 라벨링합니다.

**처리 흐름:**
1. OpenCV 디코딩 + 전처리 (리사이즈, 패딩)
2. YOLO 객체 감지 → 바운딩박스 + 클래스 라벨
3. dataset/images/, dataset/labels/ 에 파일 저장 (파인튜닝 데이터)
4. 바운딩박스 시각화 이미지 반환
""",
)
async def label_image(
    request: LabelRequest,
    db: AsyncSession = Depends(get_db),
) -> LabelResponse:
    try:
        result = labeling.run_labeling(
            image_base64=request.image_base64,
            category_hint=request.category_hint,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"라벨링 오류: {e}")

    # DB 저장
    record = LabelRecord(
        primary_label=result["primary_label"],
        confidence=result["confidence"],
        detections_json=result["detections_json"],
        corrected_label=None,
        is_verified=False,
        category_hint=request.category_hint,
        image_path=result["image_path"],
        label_path=result["label_path"],
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return LabelResponse(
        label_id=record.id,
        detected_objects=result["detections"],
        primary_label=result["primary_label"],
        confidence=result["confidence"],
        annotated_image_base64=result["annotated_image_base64"],
        saved_to_dataset=True,
    )


@router.put(
    "/label/{label_id}",
    response_model=LabelItem,
    summary="라벨 수동 검수 (수정)",
    description="YOLO 자동 라벨이 틀렸을 때 올바른 클래스명으로 수정합니다. 수정된 데이터가 파인튜닝 품질을 높입니다.",
)
async def correct_label(
    label_id: int,
    correction: LabelCorrection,
    db: AsyncSession = Depends(get_db),
) -> LabelItem:
    record = await db.get(LabelRecord, label_id)
    if not record:
        raise HTTPException(status_code=404, detail="라벨 기록을 찾을 수 없습니다.")

    record.corrected_label = correction.corrected_label
    record.is_verified = correction.is_verified

    # YOLO .txt 파일도 수정된 라벨로 업데이트
    if record.label_path:
        import re
        from pathlib import Path
        label_file = Path(record.label_path)
        if label_file.exists():
            lines = label_file.read_text().splitlines()
            # 첫 번째 감지 객체의 class_id를 corrected_label 기반으로 교체하려면
            # 실제 class_id 매핑이 필요하므로 여기선 주석으로 기록
            label_file.write_text(
                "\n".join(lines) + f"\n# corrected: {correction.corrected_label}"
            )

    await db.commit()
    await db.refresh(record)

    return LabelItem(
        label_id=record.id,
        primary_label=record.primary_label,
        corrected_label=record.corrected_label,
        confidence=record.confidence,
        is_verified=record.is_verified,
        category_hint=record.category_hint,
        created_at=record.created_at,
    )


@router.get(
    "/label/dataset/stats",
    response_model=DatasetStats,
    summary="파인튜닝 데이터셋 현황",
    description="수집된 라벨 데이터 통계. 파인튜닝 시작 전 데이터 분포를 확인하세요.",
)
async def dataset_stats(db: AsyncSession = Depends(get_db)) -> DatasetStats:
    total = (await db.execute(select(func.count()).select_from(LabelRecord))).scalar_one()
    verified = (
        await db.execute(
            select(func.count()).select_from(LabelRecord).where(LabelRecord.is_verified == True)
        )
    ).scalar_one()
    records = (await db.execute(select(LabelRecord))).scalars().all()
    dist = labeling.get_dataset_stats(records)

    return DatasetStats(
        total_samples=total,
        verified_samples=verified,
        unverified_samples=total - verified,
        label_distribution=dist,
        dataset_path="dataset/",
    )


@router.get(
    "/label/dataset/export-yaml",
    summary="YOLO 학습용 data.yaml 생성",
    description="`ultralytics yolo train data=dataset/data.yaml` 명령으로 파인튜닝을 시작할 수 있습니다.",
)
async def export_yaml():
    try:
        yaml_path = labeling.export_dataset_yaml()
        return {"yaml_path": yaml_path, "message": "data.yaml 생성 완료. YOLO 파인튜닝 준비 완료."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/label",
    response_model=list[LabelItem],
    summary="라벨링 기록 목록",
)
async def list_labels(
    skip: int = 0,
    limit: int = 50,
    only_unverified: bool = False,
    db: AsyncSession = Depends(get_db),
) -> list[LabelItem]:
    query = select(LabelRecord).order_by(LabelRecord.created_at.desc())
    if only_unverified:
        query = query.where(LabelRecord.is_verified == False)
    rows = (await db.execute(query.offset(skip).limit(limit))).scalars().all()
    return [
        LabelItem(
            label_id=r.id,
            primary_label=r.primary_label,
            corrected_label=r.corrected_label,
            confidence=r.confidence,
            is_verified=r.is_verified,
            category_hint=r.category_hint,
            created_at=r.created_at,
        )
        for r in rows
    ]
