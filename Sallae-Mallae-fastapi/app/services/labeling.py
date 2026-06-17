"""
라벨링 서비스
1. Flutter에서 OpenCV 압축된 이미지(base64) 수신
2. 백엔드에서 OpenCV 전처리 (디코딩, 리사이즈, 정규화)
3. YOLO 객체 감지 → 바운딩박스 + 클래스 라벨
4. 라벨링된 이미지 + YOLO 포맷 .txt 파일을 dataset 폴더에 저장
   → 이 데이터가 파인튜닝 학습 데이터가 됨
"""
import base64
import json
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.schemas.labeling import BoundingBox, DetectedObject, LabelResponse

logger = logging.getLogger(__name__)

# 파인튜닝 데이터셋 저장 경로
DATASET_ROOT = Path("dataset")
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
ANNOTATED_DIR = DATASET_ROOT / "annotated"  # 바운딩박스 시각화 이미지

# YOLO 모델 (전역, 서버 시작 시 1회 로드)
_yolo_model = None


def load_yolo_model(model_name: str = "yolo11n.pt") -> None:
    """서버 시작 시 YOLO 모델 1회 로드"""
    global _yolo_model
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(model_name)
        logger.info(f"YOLO 모델 로드 완료: {model_name} ✅")
    except Exception as e:
        logger.error(f"YOLO 모델 로드 실패: {e}")
        _yolo_model = None


def is_loaded() -> bool:
    return _yolo_model is not None


def _ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


def _decode_image(image_base64: str) -> np.ndarray:
    """
    Flutter에서 받은 base64 이미지를 OpenCV BGR 배열로 디코딩
    (Flutter에서 이미 OpenCV 압축이 완료된 상태로 수신)
    """
    img_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패 — 유효한 이미지 데이터가 아닙니다.")
    return img


def _preprocess(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    YOLO 입력 전처리
    - 가로/세로 비율 유지하며 640x640 리사이즈
    - 빈 영역은 회색(114) 패딩
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded


def _draw_boxes(img: np.ndarray, detections: list[DetectedObject]) -> np.ndarray:
    """감지된 객체에 바운딩박스 + 라벨 텍스트 그리기"""
    h, w = img.shape[:2]
    annotated = img.copy()

    for obj in detections:
        bbox = obj.bbox
        x1 = int((bbox.x_center - bbox.width / 2) * w)
        y1 = int((bbox.y_center - bbox.height / 2) * h)
        x2 = int((bbox.x_center + bbox.width / 2) * w)
        y2 = int((bbox.y_center + bbox.height / 2) * h)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{obj.class_name} {obj.bbox.confidence:.2f}"
        cv2.putText(
            annotated, label_text, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )

    return annotated


def _save_yolo_label(label_path: Path, detections: list[DetectedObject]) -> None:
    """
    YOLO 학습용 .txt 파일 저장
    포맷: class_id x_center y_center width height (정규화 좌표)
    """
    lines = []
    for obj in detections:
        b = obj.bbox
        lines.append(
            f"{obj.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}"
        )
    label_path.write_text("\n".join(lines))


def run_labeling(
    image_base64: str,
    category_hint: str | None = None,
    conf_threshold: float = 0.25,
) -> dict:
    """
    전체 라벨링 파이프라인 실행

    Returns:
        dict with keys: detections, primary_label, confidence,
                        annotated_image_base64, image_path, label_path
    """
    _ensure_dirs()

    # 1. OpenCV 디코딩 + 전처리
    img_original = _decode_image(image_base64)
    h_orig, w_orig = img_original.shape[:2]

    # 2. YOLO 감지
    detections: list[DetectedObject] = []

    if _yolo_model is not None:
        results = _yolo_model.predict(
            source=img_original,
            conf=conf_threshold,
            verbose=False,
        )
        for result in results:
            for box in result.boxes:
                x_c, y_c, bw, bh = box.xywhn[0].tolist()  # 정규화 좌표
                detections.append(DetectedObject(
                    class_id=int(box.cls[0]),
                    class_name=result.names[int(box.cls[0])],
                    bbox=BoundingBox(
                        x_center=x_c,
                        y_center=y_c,
                        width=bw,
                        height=bh,
                        confidence=float(box.conf[0]),
                    ),
                ))
        # 신뢰도 내림차순 정렬
        detections.sort(key=lambda d: d.bbox.confidence, reverse=True)
    else:
        logger.warning("YOLO 미로드 — 감지 건너뜀")

    # 3. 주요 라벨 결정
    #    category_hint(사용자 입력) 우선, 없으면 YOLO 최고 신뢰도 객체
    if detections:
        primary_label = category_hint or detections[0].class_name
        confidence = detections[0].bbox.confidence
    else:
        primary_label = category_hint or "unknown"
        confidence = 0.0

    # 4. 바운딩박스 시각화 이미지 생성
    annotated_img = _draw_boxes(img_original, detections)
    _, buf = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(buf).decode()

    # 5. 데이터셋 파일 저장
    file_id = uuid.uuid4().hex
    img_path = IMAGES_DIR / f"{file_id}.jpg"
    label_path = LABELS_DIR / f"{file_id}.txt"
    annotated_path = ANNOTATED_DIR / f"{file_id}_annotated.jpg"

    cv2.imwrite(str(img_path), img_original)
    cv2.imwrite(str(annotated_path), annotated_img)
    _save_yolo_label(label_path, detections)

    logger.info(
        f"라벨링 완료 | label={primary_label} conf={confidence:.2f} "
        f"objects={len(detections)} file={file_id}"
    )

    return {
        "detections": detections,
        "primary_label": primary_label,
        "confidence": confidence,
        "annotated_image_base64": annotated_b64,
        "image_path": str(img_path),
        "label_path": str(label_path),
        "detections_json": json.dumps([d.model_dump() for d in detections], ensure_ascii=False),
    }


def get_dataset_stats(records: list) -> dict[str, int]:
    """라벨 분포 집계 (DB 레코드 리스트 → {클래스명: 개수})"""
    dist: dict[str, int] = {}
    for r in records:
        label = r.corrected_label or r.primary_label
        dist[label] = dist.get(label, 0) + 1
    return dist


def export_dataset_yaml(dataset_path: str = "dataset") -> str:
    """
    YOLO 학습용 data.yaml 생성
    ultralytics train data=dataset/data.yaml 로 파인튜닝 시작 가능
    """
    yaml_path = Path(dataset_path) / "data.yaml"
    content = f"""path: {Path(dataset_path).resolve()}
train: images
val: images

nc: 80
names: {_yolo_model.names if _yolo_model else {}}
"""
    yaml_path.write_text(content)
    return str(yaml_path)
