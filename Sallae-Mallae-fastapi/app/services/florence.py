"""
Florence-2 서비스
- 서버 시작 시 1회 모델 로드 (lifespan)
- 이미지(base64) → 상품 인식 캡션 텍스트 생성
"""
import base64
import logging
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# 전역 모델 상태 (lifespan에서 주입)
_model: Any = None
_processor: Any = None
_device: str = "cpu"


def load_model(model_id: str) -> None:
    """앱 시작 시 Florence-2 모델 로드 (1회)"""
    global _model, _processor, _device

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Florence-2 로드 시작: {model_id} (device={_device})")

        _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(_device)
        _model.eval()

        logger.info("Florence-2 로드 완료 ✅")
    except Exception as e:
        logger.error(f"Florence-2 로드 실패: {e}")
        # 모델 없이도 서버 구동 가능 (fallback 모드)
        _model = None
        _processor = None


def is_loaded() -> bool:
    return _model is not None and _processor is not None


def generate_caption(image_base64: str) -> str:
    """
    이미지 base64 → 상품 캡션 텍스트
    Florence-2 미로드 시 fallback 문자열 반환
    """
    if not is_loaded():
        logger.warning("Florence-2 미로드 — fallback 캡션 사용")
        return "상품 이미지 (AI 캡션 생성 불가)"

    try:
        import torch

        # base64 → PIL Image
        #image_data = base64.b64decode(image_base64)
        # (아래 방어 코드로 교체!)
        clean_b64 = image_base64.strip()
        padded_b64 = clean_b64 + "=" * (-len(clean_b64) % 4)
        image_data = base64.b64decode(padded_b64)

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Florence-2 캡션 생성 태스크: <MORE_DETAILED_CAPTION>
        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = _processor(text=task_prompt, images=image, return_tensors="pt").to(_device)

        with torch.no_grad():
            generated_ids = _model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
            )

        generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = _processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        caption = result.get(task_prompt, "")
        logger.info(f"Florence-2 캡션: {caption[:80]}...")
        return caption

    except Exception as e:
        logger.error(f"Florence-2 캡션 생성 오류: {e}")
        return "상품 이미지 (캡션 생성 중 오류)"

def test_local_image(image_path: str) -> str:
    """test_pipeline.py에서 로컬 이미지를 직접 테스트하기 위한 함수"""
    if not is_loaded():
        return "Florence-2 미로드"
    
    try:
        import torch
        # 로컬 파일 경로에서 직접 이미지 열기
        image = Image.open(image_path).convert("RGB")

        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = _processor(text=task_prompt, images=image, return_tensors="pt").to(_device)

        with torch.no_grad():
            generated_ids = _model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
            )

        generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = _processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        return result.get(task_prompt, "")

    except Exception as e:
        logger.error(f"로컬 이미지 캡션 생성 오류: {e}")
        return "상품 이미지 (캡션 생성 중 오류)"