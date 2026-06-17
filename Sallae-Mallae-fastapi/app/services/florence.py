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

_model: Any = None
_processor: Any = None
_device: str = "cpu"


def load_model(model_id: str) -> None:
    global _model, _processor, _device
    try:
        import torch
        import transformers.dynamic_module_utils as _dmu
        import transformers.utils.import_utils as _import_utils

        # flash_attn import 체크 우회 (CPU 환경)
        _orig_check = _dmu.check_imports
        def _patched_check(filename):
            try:
                return _orig_check(filename)
            except ImportError as e:
                if "flash_attn" in str(e):
                    return []
                raise
        _dmu.check_imports = _patched_check
        _import_utils.is_flash_attn_2_available = lambda: False
        _import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False

        from transformers import AutoProcessor, AutoModelForCausalLM

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Florence-2 로드 시작: {model_id} (device={_device})")

        _processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, revision="main"
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            trust_remote_code=True,
            revision="main",
            ignore_mismatched_sizes=True,
            attn_implementation="eager",  # flash_attn 없이 CPU에서 동작
        ).to(_device)
        _model.eval()
        logger.info("Florence-2 로드 완료 ✅")
    except Exception as e:
        import traceback
        logger.warning(f"Florence-2 로드 실패 (fallback 모드로 동작): {e}")
        logger.warning(traceback.format_exc())
        _model = None
        _processor = None


def is_loaded() -> bool:
    return _model is not None and _processor is not None


def generate_caption(image_base64: str) -> str:
    """
    이미지 base64 → 상품 캡션 텍스트
    Florence-2 미로드 시 fallback 반환
    """
    if not is_loaded():
        logger.warning("Florence-2 미로드 — fallback 캡션 사용")
        return "상품 이미지 (Florence-2 캡션 생성 불가 — fallback 모드)"

    try:
        import torch

        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")

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
        logger.info(f"Florence-2 캡션: {caption[:100]}...")
        return caption

    except Exception as e:
        logger.error(f"Florence-2 캡션 생성 오류: {e}")
        return "상품 이미지 (캡션 생성 중 오류)"
