"""
Gemini Flash LLM 서비스
- Florence-2 캡션 + 맥락 정보 → 살래/고민/말래 JSON 판단
- Flutter에서 AI 모델 선택(설정 화면) 시 model명 파라미터로 교체 가능
"""
import json
import logging
import re

import httpx

from app.core.config import settings
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse, Verdict

logger = logging.getLogger(__name__)

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={api_key}"
)

SYSTEM_PROMPT = """
당신은 소비자의 구매 판단을 도와주는 AI 어시스턴트입니다.
아래 상품 정보와 맥락을 분석해 JSON 형식으로 구매 판단을 반환하세요.

판단 기준:
- buy (살래요): 가격 대비 가치가 높고 상태가 양호하며 목적에 부합할 때
- maybe (고민해요): 조건에 따라 달라질 수 있거나 추가 확인이 필요할 때
- no (말래요): 상태가 불량하거나 가격이 높거나 목적에 맞지 않을 때

반드시 다음 JSON 형식만 반환하세요 (다른 텍스트 없이):
{
  "verdict": "buy" | "maybe" | "no",
  "reason": "판단 이유 (2~3문장)",
  "caution": "주의사항 (있으면 1문장, 없으면 null)",
  "recommendation": "추천 행동 (maybe/no 시 1문장, buy 시 null 가능)"
}
""".strip()


def _build_user_prompt(caption: str, request: AnalyzeRequest) -> str:
    ctx = request.context
    lines = [
        f"[상품 이미지 분석]\n{caption}",
        "",
        "[사용자 질문]",
        request.question or "(질문 없음)",
        "",
        "[맥락 정보]",
        f"- 분류: {ctx.category or '미입력'}",
        f"- 가격: {ctx.price or '미입력'}",
        f"- 사용 목적: {ctx.purpose or '미입력'}",
        f"- 상품 상태: {_condition_label(ctx.condition)}",
        f"- 중요 기준: {', '.join(ctx.criteria) if ctx.criteria else '미선택'}",
    ]
    return "\n".join(lines)


def _condition_label(condition) -> str:
    mapping = {"good": "문음(좋음)", "normal": "보통", "poor": "불량"}
    return mapping.get(str(condition), "미입력") if condition else "미입력"


def _verdict_label(verdict: Verdict) -> str:
    return {"buy": "살래요", "maybe": "고민해요", "no": "말래요"}[verdict]


async def judge(
    caption: str,
    request: AnalyzeRequest,
    model: str | None = None,
) -> AnalyzeResponse:
    """Gemini Flash로 구매 판단 수행"""
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")

    model_name = model or settings.gemini_model
    url = GEMINI_API_URL.format(model=model_name, api_key=settings.gemini_api_key)
    user_prompt = _build_user_prompt(caption, request)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\n" + user_prompt}
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 512,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()

    data = resp.json()
    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    logger.info(f"Gemini 응답 raw: {raw_text[:200]}")

    # JSON 파싱 (마크다운 코드블록 제거 후)
    json_str = re.sub(r"```(?:json)?|```", "", raw_text).strip()
    parsed = json.loads(json_str)

    verdict = Verdict(parsed["verdict"])
    return AnalyzeResponse(
        verdict=verdict,
        verdict_label=_verdict_label(verdict),
        reason=parsed["reason"],
        caution=parsed.get("caution"),
        recommendation=parsed.get("recommendation"),
        caption=caption,
        history_id=None,  # 저장 후 주입
    )
