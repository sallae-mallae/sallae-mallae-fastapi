"""
Gemini Flash LLM 서비스
Florence-2 캡션 + Supabase RAG 족보 + 맥락 → 살래/고민해요/말래요 잔소리 판단
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
당신은 소비자의 충동구매를 막아주는 잔소리꾼 AI입니다.
아래 상품 정보, 구매 맥락, 그리고 비슷한 상품의 구매 사례/후회 데이터를 바탕으로
구매 판단을 JSON 형식으로 반환하세요.

판단 기준:
- buy (살래요): 가격 대비 가치가 높고 상태 양호, 목적에 부합, 후회 사례 적음
- maybe (고민해요): 조건부 추천, 추가 확인 필요, 비슷한 후회 사례 일부 존재
- no (말래요): 상태 불량, 가격 과대, 목적 불일치, 유사 후회 사례 다수

반드시 아래 JSON만 반환하세요 (마크다운, 설명 텍스트 절대 금지):
{
  "verdict": "buy" | "maybe" | "no",
  "reason": "판단 이유 — 잔소리 말투로 2~3문장. RAG 사례가 있으면 인용.",
  "caution": "주의사항 1문장 또는 null",
  "recommendation": "추천 행동 1문장 또는 null"
}
""".strip()


def _build_prompt(caption: str, rag_context: str, request: AnalyzeRequest) -> str:
    ctx = request.context
    parts = [
        "[상품 이미지 분석 결과 (Florence-2)]",
        caption,
        "",
        "[사용자 질문]",
        request.question or "(질문 없음)",
        "",
        "[구매 맥락]",
        f"- 분류: {ctx.category or '미입력'}",
        f"- 가격: {ctx.price or '미입력'}",
        f"- 사용 목적: {ctx.purpose or '미입력'}",
        f"- 상품 상태: {_condition_label(ctx.condition)}",
        f"- 중요 기준: {', '.join(ctx.criteria) if ctx.criteria else '미선택'}",
    ]
    if rag_context:
        parts += ["", rag_context]
    return "\n".join(parts)


def _condition_label(condition) -> str:
    mapping = {"good": "문음(좋음)", "normal": "보통", "poor": "불량"}
    return mapping.get(str(condition), "미입력") if condition else "미입력"


def _verdict_label(verdict: Verdict) -> str:
    return {"buy": "살래요", "maybe": "고민해요", "no": "말래요"}[verdict]


async def judge(
    caption: str,
    rag_context: str,
    request: AnalyzeRequest,
    model: str | None = None,
) -> AnalyzeResponse:
    """Florence-2 캡션 + RAG 컨텍스트 → Gemini 판단"""
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")

    model_name = model or settings.gemini_model
    url = GEMINI_API_URL.format(model=model_name, api_key=settings.gemini_api_key)
    prompt = _build_prompt(caption, rag_context, request)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}],
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
    logger.info(f"Gemini 응답: {raw_text[:200]}")

    # JSON 파싱 — 마크다운 코드블록 제거 후 파싱, 실패 시 1회 재시도
    json_str = re.sub(r"```(?:json)?|```", "", raw_text).strip()
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # JSON만 추출 재시도
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if not match:
            raise ValueError(f"Gemini 응답 파싱 실패: {raw_text[:300]}")
        parsed = json.loads(match.group())

    verdict = Verdict(parsed["verdict"])
    return AnalyzeResponse(
        verdict=verdict,
        verdict_label=_verdict_label(verdict),
        reason=parsed["reason"],
        caution=parsed.get("caution"),
        recommendation=parsed.get("recommendation"),
        caption=caption,
        history_id=None,
    )
