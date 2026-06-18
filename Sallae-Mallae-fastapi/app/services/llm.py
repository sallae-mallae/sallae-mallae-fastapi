"""
Gemini Flash LLM 서비스
Florence-2 캡션 + Supabase RAG 족보 + 맥락 → 살래/고민해요/말래요 잔소리 판단
"""
import asyncio
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
당신은 소비자의 현명한 소비를 도와주는 친근한 쇼핑 도우미 AI입니다.
첨부된 실제 상품 사진을 직접 보고 어떤 제품인지 정확히 파악한 뒤 살지 말지를 추론합니다.
함께 제공되는 이미지 캡션과 검색 정보는 보조 자료로 참고하세요.
부가 정보(가격, 목적 등)는 입력된 경우에만 참고하고, 없어도 사진만으로 판단하세요.

진행 순서:
1. 먼저 사진 속 제품이 무엇인지 파악하고 상세 정보(종류, 특징, 일반적인 용도/스펙)를 설명합니다.
2. 그 다음 가성비, 이점, 단점을 균형 있게 따져 구매 판단을 합니다.

톤 가이드 (중요):
- 무조건 말리는 잔소리꾼이 아니라, 좋은 점은 확실히 칭찬하고 응원하는 긍정적인 태도를 유지하세요.
- 단점은 솔직하게 알려주되, 비난조가 아니라 "이런 점만 확인하면 좋아요" 식의 따뜻한 조언으로 표현하세요.
- 살 만한 물건이면 자신 있게 추천해 주세요.

판단 기준:
- buy (살래요): 가성비가 좋고 이점이 분명함 — 자신 있게 추천
- maybe (고민해요): 매력적이지만 한두 가지 확인하면 더 좋음
- no (말래요): 아쉬운 점이 분명히 커서 신중할 필요가 있음

반드시 아래 JSON만 반환하세요 (마크다운, 설명 텍스트 절대 금지):
{
  "product_info": "사진 속 제품이 무엇인지 + 종류/특징/일반적 용도 설명 2~3문장",
  "verdict": "buy" | "maybe" | "no",
  "reason": "종합 판단 이유 — 가성비 평가 포함, 긍정적이고 친근한 말투로 2~3문장",
  "pros": "이 상품을 사면 좋은 점 1~2문장",
  "cons": "확인하면 좋을 아쉬운 점 1~2문장",
  "caution": "특히 챙기면 좋을 점 1문장 또는 null",
  "recommendation": "추천 행동 1문장 또는 null"
}
""".strip()


def _build_prompt(caption: str, rag_context: str, request: AnalyzeRequest) -> str:
    ctx = request.context
    parts = [
        "[상품 이미지 분석 결과 (사진 기반)]",
        caption,
    ]

    # 부가 정보는 입력된 것만 추가 (사진만으로도 판단 가능)
    extras = []
    if request.question:
        extras.append(f"- 사용자 질문: {request.question}")
    if ctx.category:
        extras.append(f"- 분류: {ctx.category}")
    if ctx.price:
        extras.append(f"- 가격: {ctx.price}")
    if ctx.purpose:
        extras.append(f"- 사용 목적: {ctx.purpose}")
    if ctx.condition:
        extras.append(f"- 상품 상태: {_condition_label(ctx.condition)}")
    if ctx.criteria:
        extras.append(f"- 중요 기준: {', '.join(ctx.criteria)}")

    if extras:
        parts += ["", "[참고용 부가 정보 (있으면 참고)]"] + extras

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

    # 텍스트 프롬프트 + 실제 상품 이미지를 함께 전송 → 더 정확한 분석
    parts: list[dict] = [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]
    if request.image_base64:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": request.image_base64,
            }
        })

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",  # JSON 강제 출력 (마크다운 코드블록 방지)
            "thinkingConfig": {"thinkingBudget": 0},  # 2.5-flash thinking 비활성화 (토큰 절약)
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code == 429:
            logger.error(f"Gemini 429 응답 본문: {resp.text}")
            raise ValueError("Gemini API 호출 한도 초과 (429). 잠시 후 다시 시도하세요.")
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
        product_info=parsed.get("product_info"),
        reason=parsed["reason"],
        pros=parsed.get("pros"),
        cons=parsed.get("cons"),
        caution=parsed.get("caution"),
        recommendation=parsed.get("recommendation"),
        caption=caption,
        history_id=None,
    )
