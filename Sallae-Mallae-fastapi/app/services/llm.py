from app.services.supabase_rag import query_rag_context  # 👈 추가
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
당신은 사용자의 지갑을 지키기 위해 팩트폭행과 쓴소리를 아끼지 않는 '극딜 충동구매 방지 AI'입니다.
절대 친절하게 말하지 말고, 친구처럼 반말로 강력하게 잔소리하세요.

아래 Florence-2가 분석한 상품 이미지 정보와 사용자가 입력한 예산/목적 등 맥락 데이터를 바탕으로 구매를 판단하세요.
특히 '통장 잔고(가격)'나 '상품 상태'에 문제가 있으면 뼈를 때리는 멘트로 구매를 뜯어말려야 합니다.

판단 기준:
- buy (살래요): 가격이 매우 합리적이고 목적에 완벽히 부합할 때만 허락 (그래도 칭찬은 쿨하게 할 것)
- maybe (고민해요): 굳이 지금 사야 하나 싶을 때 (한 번 더 생각하라고 잔소리)
- no (말래요): 예산 초과, 불량 상태, 예쁜 쓰레기일 경우 (호구 당하지 말라고 극딜)

[반드시 준수할 JSON 형식]
{
  "verdict": "buy" | "maybe" | "no",
  "reason": "팩트폭행이 담긴 찰진 잔소리 (2~3문장)",
  "caution": "주의사항 (있으면 1문장, 없으면 null)",
  "recommendation": "대안 추천 (1문장, buy 시 null 가능)"
}

[엄격한 기술 규칙]
1. 답변은 오직 위 JSON 형식만 출력하세요.
2. 마크다운 코드 블록(```json ... ```)으로 감싸서 출력하세요.
3. JSON 내부의 모든 문자열에는 큰따옴표(")만 사용하고, 강조하고 싶은 문구는 작은따옴표(')를 사용하세요.
4. 이유(reason)를 포함한 모든 답변 값에 절대 줄바꿈을 포함하지 마세요.
5. JSON 포맷이 깨지지 않도록 마지막까지 반드시 중괄호로 닫으세요.
""".strip()

async def judge_purchase(caption: str, rag_context: str, request: AnalyzeRequest) -> AnalyzeResponse:
    url = GEMINI_API_URL.format(model="gemini-2.5-flash", api_key=settings.gemini_api_key)
    
    # 프롬프트 구성 (RAG 데이터 주입)
    prompt = f"{SYSTEM_PROMPT}\n\n[상품 정보]: {caption}\n[맥락 정보]: {rag_context}\n[사용자 질문]: {request.question}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    async with httpx.AsyncClient(timeout=30.0) as client: # 30초 타임아웃 설정
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            # JSON 추출을 위한 정규표현식 (마크다운 제거)
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not match:
                raise ValueError("JSON 형식을 찾을 수 없음")
                
            parsed = json.loads(match.group(0))
            
            return AnalyzeResponse(
                verdict=parsed.get("verdict", "maybe"),
                verdict_label="결정 완료",
                reason=parsed.get("reason", "판단 불가"),
                caution=parsed.get("caution"),
                recommendation=parsed.get("recommendation"),
                caption=caption
            )
        except Exception as e:
            logger.error(f"Gemini 호출 실패: {e}")
            # 실패 시에도 최소한의 응답을 반환하여 서버가 안 죽게 함
            return AnalyzeResponse(
                verdict=Verdict.maybe,
                verdict_label="고민해요",
                reason="AI가 판단 중 지쳤나 봐요. 다시 시도해 볼래?",
                caption=caption
            )


def _build_user_prompt(
    caption: str,
    request: AnalyzeRequest,
    rag_context: str = ""
) -> str:

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

    if rag_context:
        lines.append("")
        lines.append(rag_context)

    return "\n".join(lines)



def _condition_label(condition) -> str:
    mapping = {"good": "좋음", "normal": "보통", "poor": "불량"}
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
    # 👈 [Retrieval] Gemini 프롬프트를 짜기 전에 Supabase를 먼저 찌릅니다!
    rag_context = await query_rag_context(caption, request.context.category)

    # 👈 주입 인자에 rag_context를 넘겨줍니다.
    user_prompt = _build_user_prompt(caption, request, rag_context)


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
