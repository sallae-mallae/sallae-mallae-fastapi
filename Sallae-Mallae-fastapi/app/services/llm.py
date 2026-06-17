"""
Groq (Llama 3.1) LLM 서비스
Florence-2 캡션 + Supabase RAG 족보 + 맥락 → 살래/고민해요/말래요 잔소리 판단
"""
import json
import logging
import httpx
import os

from app.core.config import settings
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse, Verdict

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# app/services/llm.py 상단 수정본

SYSTEM_PROMPT = """너는 사용자의 지갑을 지키는 '극딜 충동구매 방지 AI'다. 
친절함은 버리고, 친구처럼 반말로 뼈 때리는 팩트폭행 잔소리를 해라.
상품 이미지, 사진 속 글자(OCR), 맥락, RAG 후회 데이터를 바탕으로 가성비를 따져서 구매를 뜯어말려라.

판단 기준:
- buy: 완벽히 합리적일 때만 허락 (그래도 쿨하게 칭찬할 것)
- maybe: 굳이 지금? (한 번 더 생각하라고 잔소리)
- no: 예산 초과, 예쁜 쓰레기 (호구 당하지 말라고 극딜)

[중요 규칙]
- 💡 [핵심] '사진 속 글자(OCR)'에 가격, 브랜드, 모델명 등이 보인다면 최우선 팩트로 삼아서 잔소리해.
- 만약 RAG 데이터가 현재 상품과 무관하면, 억지로 연관 짓지 말고 상품 정보와 사용자 입력값에만 집중해.
- 저렴한 생필품은 관대하게 판단해.

출력은 반드시 아래 JSON 스키마를 엄격히 따를 것. (이유는 최대 2~3문장으로 짧고 강렬하게)
{
  "verdict": "buy" | "maybe" | "no",
  "reason": "팩트폭행이 담긴 찰진 잔소리 (2~3문장)",
  "caution": "주의사항 (있으면 1문장, 없으면 null)",
  "recommendation": "대안 추천 (1문장, buy 시 null 가능)"
}"""

def _build_prompt(caption: str, rag_context: str, request: AnalyzeRequest) -> str:
    ctx = request.context
    
    # 💡 [핵심] 프론트엔드에서 넘어온 ML Kit OCR 데이터가 있다면 프롬프트에 추가!
    # getattr를 쓰는 이유는 기존 테스트 코드나 다른 곳에서 ocr_text를 안 보냈을 때 에러 방지용입니다.
    ocr_data = getattr(request, 'ocr_text', None)
    ocr_info = f"사진 속 글자(OCR): {ocr_data}" if ocr_data else "사진 속 글자: 없음"

    parts = [
        f"이미지: {caption}",
        ocr_info,  # 💡 Groq야, 이것도 읽어봐!
        f"질문: {request.question or '없음'}",
        f"가격: {ctx.price or '미입력'}, 목적: {ctx.purpose or '미입력'}, 상태: {_condition_label(ctx.condition)}",
    ]
    if rag_context:
        parts.append(f"RAG 데이터: {rag_context}")
    return " | ".join(parts)

# ... (아래 judge 함수 코드는 올려주신 그대로 두시면 됩니다!) ...
def _condition_label(condition) -> str:
    mapping = {"good": "좋음", "normal": "보통", "poor": "불량"}
    return mapping.get(str(condition), "미입력") if condition else "미입력"

def _verdict_label(verdict: Verdict) -> str:
    return {"buy": "살래요", "maybe": "고민해요", "no": "말래요"}[verdict]

async def judge(
    caption: str,
    rag_context: str,
    request: AnalyzeRequest,
    model: str | None = None,
) -> AnalyzeResponse:
    
    groq_api_key = getattr(settings, "groq_api_key", os.getenv("GROQ_API_KEY"))
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    model_name = model or "llama-3.1-8b-instant"
    
    prompt = _build_prompt(caption, rag_context, request)

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.5,
        "max_tokens": 250
    }

    async with httpx.AsyncClient(timeout=6.0) as client:
        try:
            resp = await client.post(GROQ_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
            # 💡 [여기 복구됨!] 결과값을 파싱해서 드디어 밖으로 내보냅니다!
            raw_text = data["choices"][0]["message"]["content"]
            logger.info(f"Groq 응답: {raw_text}")

            parsed = json.loads(raw_text)
            verdict = Verdict(parsed.get("verdict", "maybe"))
            
            return AnalyzeResponse(
                verdict=verdict,
                verdict_label=_verdict_label(verdict),
                reason=parsed.get("reason", "판단 불가"),
                caution=parsed.get("caution"),
                recommendation=parsed.get("recommendation"),
                caption=caption,
                history_id=None,
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"🔥 Groq 진짜 에러 원인: {e.response.text}")
            return AnalyzeResponse(
                verdict=Verdict.maybe,
                verdict_label="고민해요",
                reason="에러 상세 파악 중입니다!",
                caption=caption,
                history_id=None
            )
        except Exception as e:
            logger.error(f"Groq 기타 오류 실패: {e}")
            return AnalyzeResponse(
                verdict=Verdict.maybe,
                verdict_label="고민해요",
                reason="잔소리 하다가 지쳤다. 쫌따 다시 물어봐!",
                caption=caption,
                history_id=None
            )