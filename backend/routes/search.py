import json
import logging
import os
from typing import List

import httpx
from fastapi import APIRouter
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

# -----------------------------
# LLM fallback (Emergent)
# -----------------------------
# Keep the existing behavior as a fallback only.
_llm_client = AsyncOpenAI(
    api_key=os.environ.get("EMERGENT_LLM_KEY"),
    base_url="https://api.emergent.sh/v1",
)

# -----------------------------
# Response model (keep API contract)
# -----------------------------
class SuggestionsResponse(BaseModel):
    suggestions: List[str]
    query: str


async def _google_autocomplete(q: str, limit: int) -> List[str]:
    """Fetch suggestions from Google's free suggest endpoint (no API key).

    Uses: https://suggestqueries.google.com/complete/search?client=chrome&q=...
    Response example: ["query", ["s1", "s2"...], ...]
    """

    # Use small timeout to avoid hanging the UI
    timeout = httpx.Timeout(3.0, connect=2.0)
    url = "https://suggestqueries.google.com/complete/search"

    headers = {
        # A realistic UA reduces the chance of 429s / odd responses
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        r = await client.get(url, params={"client": "chrome", "q": q})
        r.raise_for_status()

        data = r.json()
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            items = [s for s in data[1] if isinstance(s, str)]
            return items[:limit]

    return []


async def _llm_suggestions(q: str, limit: int) -> List[str]:
    """Fallback to LLM for suggestions (uses EMERGENT_LLM_KEY)."""

    response = await _llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search suggestion assistant. Given a partial search query, "
                    "provide relevant autocomplete suggestions. Return ONLY a JSON array of strings."
                ),
            },
            {"role": "user", "content": f"Provide {limit} search suggestions for: \"{q}\""},
        ],
        max_tokens=200,
        temperature=0.7,
    )

    content = (response.choices[0].message.content or "").strip()

    try:
        if content.startswith("["):
            arr = json.loads(content)
        else:
            start = content.find("[")
            end = content.rfind("]") + 1
            arr = json.loads(content[start:end]) if start >= 0 and end > start else []

        if isinstance(arr, list):
            return [s for s in arr if isinstance(s, str)][:limit]
    except Exception:
        pass

    # last-resort fallback (no credits)
    return [
        f"{q} tutorial",
        f"{q} example",
        f"{q} documentation",
        f"how to {q}",
        f"{q} guide",
    ][:limit]


@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_search_suggestions(q: str, limit: int = 5):
    """Get search suggestions.

    Priority:
      1) Google Suggest (free, no key)
      2) LLM suggestions via EMERGENT (fallback)
      3) Local heuristic fallback
    """

    if not q or len(q.strip()) < 2:
        return SuggestionsResponse(suggestions=[], query=q)

    q = q.strip()
    limit = max(1, min(int(limit), 10))

    # 1) Google suggest
    try:
        suggestions = await _google_autocomplete(q, limit)
        if suggestions:
            return SuggestionsResponse(suggestions=suggestions[:limit], query=q)
    except Exception as e:
        logger.warning(f"Google suggest failed (fallback to LLM): {e}")

    # 2) LLM fallback
    try:
        suggestions = await _llm_suggestions(q, limit)
        return SuggestionsResponse(suggestions=suggestions[:limit], query=q)
    except Exception as e:
        logger.error(f"LLM suggestions failed: {e}")

    # 3) final fallback
    fallback = [
        f"{q} tutorial",
        f"{q} example",
        f"{q} documentation",
        f"how to {q}",
        f"{q} guide",
    ]
    return SuggestionsResponse(suggestions=fallback[:limit], query=q)
