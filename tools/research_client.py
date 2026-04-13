import asyncio
import logging
import os

import httpx
from langchain_core.tools import tool

from agent_sdk.context import user_id_var as _current_user_id, request_id_var as _current_request_id

logger = logging.getLogger("agent_interview_prep.tools.research_client")

RESEARCH_AGENT_URL = os.getenv("RESEARCH_AGENT_URL", "http://localhost:9002")

_STUDY_NOTES_CONTEXT = (
    "\n\n[RESEARCH CONTEXT: This query is for generating comprehensive study notes. "
    "Use your research tools (arXiv papers, Tavily web search, vector DB) to gather "
    "thorough, in-depth information. Do NOT answer from general knowledge alone.]"
)


@tool
async def research_topic(query: str) -> str:
    """Delegate a research query to the Research Agent.

    ONLY use this for topics that are niche, recent (post-2023), or require academic paper
    summaries. Do NOT call it for well-established concepts (e.g., dropout, backpropagation,
    attention mechanisms, transformers, common algorithms, standard data structures, classic
    system design patterns) — answer those directly from your own knowledge.
    Call this tool at most ONCE per user turn.

    Args:
        query: The research query to send to the research agent.
    """
    user_id = _current_user_id.get()
    request_id = _current_request_id.get()
    enriched_query = query + _STUDY_NOTES_CONTEXT
    headers: dict[str, str] = {}
    if user_id:
        headers["X-User-Id"] = user_id
    if request_id:
        headers["X-Request-ID"] = request_id
    
    internal_key = os.getenv("INTERNAL_API_KEY")
    if internal_key:
        headers["X-Internal-API-Key"] = internal_key

    logger.info("Delegating research query to %s: '%s' (user='%s')",
                RESEARCH_AGENT_URL, query[:100], user_id or "anonymous")

    _MAX_RETRIES = 2
    _RETRY_DELAY = 2  # linear backoff — avoids compounding delay on flaky agent
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{RESEARCH_AGENT_URL}/ask",
                    json={"query": enriched_query},
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                result = data.get("response", "")
                logger.info("Research agent returned %d chars", len(result))
                return result

        except httpx.TimeoutException:
            logger.error("Research agent timed out for query: '%s'", query[:100])
            return "The research agent timed out. Please try a more specific query."
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or attempt == _MAX_RETRIES - 1:
                logger.error("Research agent returned %d: %s", e.response.status_code, e)
                return f"Research agent error (HTTP {e.response.status_code}). It may be unavailable."
            logger.warning(
                "Research agent HTTP %d (attempt %d/%d) — retrying in %ds",
                e.response.status_code, attempt + 1, _MAX_RETRIES, _RETRY_DELAY,
            )
            await asyncio.sleep(_RETRY_DELAY)
        except httpx.ConnectError:
            if attempt == _MAX_RETRIES - 1:
                logger.error("Cannot connect to research agent at %s", RESEARCH_AGENT_URL)
                return (
                    "Cannot connect to the research agent. "
                    "Please ensure it is running and try again."
                )
            logger.warning(
                "Research agent unreachable (attempt %d/%d) — retrying in %ds",
                attempt + 1, _MAX_RETRIES, _RETRY_DELAY,
            )
            await asyncio.sleep(_RETRY_DELAY)
    # Unreachable, but satisfies type checker
    return "Research agent unavailable after retries."
