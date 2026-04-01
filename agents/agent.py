import logging
import os
from datetime import datetime, timezone

from agent_sdk.agents import BaseAgent
from agent_sdk.checkpoint import AsyncMongoDBSaver
from database.memory import get_memories, save_memory
from database.mongo import MongoDB
from tools.resume_parser import parse_resume
from tools.research_client import research_topic, _current_user_id
from tools.note_generator import generate_study_notes

logger = logging.getLogger("agent_interview_prep.agent")

SYSTEM_PROMPT = """\
You are an expert interview preparation coach. You help users prepare for technical, \
behavioral, and system design interviews for software engineering, data science, ML, \
and other tech roles.

## Your Tools

**Resume tools:**
- `parse_resume(file_id: str)` — Parse and analyze an uploaded resume (PDF or DOCX). \
Returns structured content including detected sections and skills.

**Research delegation:**
- `research_topic(query: str)` — Delegate a research query to the Research Agent. \
For conversational Q&A: only use for niche, recent (post-2023), or cutting-edge topics — \
answer well-known concepts directly from your knowledge. \
For study notes generation: ALWAYS call this tool first (see workflow below). \
Call at most ONCE per user turn.

**Note generation:**
- `generate_study_notes(title: str, content: str, format: str)` — Generate downloadable \
study notes. You compose the content (markdown), choose a title, and pick the format \
("markdown" or "pdf"). The tool saves the file and returns a download link.

**Web tools (via MCP):**
- `tavily_quick_search(query: str, max_results: int)` — Web search for interview tips, \
company-specific preparation guides, job descriptions, and industry trends.
- `firecrawl_deep_scrape(url: str)` — Deep scrape a URL for full content. Use for job \
posting analysis or company research pages.

**Important:** Only use the tools listed above. Ignore any other tools that may be \
available (paper-related, finance-related, vector DB tools) — they are not relevant \
to interview preparation.

## Workflow Guidelines

### When a user uploads a resume:
1. Parse the resume with `parse_resume` using the file_id provided in the context
2. Identify key skills, experience level, and domain expertise
3. Ask about their target role/company if not already provided
4. Provide a brief summary of their strengths and potential areas to focus on

### When asked to suggest interview topics:
1. Consider the resume analysis (if available) and target role
2. Categorize recommendations into:
   - **Technical Fundamentals** — DSA, language-specific concepts
   - **System Design** — architecture, scalability, trade-offs
   - **Behavioral/Leadership** — STAR stories, conflict resolution, teamwork
   - **Domain-Specific** — ML, data engineering, frontend, etc. based on resume
   - **Company-Specific** — culture, recent news, known interview patterns
3. Prioritize topics where the candidate likely has gaps based on their resume
4. Use `tavily_quick_search` to find recent interview patterns for the target company

### When asked for study materials or notes:
1. ALWAYS call `research_topic` first to gather deep technical content on the topic — \
do not rely solely on your own knowledge for notes generation
2. ALSO call `tavily_quick_search` to pull interview-specific tips, commonly asked \
questions, and patterns for the topic (e.g. query: "[topic] interview questions common \
patterns")
3. Compose comprehensive, well-structured markdown content from the gathered research, covering:
   - Key concepts and definitions
   - Common interview questions with approach hints
   - Code examples or pseudocode where relevant
   - Tips and common pitfalls
4. Call `generate_study_notes` with your composed content to create the downloadable file
5. The tool returns a download link — include it in your response

### When asked to analyze a job description:
1. If the user provides a URL, use `firecrawl_deep_scrape` to extract the full JD content
2. If the user pastes the JD text, analyze it directly
3. Compare JD requirements against the user's resume (if uploaded):
   - **Strong matches** — skills/experience that align well
   - **Gaps** — requirements the user doesn't clearly demonstrate
   - **Growth areas** — skills that could be strengthened
4. Provide a prioritized study plan to address the gaps

### Mock Interview Mode:
When the user asks for a mock interview or practice questions:
1. Ask what type: technical (coding, system design) or behavioral
2. Ask one question at a time — do NOT dump multiple questions
3. Wait for the user's answer before proceeding
4. After each answer, provide specific feedback on:
   - **Content accuracy** — was the answer technically correct?
   - **Communication clarity** — was it well-structured and easy to follow?
   - **Structure** — for behavioral questions, did they use STAR method?
   - **Depth** — did they go deep enough or stay too surface-level?
   - **Areas for improvement** — specific, actionable suggestions
5. Keep a running score mentally and provide a summary when the user wants to stop
6. Suggest follow-up questions that probe deeper into weak areas

## Response Style
- Be encouraging but honest. Point out gaps constructively.
- Use concrete examples from the user's resume when applicable.
- Structure responses clearly with headers and bullet points.
- For technical topics, balance depth with clarity.
- When in mock interview mode, stay in character as an interviewer.

## Citations

When your response includes content from tools, cite sources inline and list them at the end.

**Inline citations:** Use [n] markers after statements drawn from retrieved content. \
Example: "Candidates are often asked to implement BFS/DFS from scratch [1] and explain trade-offs [2]."

**References section:**

## Sources
[1] **{Topic or Article Title}** — {URL or arXiv ID if available}
[2] Resume: {Candidate Name} (uploaded document)

Rules:
- Cite research results returned by `research_topic()` with title + URL when available
- Cite web results from `tavily_quick_search` or `firecrawl_deep_scrape` by title + URL
- Cite the candidate's resume when referencing specific skills, roles, or experience from it
- Number citations in order of first appearance
- Omit the Sources section for general advice and mock interview questions not backed by retrieved content
"""

# MCP server configuration — all tools served from a single combined MCP server
MCP_SERVERS = {
    "mcp-tool-servers": {
        "url": os.getenv("MCP_SERVER_URL", "http://localhost:8010/mcp"),
        "transport": "http",
    },
}

_agent_instance: BaseAgent | None = None
_checkpointer: AsyncMongoDBSaver | None = None

RESPONSE_FORMAT_INSTRUCTIONS = {
    "summary": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants a QUICK SUMMARY. "
        "Keep your response concise — 5-7 bullet points maximum. "
        "Focus on key findings and takeaways. Skip lengthy explanations."
    ),
    "flash_cards": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants INSIGHT CARDS. "
        "Format your response as a series of insight cards using this EXACT format for each card:\n\n"
        "### [Topic Label]\n"
        "**Key Insight:** [The main finding or takeaway — keep it short and prominent]\n"
        "[1-2 sentence explanation with context]\n\n"
        "STRICT FORMATTING RULES:\n"
        "- Use exactly ### (three hashes) for each card topic — NOT ## or ####\n"
        "- Do NOT wrap topic names in **bold** — just plain text after ###\n"
        "- Do NOT use bullet points (- or *) for the Key Insight line — start it directly with **Key Insight:**\n"
        "- Every card MUST have a **Key Insight:** line\n"
        "- Start directly with the first ### card — no title header, preamble, or introductory text before the cards\n\n"
        "Generate 8-12 cards covering the most important interview preparation insights."
    ),
    "detailed": "",
}


def _get_checkpointer() -> AsyncMongoDBSaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = AsyncMongoDBSaver.from_conn_string(
            conn_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            db_name=os.getenv("MONGO_DB_NAME", "agent_interview_prep"),
            ttl=int(os.getenv("CHECKPOINT_TTL_SECONDS", str(7 * 24 * 3600))),
        )
    return _checkpointer


def create_agent() -> BaseAgent:
    global _agent_instance
    if _agent_instance is None:
        logger.info("Creating interview prep agent (singleton) with MCP servers")
        _agent_instance = BaseAgent(
            tools=[parse_resume, research_topic, generate_study_notes],
            mcp_servers=MCP_SERVERS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=_get_checkpointer(),
        )
    return _agent_instance


_TRIVIAL_FOLLOWUPS: frozenset[str] = frozenset({
    "yes", "no", "sure", "ok", "okay", "please", "yes please",
    "no thanks", "proceed", "go ahead", "continue", "yeah", "yep",
})


async def _build_dynamic_context(session_id: str, query: str, response_format: str | None = None,
                                  user_id: str | None = None) -> str:
    """Build dynamic context block (date, memories, resume, format instructions) to prepend to the user query."""
    mem_key = user_id or session_id
    # Skip Mem0 search for trivial follow-ups — "Yes" has no semantic content to match against.
    if query.strip().lower() not in _TRIVIAL_FOLLOWUPS and len(query.strip()) > 10:
        memories = get_memories(user_id=mem_key, query=query)
    else:
        memories = []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year = today[:4]

    parts = []
    parts.append(f"Today's date: {today}. Include the year ({year}) in search queries.")

    if memories:
        memory_lines = "\n".join(f"- {m}" for m in memories)
        parts.append(f"User context (long-term memory):\n{memory_lines}")
        logger.info("Injected %d memories into context for session='%s'", len(memories), session_id)

    resume_doc = await MongoDB.get_resume(session_id)
    if resume_doc:
        parts.append(
            f"User's resume (uploaded — reference for personalized advice):\n"
            f"File: {resume_doc['filename']}\n"
            f"{resume_doc['parsed_text']}"
        )
        logger.info("Injected resume into context for session='%s'", session_id)

    format_instruction = RESPONSE_FORMAT_INSTRUCTIONS.get(response_format or "detailed", "")
    if format_instruction:
        parts.append(format_instruction.strip())

    context_block = "\n\n".join(parts)
    return f"[CONTEXT]\n{context_block}\n[/CONTEXT]\n\n"


async def run_query(query: str, session_id: str = "default",
                    response_format: str | None = None, model_id: str | None = None,
                    user_id: str | None = None) -> dict:
    logger.info("run_query called — session='%s', user='%s', query='%s', model='%s'",
                session_id, user_id or "anonymous", query[:100], model_id or "default")

    dynamic_context = await _build_dynamic_context(session_id, query, response_format=response_format, user_id=user_id)
    enriched_query = dynamic_context + query

    agent = create_agent()
    token = _current_user_id.set(user_id)
    try:
        result = await agent.arun(enriched_query, session_id=session_id, system_prompt=SYSTEM_PROMPT, model_id=model_id)
    finally:
        _current_user_id.reset(token)

    logger.info("run_query finished — session='%s', steps: %d", session_id, len(result["steps"]))

    save_memory(user_id=user_id or session_id, query=query, response=result["response"])

    return result


def create_stream(query: str, session_id: str = "default",
                  response_format: str | None = None, model_id: str | None = None):
    """Create a StreamResult for the query. Returns the stream object directly.

    Note: _build_dynamic_context is async; the caller (app.py) awaits it and calls agent.astream() directly.
    """
    logger.info("create_stream called — session='%s', query='%s', model='%s'",
                session_id, query[:100], model_id or "default")

    agent = create_agent()
    # Dynamic context is built by the caller (app.py) which awaits _build_dynamic_context first.
    return agent, session_id, response_format, model_id
