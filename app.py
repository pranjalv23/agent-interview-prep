import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status, Depends
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agents.agent import create_agent, run_query, _build_dynamic_context, SYSTEM_PROMPT, save_memory
from database.mongo import MongoDB
from a2a_service.server import create_a2a_app
from tools.resume_parser import parse_resume_file
from tools.research_client import _current_user_id, _current_request_id

def _fix_math_delimiters(text: str) -> str:
    r"""Convert LaTeX parenthesis delimiters to Markdown math notation.

    \[...\]  →  $$...$$   (display math — must run before inline to avoid overlap)
    \(...\)  →  $...$     (inline math)
    """
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$$\n{m.group(1)}\n$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    return text


class StreamingMathFixer:
    """Wraps an async chunk stream and converts \\(...\\) / \\[...\\] math delimiters on-the-fly.

    Non-math text is yielded immediately so the streaming feel is preserved.
    Math sections are buffered only until their closing delimiter arrives,
    then emitted with the correct $...$ / $$...$$ notation.
    """

    def __init__(self, source):
        self._source = source

    async def __aiter__(self):
        buffer = ""
        in_math = False   # inside \( ... \)
        in_block = False  # inside \[ ... \]

        async for chunk in self._source:
            buffer += chunk
            result = ""

            while buffer:
                if not in_math and not in_block:
                    bi = buffer.find("\\[")
                    ii = buffer.find("\\(")
                    if bi == -1 and ii == -1:
                        # Only buffer if the chunk ends with a backslash that might start a delimiter
                        if buffer.endswith("\\"):
                            if len(buffer) > 1:
                                result += buffer[:-1]
                                buffer = "\\"
                            break
                        else:
                            result += buffer
                            buffer = ""
                            break
                    if bi == -1 or (ii != -1 and ii < bi):
                        result += buffer[:ii]
                        buffer = buffer[ii + 2:]
                        in_math = True
                    else:
                        result += buffer[:bi]
                        buffer = buffer[bi + 2:]
                        in_block = True
                elif in_math:
                    close = buffer.find("\\)")
                    if close == -1:
                        break
                    result += "$" + buffer[:close] + "$"
                    buffer = buffer[close + 2:]
                    in_math = False
                else:  # in_block
                    close = buffer.find("\\]")
                    if close == -1:
                        break
                    result += "$$\n" + buffer[:close] + "\n$$"
                    buffer = buffer[close + 2:]
                    in_block = False

            if result:
                yield result

        # Flush any remaining buffer after the source stream ends
        if buffer:
            if in_math:
                yield "$" + buffer + "$"
            elif in_block:
                yield "$$\n" + buffer + "\n$$"
            else:
                yield buffer

    @property
    def steps(self):
        return self._source.steps


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)

_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logging.root.addHandler(_handler)
logger = logging.getLogger("agent_interview_prep.api")
limiter = Limiter(key_func=get_remote_address)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = create_agent()
    await agent._ensure_initialized()
    logger.info("MCP servers connected, agent ready")
    await MongoDB.ensure_indexes()
    yield
    await agent._disconnect_mcp()
    await MongoDB.close()
    logger.info("Shutdown complete")


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom handler for RateLimitExceeded to return a JSON response."""
    return Response(
        content=json.dumps({"detail": "Rate limit exceeded. Please try again later."}),
        status_code=429,
        media_type="application/json",
    )

app = FastAPI(
    title="Interview Prep Agent API",
    description="AI-powered interview preparation assistant with resume analysis and study material generation.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Mount the A2A server as a sub-application
a2a_app = create_a2a_app()
app.mount("/a2a", a2a_app.build())


# ── Request/Response models ──

class AskRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None

    model_config = {"json_schema_extra": {"examples": [{"query": "", "session_id": None, "response_format": "detailed", "model_id": None}]}}




class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    parsed_preview: str


class FileListItem(BaseModel):
    file_id: str
    filename: str
    file_type: str
    created_at: str | None = None


class CodebaseUploadResponse(BaseModel):
    repo_name: str
    owner: str
    total_files: int
    language: str
    preview: str


# ── Standard agent endpoints ──

@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(body: AskRequest, request: Request):
    user_id = request.headers.get("X-User-Id") or None
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    logger.info("POST /ask — session='%s' (%s), user='%s', query='%s'",
                session_id, "new" if is_new else "existing", user_id or "anonymous", body.query[:100])

    _uid_token = _current_user_id.set(user_id)
    _rid_token = _current_request_id.set(request.headers.get("X-Request-ID"))
    try:
        result = await run_query(body.query, session_id=session_id,
                                 response_format=body.response_format, model_id=body.model_id,
                                 user_id=user_id)
    finally:
        _current_user_id.reset(_uid_token)
        _current_request_id.reset(_rid_token)
    response = _fix_math_delimiters(result["response"])
    steps = result["steps"]

    await MongoDB.save_conversation(
        session_id=session_id,
        query=body.query,
        response=response,
        steps=steps,
        user_id=user_id,
    )

    logger.info("POST /ask complete — session='%s', response length: %d chars, tool_calls: %d",
                session_id, len(response),
                sum(1 for s in steps if s.get("action") == "tool_call"))

    return AskResponse(
        session_id=session_id,
        query=body.query,
        response=response,
    )


@app.post("/ask/stream")
@limiter.limit("30/minute")
async def ask_stream(body: AskRequest, request: Request):
    """Stream the agent's response as Server-Sent Events (SSE)."""
    user_id = request.headers.get("X-User-Id") or None
    session_id = body.session_id or MongoDB.generate_session_id()
    logger.info("POST /ask/stream — session='%s', user='%s', query='%s'",
                session_id, user_id or "anonymous", body.query[:100])

    dynamic_context = await _build_dynamic_context(
        session_id, body.query, response_format=body.response_format, user_id=user_id
    )
    enriched_query = dynamic_context + body.query
    agent = create_agent()
    
    stream = StreamingMathFixer(agent.astream(
        enriched_query, session_id=session_id,
        system_prompt=SYSTEM_PROMPT, model_id=body.model_id
    ))

    _incoming_request_id = request.headers.get("X-Request-ID")
    async def event_stream():
        # Set the tokens INSIDE the streaming context generator
        _uid_token = _current_user_id.set(user_id)
        _rid_token = _current_request_id.set(_incoming_request_id)

        try:
            full_response = []
            queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

            async def _stream_producer():
                try:
                    async for chunk in stream:
                        await queue.put(("chunk", chunk))
                    await queue.put(("done", None))
                except Exception as exc:
                    logger.error("Stream producer failed: %s", exc)
                    await queue.put(("error", str(exc)))

            async def _keepalive_producer():
                while True:
                    await asyncio.sleep(15)
                    await queue.put(("keepalive", None))

            producer_task = asyncio.create_task(_stream_producer())
            keepalive_task = asyncio.create_task(_keepalive_producer())

            try:
                while True:
                    kind, data = await queue.get()
                    if kind == "chunk":
                        if isinstance(data, str) and data.startswith("__PROGRESS__:"):
                            phase_label = data[len("__PROGRESS__:"):]
                            yield f"event: progress\ndata: {json.dumps({'phase': phase_label})}\n\n"
                        else:
                            full_response.append(data)
                            yield f"data: {json.dumps({'text': data})}\n\n"
                    elif kind == "keepalive":
                        yield ": keep-alive\n\n"
                    elif kind == "error":
                        error_msg = "An error occurred processing your request. Please try again or switch to a different model."
                        yield f"data: {json.dumps({'text': error_msg})}\n\n"
                        break
                    elif kind == "done":
                        break
            finally:
                keepalive_task.cancel()
                try:
                    await keepalive_task
                except asyncio.CancelledError:
                    pass
                await producer_task

            response_text = "".join(full_response)

            if not response_text.strip():
                fallback = "Sorry, the model returned an empty response. Please try again or switch to a different model."
                yield f"data: {json.dumps({'text': fallback})}\n\n"
                response_text = fallback

            try:
                save_memory(user_id=user_id or session_id, query=body.query, response=response_text)

                await MongoDB.save_conversation(
                    session_id=session_id,
                    query=body.query,
                    response=response_text,
                    steps=stream.steps if hasattr(stream, 'steps') else [],
                    user_id=user_id,
                )
            except Exception as e:
                logger.error("Failed to save memory/conversation: %s", e)

            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            
        finally:
            yield "data: [DONE]\n\n"
            # Safely reset the tokens within the correct context block
            _current_user_id.reset(_uid_token)
            _current_request_id.reset(_rid_token)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history/user/me", response_model=HistoryResponse)
async def get_history_by_user(http_request: Request):
    user_id = http_request.headers.get("X-User-Id") or None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    logger.info("GET /history/user/me — user='%s'", user_id)
    history = await MongoDB.get_history_by_user(user_id)
    return HistoryResponse(session_id=user_id, history=history)


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    logger.info("GET /history — session='%s'", session_id)
    history = await MongoDB.get_history(session_id)
    logger.info("Returning %d history entries for session='%s'", len(history), session_id)
    return HistoryResponse(session_id=session_id, history=history)


# ── File upload/download endpoints ──

@app.post("/upload/resume", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Upload a resume file (PDF or DOCX) for analysis."""
    user_id = request.headers.get("X-User-Id") or None
    _uid_token = _current_user_id.set(user_id)
    _rid_token = _current_request_id.set(request.headers.get("X-Request-ID"))

    try:
        # Validate file type
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Only PDF and DOCX files are accepted.",
            )

        file_id = uuid.uuid4().hex
        file_bytes = await file.read()

        logger.info("Received resume upload — session='%s', user='%s', file_id='%s', size=%d bytes",
                    session_id, user_id or "anonymous", file_id, len(file_bytes))

        # Write to a temp file for parsing (pymupdf/python-docx need a file path)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            parsed = parse_resume_file(tmp_path)
        except Exception as e:
            logger.error("Failed to parse uploaded resume: %s", e)
            raise HTTPException(status_code=422, detail=f"Failed to parse resume: {e}")
        finally:
            os.unlink(tmp_path)

        # Build a readable parsed text for storage
        parsed_parts = []
        if parsed["detected_skills"]:
            parsed_parts.append(f"Detected Skills: {', '.join(parsed['detected_skills'])}")
        for section_name, content_text in parsed["sections"].items():
            if section_name != "other":
                parsed_parts.append(f"[{section_name.title()}]\n{content_text}")
        if "other" in parsed["sections"]:
            parsed_parts.append(f"[Additional]\n{parsed['sections']['other']}")

        parsed_text = "\n\n".join(parsed_parts)

        # Store file bytes in GridFS and metadata in MongoDB
        await MongoDB.store_file(
            file_id=file_id,
            filename=file.filename or f"resume{ext}",
            data=file_bytes,
            file_type="resume",
            session_id=session_id,
        )
        await MongoDB.save_resume(
            session_id=session_id,
            file_id=file_id,
            filename=file.filename or f"resume{ext}",
            parsed_text=parsed_text,
        )

        # Preview: first 500 chars
        preview = parsed_text[:500] + ("..." if len(parsed_text) > 500 else "")

        return UploadResponse(
            file_id=file_id,
            filename=file.filename or f"resume{ext}",
            parsed_preview=preview,
        )
    finally:
        _current_user_id.reset(_uid_token)
        _current_request_id.reset(_rid_token)


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download a generated file (study notes, etc.) from GridFS."""
    result = await MongoDB.retrieve_file(file_id)
    if not result:
        raise HTTPException(status_code=404, detail="File not found.")

    data, meta = result
    filename = meta.get("filename", "download")

    # Determine content type from extension
    if filename.endswith(".pdf"):
        media_type = "application/pdf"
    elif filename.endswith(".md"):
        media_type = "text/markdown"
    else:
        media_type = "application/octet-stream"

    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/files/{session_id}", response_model=list[FileListItem])
async def list_files(session_id: str):
    """List all files (uploads + generated notes) for a session."""
    files = await MongoDB.list_files(session_id)
    return [
        FileListItem(
            file_id=f["file_id"],
            filename=f["filename"],
            file_type=f["file_type"],
            created_at=f.get("created_at").isoformat() if isinstance(f.get("created_at"), datetime) else (f.get("created_at") if f.get("created_at") else None),
        )
        for f in files
    ]


@app.post("/upload/codebase", response_model=CodebaseUploadResponse)
@limiter.limit("10/minute")
async def upload_codebase(
    request: Request,
    github_url: str = Form(...),
    session_id: str = Form(...),
):
    """Fetch a public GitHub repository and store it for codebase interview mode."""
    github_url = github_url.strip()
    if not github_url:
        raise HTTPException(status_code=400, detail="github_url is required.")

    logger.info("POST /upload/codebase — session='%s', url='%s'", session_id, github_url)

    try:
        agent = create_agent()
        await agent._ensure_initialized()
        fetch_tool = agent.tools_by_name.get("fetch_github_repo")
        if fetch_tool is None:
            raise HTTPException(status_code=503, detail="fetch_github_repo tool not available on MCP server")
        raw = await fetch_tool.ainvoke({"repo_url": github_url})
        if isinstance(raw, str) and raw.startswith("Error:"):
            raise ValueError(raw[len("Error:"):].strip())
        codebase_doc = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error fetching GitHub repo '%s': %s", github_url, e)
        raise HTTPException(status_code=500, detail="Failed to fetch repository. Please check the URL and try again.")

    await MongoDB.store_codebase(session_id=session_id, codebase_doc=codebase_doc)

    preview_lines = codebase_doc["summary"].splitlines()[:6]
    preview = "\n".join(preview_lines)

    return CodebaseUploadResponse(
        repo_name=codebase_doc["repo_name"],
        owner=codebase_doc["owner"],
        total_files=codebase_doc["total_files"],
        language=codebase_doc["language"],
        preview=preview,
    )


@app.get("/health")
async def health():
    """Service health check with dependency verification."""
    status_code = 200
    checks = {
        "status": "ok",
        "mongodb": "ok",
        "mem0": "ok" if os.getenv("MEM0_API_KEY") else "unconfigured",
        "service": "agent-interview-prep",
    }

    try:
        # Check MongoDB connectivity
        await MongoDB.db.command("ping")
    except Exception as e:
        logger.error("Health check: MongoDB connection failed: %s", e)
        checks["mongodb"] = "error"
        checks["status"] = "degraded"
        status_code = 503

    return Response(
        content=json.dumps(checks),
        status_code=status_code,
        media_type="application/json",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 9003))
    ssl_certfile = os.getenv("SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("SSL_KEYFILE") or None
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)