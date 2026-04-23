import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status, Depends
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent_sdk.logging import configure_logging
from agent_sdk.context import request_id_var, user_id_var
from agent_sdk.metrics import metrics_response
from agent_sdk.server.streaming import StreamingMathFixer, _fix_math_delimiters
from agents.agent import create_agent, run_query, create_stream, save_memory
from database.mongo import MongoDB
from a2a_service.server import create_a2a_app
from tools.resume_parser import parse_resume_file
from tools.research_client import _current_user_id, _current_request_id

configure_logging("agent_interview_prep")
logger = logging.getLogger("agent_interview_prep.api")
limiter = Limiter(key_func=get_remote_address)

_GITHUB_REPO_RE = re.compile(
    r'^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(/[^\s]*)?$'
)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_RESUME_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("INTERNAL_API_KEY"):
        logger.warning("INTERNAL_API_KEY is not set — internal API is unprotected. Set this in production.")
    agent = create_agent()
    try:
        await agent._ensure_initialized()
        if getattr(agent, '_degraded', False):
            logger.warning("Agent started in DEGRADED mode — MCP tools unavailable")
        else:
            logger.info("MCP servers connected, agent ready")
    except Exception as e:
        logger.error("Agent initialization failed (continuing without MCP): %s", e)
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Internal-API-Key", "X-User-Id", "X-Request-ID"],
)

_PUBLIC_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/a2a/.well-known/agent.json"}

@app.middleware("http")
async def inject_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    tok_r = request_id_var.set(request_id)
    tok_u = user_id_var.set(request.headers.get("X-User-Id"))
    response = await call_next(request)
    request_id_var.reset(tok_r)
    user_id_var.reset(tok_u)
    response.headers["X-Request-ID"] = request_id
    return response

@app.middleware("http")
async def verify_internal_key(request: Request, call_next):
    if request.url.path not in _PUBLIC_PATHS:
        expected = os.getenv("INTERNAL_API_KEY")
        if expected and request.headers.get("X-Internal-API-Key") != expected:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Unauthorized internal access"})
    return await call_next(request)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Mount the A2A server as a sub-application
a2a_app = create_a2a_app()
app.mount("/a2a", a2a_app.build())


# ── Request/Response models ──

class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
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

_codebase_locks_cache: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

def _get_codebase_lock(session_id: str) -> asyncio.Lock:
    return _codebase_locks_cache[session_id]


# ── Standard agent endpoints ──

@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(body: AskRequest, request: Request):
    user_id = request.headers.get("X-User-Id") or None
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    if not is_new and user_id:
        owned_history = await MongoDB.get_history(session_id, user_id=user_id)
        if not owned_history:
            any_history = await MongoDB.get_history(session_id)
            if any_history:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

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
        plan=result.get("plan"),
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
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    if not is_new and user_id:
        owned_history = await MongoDB.get_history(session_id, user_id=user_id)
        if not owned_history:
            any_history = await MongoDB.get_history(session_id)
            if any_history:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    logger.info("POST /ask/stream — session='%s', user='%s', query='%s'",
                session_id, user_id or "anonymous", body.query[:100])

    raw_stream = await create_stream(
        body.query, session_id=session_id,
        response_format=body.response_format, model_id=body.model_id,
        user_id=user_id
    )
    stream = StreamingMathFixer(raw_stream)

    _STREAM_TIMEOUT = float(os.getenv("STREAM_TIMEOUT_SECONDS", "300"))
    _incoming_request_id = request.headers.get("X-Request-ID")

    async def event_stream():
        # Set the tokens INSIDE the streaming context generator
        _uid_token = _current_user_id.set(user_id)
        _rid_token = _current_request_id.set(_incoming_request_id)

        full_response = []
        queue = asyncio.Queue(maxsize=100)
        _PROGRESS_PREFIX = "__PROGRESS__:"
        _HEARTBEAT_INTERVAL = 15.0

        async def heartbeat_worker():
            try:
                while True:
                    await asyncio.sleep(_HEARTBEAT_INTERVAL)
                    await queue.put(f": heartbeat {int(asyncio.get_running_loop().time())}\n\n")
            except asyncio.CancelledError:
                pass

        async def agent_worker():
            try:
                async with asyncio.timeout(_STREAM_TIMEOUT):
                    async for chunk in stream:
                        try:
                            await asyncio.wait_for(queue.put(chunk), timeout=30.0)
                        except asyncio.TimeoutError:
                            logger.warning("Stream queue full for session='%s' — client likely disconnected", session_id)
                            return
            except TimeoutError:
                logger.error("Stream producer timed out after %.0fs", _STREAM_TIMEOUT)
                await queue.put(f"__ERROR__:Response timed out after {_STREAM_TIMEOUT:.0f} seconds.")
            except Exception as exc:
                logger.error("Stream producer failed: %s", exc)
                await queue.put("__ERROR__:An internal error occurred while generating the response.")
            finally:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        heartbeat_task = asyncio.create_task(heartbeat_worker())
        agent_task = asyncio.create_task(agent_worker())

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break

                if isinstance(chunk, str):
                    if chunk.startswith(": heartbeat"):
                        yield chunk
                    elif chunk.startswith(_PROGRESS_PREFIX):
                        phase_label = chunk[len(_PROGRESS_PREFIX):]
                        yield f"event: progress\ndata: {json.dumps({'phase': phase_label})}\n\n"
                    elif chunk.startswith("__ERROR__:"):
                        error_msg = chunk[len("__ERROR__:"):]
                        yield f"event: error\ndata: {json.dumps({'message': error_msg})}\n\n"
                        fallback = f"\n\n[{error_msg}]"
                        yield f"data: {json.dumps({'text': fallback})}\n\n"
                        full_response.append(fallback)
                    else:
                        full_response.append(chunk)
                        yield f"data: {json.dumps({'text': chunk})}\n\n"

            response_text = "".join(full_response)
            if not response_text.strip():
                fallback = "Sorry, the model returned an empty response. Please try again."
                yield f"data: {json.dumps({'text': fallback})}\n\n"
                response_text = fallback

            try:
                save_memory(user_id=user_id or session_id, query=body.query, response=response_text)
                await MongoDB.save_conversation(
                    session_id=session_id, query=body.query, response=response_text,
                    steps=raw_stream.steps if hasattr(raw_stream, 'steps') else [],
                    user_id=user_id, plan=raw_stream.plan if hasattr(raw_stream, 'plan') else None,
                )
            except Exception as e:
                logger.error("Failed to save conversation: %s", e)

            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            
        finally:
            heartbeat_task.cancel()
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            yield "data: [DONE]\n\n"
            # Safely reset the tokens within the correct context block
            _current_user_id.reset(_uid_token)
            _current_request_id.reset(_rid_token)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history/user/me", response_model=HistoryResponse)
@limiter.limit("60/minute")
async def get_history_by_user(request: Request):
    user_id = request.headers.get("X-User-Id") or None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    logger.info("GET /history/user/me — user='%s'", user_id)
    history = await MongoDB.get_history_by_user(user_id)
    return HistoryResponse(session_id=user_id, history=history)


@app.get("/history/{session_id}", response_model=HistoryResponse)
@limiter.limit("60/minute")
async def get_history(request: Request, session_id: str):
    user_id = request.headers.get("X-User-Id") or None
    logger.info("GET /history — session='%s'", session_id)
    history = await MongoDB.get_history(session_id, user_id=user_id)
    logger.info("Returning %d history entries for session='%s'", len(history), session_id)
    return HistoryResponse(session_id=session_id, history=history)


class SessionsHistoryRequest(BaseModel):
    session_ids: list[str]

@app.post("/history/sessions")
@limiter.limit("30/minute")
async def get_history_by_sessions(request: Request, body: SessionsHistoryRequest):
    user_id = request.headers.get("X-User-Id") or None
    safe_ids = [s for s in body.session_ids[:20] if isinstance(s, str) and re.match(r'^[a-zA-Z0-9\-]{1,64}$', s)]
    logger.info("POST /history/sessions — %d session(s)", len(safe_ids))
    history = await MongoDB.get_history_by_sessions(safe_ids, user_id=user_id)
    return {"history": history}


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

        if len(file_bytes) > MAX_RESUME_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({len(file_bytes) // (1024 * 1024)} MB) exceeds the 10 MB limit.",
            )

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
            raise HTTPException(status_code=422, detail="Failed to parse the uploaded file. Ensure it is a valid PDF or DOCX.")
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
    if not _GITHUB_REPO_RE.match(github_url):
        raise HTTPException(status_code=400, detail="Only public GitHub repository URLs are supported (https://github.com/owner/repo).")

    logger.info("POST /upload/codebase — session='%s', url='%s'", session_id, github_url)

    async with _get_codebase_lock(session_id):
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


@app.get("/metrics")
async def metrics_endpoint():
    content, content_type = metrics_response()
    return Response(content=content, media_type=content_type)


# ── Mock interview score endpoints ──

class MockScoreRequest(BaseModel):
    session_id: str
    question: str
    topic: str
    accuracy: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    depth: int = Field(ge=1, le=10)
    star: int | None = Field(default=None, ge=1, le=10, description="STAR score for behavioral questions")
    notes: str = ""


@app.post("/scores")
@limiter.limit("60/minute")
async def record_score(body: MockScoreRequest, request: Request):
    """Record a mock interview question score for a session."""
    user_id = request.headers.get("X-User-Id") or None
    await MongoDB.save_score(
        user_id=user_id,
        session_id=body.session_id,
        question=body.question,
        topic=body.topic,
        accuracy=body.accuracy,
        clarity=body.clarity,
        depth=body.depth,
        star=body.star,
        notes=body.notes,
    )
    return {"success": True}


@app.get("/scores/{session_id}")
@limiter.limit("60/minute")
async def get_scores(session_id: str, request: Request):
    """Retrieve all mock interview scores for a session."""
    user_id = request.headers.get("X-User-Id") or None
    scores = await MongoDB.get_scores(session_id, user_id=user_id)
    return {"session_id": session_id, "scores": scores}


@app.get("/scores/user/me")
@limiter.limit("60/minute")
async def get_user_scores(request: Request):
    """Retrieve all mock interview scores for the current user across sessions."""
    user_id = request.headers.get("X-User-Id") or None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    scores = await MongoDB.get_user_scores(user_id)
    return {"user_id": user_id, "scores": scores}


# ── Shareable notes endpoint ──

class ShareNoteRequest(BaseModel):
    file_id: str


@app.post("/notes/share")
@limiter.limit("20/minute")
async def create_share_token(body: ShareNoteRequest, request: Request):
    """Create a shareable public token for a notes file."""
    user_id = request.headers.get("X-User-Id") or None
    file_meta = await MongoDB.get_file(body.file_id)
    if not file_meta:
        raise HTTPException(status_code=404, detail="File not found")
    share_token = await MongoDB.create_share_token(body.file_id, user_id=user_id)
    base = (os.getenv("BACKEND_URL") or os.getenv("PUBLIC_URL") or "").rstrip("/")
    return {"share_token": share_token, "share_url": f"{base}/notes/shared/{share_token}"}


@app.get("/notes/shared/{token}")
async def download_shared_note(token: str):
    """Download a shared note using a public token (no auth required)."""
    file_id = await MongoDB.resolve_share_token(token)
    if not file_id:
        raise HTTPException(status_code=404, detail="Shared link not found or expired")
    result = await MongoDB.retrieve_file(file_id)
    if not result:
        raise HTTPException(status_code=404, detail="File not found")
    data, meta = result
    filename = meta.get("filename", "notes")
    media_type = "application/pdf" if filename.endswith(".pdf") else "text/markdown"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
        await MongoDB.get_client().admin.command("ping")
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