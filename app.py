import asyncio
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.agent import create_agent, run_query, _build_enriched_prompt
from database.mongo import MongoDB
from a2a_service.server import create_a2a_app
from tools.resume_parser import parse_resume_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_interview_prep.api")

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
    yield
    await agent._disconnect_mcp()
    await MongoDB.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Interview Prep Agent API",
    description="AI-powered interview preparation assistant with resume analysis and study material generation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ── Standard agent endpoints ──

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    is_new = request.session_id is None
    session_id = request.session_id or MongoDB.generate_session_id()

    logger.info("POST /ask — session='%s' (%s), query='%s'",
                session_id, "new" if is_new else "existing", request.query[:100])

    result = await run_query(request.query, session_id=session_id,
                             response_format=request.response_format, model_id=request.model_id)
    response = result["response"]
    steps = result["steps"]

    await MongoDB.save_conversation(
        session_id=session_id,
        query=request.query,
        response=response,
        steps=steps,
    )

    logger.info("POST /ask complete — session='%s', response length: %d chars, tool_calls: %d",
                session_id, len(response),
                sum(1 for s in steps if s.get("action") == "tool_call"))

    return AskResponse(
        session_id=session_id,
        query=request.query,
        response=response,
    )


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """Stream the agent's response as Server-Sent Events (SSE)."""
    session_id = request.session_id or MongoDB.generate_session_id()
    logger.info("POST /ask/stream — session='%s', query='%s'", session_id, request.query[:100])

    enriched_prompt = await _build_enriched_prompt(
        session_id, request.query, response_format=request.response_format
    )
    agent = create_agent()
    stream = agent.astream(
        request.query, session_id=session_id,
        system_prompt=enriched_prompt, model_id=request.model_id
    )

    async def event_stream():
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

        from agents.agent import save_memory
        save_memory(user_id=session_id, query=request.query, response=response_text)

        await MongoDB.save_conversation(
            session_id=session_id,
            query=request.query,
            response=response_text,
            steps=stream.steps if hasattr(stream, 'steps') else [],
        )

        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    logger.info("GET /history — session='%s'", session_id)
    history = await MongoDB.get_history(session_id)
    logger.info("Returning %d history entries for session='%s'", len(history), session_id)
    return HistoryResponse(session_id=session_id, history=history)


# ── File upload/download endpoints ──

@app.post("/upload/resume", response_model=UploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Upload a resume file (PDF or DOCX) for analysis."""
    # Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only PDF and DOCX files are accepted.",
        )

    file_id = uuid.uuid4().hex
    file_bytes = await file.read()

    logger.info("Received resume upload — session='%s', file_id='%s', size=%d bytes",
                session_id, file_id, len(file_bytes))

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
            created_at=f.get("created_at", "").isoformat() if f.get("created_at") else None,
        )
        for f in files
    ]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-interview-prep"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 9003))
    uvicorn.run(app, host="0.0.0.0", port=port)
