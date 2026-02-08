"""
OpenAI-compatible API server for the RAG system.
Integrates with Open WebUI by exposing /v1/chat/completions and file upload endpoints.
"""

import os
import json
import time
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_system import RobustRAGSystem, RAGConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "9099"))
MODEL_NAME = "rag-assistant"  # Name shown in Open WebUI model list

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Chat Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# RAG system singleton
# ---------------------------------------------------------------------------

rag: Optional[RobustRAGSystem] = None


def get_rag() -> RobustRAGSystem:
    global rag
    if rag is None:
        config = RAGConfig(
            pdf_folder="./documents",
            vector_db_path="./vector_db",
            model_name="gemma3:4b",
            embedding_model="bge-m3:latest",
            debug_mode=False,
            ocr_dpi=300,
            ocr_lang="eng",
        )
        rag = RobustRAGSystem(config)
        rag.start_monitoring()
    return rag


# ---------------------------------------------------------------------------
# Pydantic models for OpenAI-compatible API
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Helper – extract the user's latest question from messages
# ---------------------------------------------------------------------------


def extract_question(messages: list[ChatMessage]) -> str:
    """Return the last user message as the question."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


# ---------------------------------------------------------------------------
# Helper – build SSE chunks (OpenAI streaming format)
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, content: str, finish_reason=None) -> str:
    data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"message": "RAG Chat Assistant API is running"}


# ---- OpenAI-compatible model list (Open WebUI calls this) ----------------

@app.get("/v1/models")
@app.get("/api/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rag-system",
            }
        ],
    }


# ---- Chat completions ----------------------------------------------------

@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    system = get_rag()
    question = extract_question(request.messages)

    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    # ---- Check if the message contains a base64-encoded PDF ---------------
    # Open WebUI sometimes sends file content inline. We won't rely on that
    # for PDFs; use the /v1/files/upload endpoint or drop files in /documents.

    if request.stream:
        return StreamingResponse(
            _stream_response(system, question),
            media_type="text/event-stream",
        )

    # Non-streaming
    answer = system.query(question)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _stream_response(system: RobustRAGSystem, question: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    for token in system.stream_query(question):
        # Skip control characters used by the CLI (carriage returns, etc.)
        if token.startswith("\r"):
            continue
        yield _make_chunk(chunk_id, token)

    yield _make_chunk(chunk_id, "", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ---- File upload (PDF) ----------------------------------------------------

@app.post("/v1/files/upload")
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file. It will be saved to the /documents folder and
    automatically processed by the RAG system (watchdog picks it up).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    system = get_rag()
    dest = Path(system.config.pdf_folder) / file.filename

    # Save the uploaded file
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logging.info(f"Uploaded file saved to {dest}")

    # Process immediately instead of waiting for watchdog
    system.process_single_file(str(dest))

    return {
        "id": f"file-{uuid.uuid4().hex[:8]}",
        "object": "file",
        "filename": file.filename,
        "status": "processed",
        "message": f"'{file.filename}' uploaded and processed successfully.",
    }


# ---- Health check ---------------------------------------------------------

@app.get("/health")
async def health():
    system = get_rag()
    info = system.get_system_info()
    return {"status": "healthy", **info}


# ---------------------------------------------------------------------------
# Open WebUI Pipelines-compatible endpoints (optional)
# ---------------------------------------------------------------------------

@app.get("/v1/pipelines")
async def list_pipelines():
    """Open WebUI may call this to discover pipelines."""
    return {
        "data": [
            {
                "id": MODEL_NAME,
                "name": "RAG Chat Assistant",
                "description": "Ask questions about your uploaded PDF documents.",
            }
        ]
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  RAG Chat Assistant API Server")
    print(f"  Listening on http://{API_HOST}:{API_PORT}")
    print(f"  Add this URL to Open WebUI → Settings → Connections")
    print(f"    → OpenAI API: http://localhost:{API_PORT}/v1")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
