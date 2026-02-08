"""
title: RAG Chat Assistant
author: shar-mayank
version: 3.0.0
license: MIT
description: >
    A Pipe function that connects Open WebUI to the local RAG Chat Assistant.
    When you upload a PDF in the chat, it is automatically copied to the RAG
    server's /documents folder for processing. Questions are answered by the
    RAG system's vector store.
requirements: requests
"""

from typing import Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
import requests
import json
import os
import shutil


class Pipe:
    """
    Open WebUI Pipe function for the RAG Chat Assistant.

    Key design decisions:
    - The pipe runs INSIDE Open WebUI's Python process, so we can read
      uploaded files directly from disk (no HTTP auth needed).
    - We use the `__files__` parameter that Open WebUI injects, which
      contains file metadata including the `id` field.
    - We look up the actual file path from Open WebUI's database using
      the file id, then copy the PDF to our RAG server's /documents folder.
    - Questions are forwarded to the RAG API server via HTTP.
    """

    class Valves(BaseModel):
        RAG_API_BASE_URL: str = Field(
            default="http://localhost:9099",
            description="Base URL of the RAG Chat Assistant API server.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.file_handler = True

    def _rag_url(self, path: str) -> str:
        return f"{self.valves.RAG_API_BASE_URL}{path}"

    def _process_uploaded_file(self, file_id: str) -> str:
        """
        Look up a file by ID from Open WebUI's database, read it from disk,
        and upload it to the RAG server.
        """
        try:
            # Import Open WebUI's file model to look up the file path
            from open_webui.models.files import Files

            file_record = Files.get_file_by_id(file_id)
            if not file_record:
                return f"❌ File {file_id} not found in Open WebUI"

            filename = file_record.filename or f"file_{file_id}.pdf"

            # Only handle PDFs
            if not filename.lower().endswith(".pdf"):
                return f"⏭️ Skipped **{filename}** (not a PDF)"

            file_path = file_record.path
            if not file_path or not os.path.exists(file_path):
                return f"❌ File path not found for **{filename}**"

            # Read the file and upload to RAG server
            with open(file_path, "rb") as f:
                files = {"file": (filename, f.read(), "application/pdf")}

            upload_resp = requests.post(
                self._rag_url("/upload"), files=files, timeout=120
            )
            upload_resp.raise_for_status()
            result = upload_resp.json()

            return f"✅ **{filename}** uploaded and processed ({result.get('status', 'done')})"

        except requests.exceptions.ConnectionError:
            return f"❌ Cannot connect to RAG server at {self.valves.RAG_API_BASE_URL}"
        except ImportError:
            return "❌ Cannot access Open WebUI file storage (import error)"
        except Exception as e:
            return f"❌ Error processing file: {str(e)}"

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __files__: Optional[list] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__=None,
    ) -> Union[str, Generator, Iterator]:

        # ------------------------------------------------------------------
        # 1. Handle file uploads
        # ------------------------------------------------------------------
        upload_messages = []

        # Collect file references from all possible sources
        files = __files__ or []
        if not files and __metadata__:
            files = __metadata__.get("files", [])
        if not files:
            files = body.get("files", [])

        # Deduplicate by file ID to avoid processing the same file twice
        seen_ids = set()
        for file_info in files:
            if not isinstance(file_info, dict):
                continue
            file_id = file_info.get("id", "")
            if not file_id or file_id in seen_ids:
                continue
            seen_ids.add(file_id)

            # Emit a status event so the user sees progress
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Processing uploaded file...",
                            "done": False,
                        },
                    }
                )

            msg = self._process_uploaded_file(file_id)
            upload_messages.append(msg)

        if upload_messages and __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Files processed",
                        "done": True,
                    },
                }
            )

        # ------------------------------------------------------------------
        # 2. Extract the user's latest question
        # ------------------------------------------------------------------
        messages = body.get("messages", [])
        question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    question = content.strip()
                    break
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            question = part.get("text", "").strip()
                            if question:
                                break

        if not question:
            if upload_messages:
                return "PDFs processed. You can now ask questions about them."
            return "Please ask a question about your documents."

        # ------------------------------------------------------------------
        # 3. Query the RAG system
        # ------------------------------------------------------------------
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Searching documents...",
                        "done": False,
                    },
                }
            )

        try:
            payload = {
                "model": "rag-assistant",
                "messages": [{"role": "user", "content": question}],
                "stream": False,
            }

            resp = requests.post(
                self._rag_url("/v1/chat/completions"),
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            result = resp.json()

            answer = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No response from RAG system.")
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Done",
                            "done": True,
                        },
                    }
                )

            return answer

        except requests.exceptions.ConnectionError:
            return (
                "❌ **Cannot connect to the RAG API server.**\n\n"
                f"Make sure it is running at `{self.valves.RAG_API_BASE_URL}`\n\n"
                "Start it with:\n```\npoetry run python api_server.py\n```"
            )
        except requests.exceptions.Timeout:
            return "❌ The RAG server took too long to respond. Please try again."
        except Exception as e:
            return f"❌ Error: {str(e)}"
