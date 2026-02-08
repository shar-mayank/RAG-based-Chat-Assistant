# RAG-based Chat Assistant

A local-first Retrieval-Augmented Generation (RAG) chat assistant that processes
PDF documents and answers questions based on their contents. The entire system
runs on your machine through Ollama, so no data ever leaves your network. It
handles text-based, image-based, and scanned PDFs by falling back to OCR
automatically when standard text extraction fails.

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Installation](#step-by-step-installation)
5. [Running the CLI](#running-the-cli)
6. [Running the API Server (for Open WebUI)](#running-the-api-server-for-open-webui)
7. [Connecting Open WebUI to the API Server](#connecting-open-webui-to-the-api-server)
8. [Installing the Pipe Function in Open WebUI](#installing-the-pipe-function-in-open-webui)
9. [Uploading PDFs](#uploading-pdfs)
10. [CLI Commands Reference](#cli-commands-reference)
11. [Configuration Reference](#configuration-reference)
12. [Project Structure](#project-structure)
13. [Troubleshooting](#troubleshooting)
14. [How It Works Internally](#how-it-works-internally)
15. [License](#license)

---

## Features

- Multi-layered PDF text extraction (PyMuPDF, then PyPDF, then OCR via
  Tesseract) so that every type of PDF is supported.
- Runs entirely locally with Ollama. No API keys, no cloud services.
- Live folder monitoring with watchdog. Drop a PDF into the documents folder and
  it is indexed automatically without restarting anything.
- Token-by-token streaming responses in both the CLI and the API server.
- Post-processing filter that strips boilerplate filler phrases such as "Based
  on the context..." from model output.
- OpenAI-compatible REST API so that any client that speaks the OpenAI protocol
  (including Open WebUI) can connect directly.
- Open WebUI Pipe function included. PDFs attached in the chat window are
  automatically saved to the documents folder and processed.
- Prompt injection detection and response security checks built in.
- Conversation caching with Jaccard similarity to avoid redundant LLM calls for
  near-identical questions.

---

## Architecture Overview

```
+------------------+       +------------------+       +-----------------+
|                  |       |                  |       |                 |
|  PDF documents   +------>+  rag_system.py   +------>+  FAISS vector   |
|  ./documents/    |       |  (processing +   |       |  store          |
|                  |       |   querying)       |       |  ./vector_db/   |
+------------------+       +--------+---------+       +-----------------+
                                    |
                       +------------+------------+
                       |                         |
              +--------v---------+     +---------v--------+
              |                  |     |                  |
              |  CLI interface   |     |  api_server.py   |
              |  (rag_system.py  |     |  (FastAPI,       |
              |   main)          |     |   port 9099)     |
              |                  |     |                  |
              +------------------+     +--------+---------+
                                                |
                                       +--------v---------+
                                       |                  |
                                       |  Open WebUI      |
                                       |  (port 8080)     |
                                       |  + Pipe function  |
                                       |  openwebui_pipe  |
                                       +------------------+
```

---

## Prerequisites

Install every item below before proceeding. Skipping any of them will cause
errors during setup or at runtime.

### 1. Python 3.10 or higher

Verify with:

```bash
python3 --version
```

If the version is below 3.10, install a newer Python from https://www.python.org
or through your system package manager.

### 2. Ollama

Ollama runs the LLM and embedding model locally. Download and install it from
https://ollama.com. After installation, confirm it is working:

```bash
ollama --version
```

Make sure the Ollama application is running (it must stay running in the
background the entire time you use this project). On macOS it appears as a
menu-bar icon. On Linux it runs as a systemd service.

### 3. Tesseract OCR Engine

Required for extracting text from scanned or image-based PDFs. Without it, the
system still works for normal text PDFs, but OCR extraction will be disabled.

macOS (Homebrew):

```bash
brew install tesseract
```

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

Windows:

Download the installer from
https://github.com/UB-Mannheim/tesseract/wiki and add the install
directory to your system PATH.

Verify with:

```bash
tesseract --version
```

### 4. Poppler (required by pdf2image for OCR)

pdf2image converts PDF pages to images before sending them to Tesseract. It
depends on the Poppler library.

macOS:

```bash
brew install poppler
```

Ubuntu / Debian:

```bash
sudo apt-get install poppler-utils
```

Windows:

Download from https://github.com/oschwartz10612/poppler-windows/releases,
extract it, and add the bin directory to your system PATH.

### 5. Poetry (Python dependency manager)

This project uses Poetry to manage its virtual environment and dependencies.

Install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or with pip:

```bash
pip install poetry
```

Verify with:

```bash
poetry --version
```

### 6. uv (optional, speeds up dependency installation)

uv is a fast Python package installer. Poetry will use it automatically if it
is installed. It is optional but recommended.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with pip:

```bash
pip install uv
```

### 7. Open WebUI (optional, for the web interface)

Only required if you want the browser-based chat interface. There are multiple
installation methods:

**Option 1: Docker (Recommended)**

```bash
docker run -d -p 3000:8080 --name open-webui ghcr.io/open-webui/open-webui:main
```

**Option 2: Python pip**

```bash
pip install open-webui
open-webui serve
```

This will start Open WebUI on http://localhost:8080 by default.

**Option 3: Other installation methods**

For additional installation options including Kubernetes, Podman, Kustomize, and
Helm, see the official installation guide at
https://github.com/open-webui/open-webui?tab=readme-ov-file#how-to-install-

---

## Step-by-Step Installation

Run every command below in order. Do not skip any step.

### 1. Clone the repository

```bash
git clone https://github.com/shar-mayank/RAG-based-Chat-Assistant.git
cd RAG-based-Chat-Assistant
```

### 2. Pull the required Ollama models

These two models must be downloaded before the system can start. The first is
the language model used for answering questions. The second is the embedding
model used for converting text into vectors.

```bash
ollama pull gemma3:4b
ollama pull bge-m3:latest
```

Each download is several gigabytes. Wait for both to finish completely.

You can confirm the models are available with:

```bash
ollama list
```

Both `gemma3:4b` and `bge-m3:latest` should appear in the output.

### 3. Install Python dependencies with Poetry

From the project root directory:

```bash
poetry install
```

This creates a virtual environment and installs all packages listed in
pyproject.toml: langchain, faiss-cpu, pymupdf, pypdf, pytesseract, pdf2image,
watchdog, fastapi, uvicorn, python-multipart, and python-dotenv.

If you see resolver errors, make sure your Python version is 3.10 or higher.
Poetry will refuse to install if the Python constraint is not met.

### 4. Create the documents folder (if it does not already exist)

```bash
mkdir -p documents
```

This is the folder where you place PDF files for the system to process.

---

## Running the CLI

The CLI is the simplest way to use the system. It runs entirely in your
terminal.

### 1. Place PDF files in the documents folder

Copy or move any PDF files you want to query into the `./documents` directory.

### 2. Start the CLI

```bash
poetry run python rag_system.py
```

Alternatively, you can activate the Poetry virtual environment first and then
run the script directly:

```bash
source $(poetry env activate)
python rag_system.py
```

### 3. What happens on startup

The system will:
- Initialize the Ollama LLM and embedding model connections.
- Check whether Tesseract is installed (prints "OCR dependencies available" or
  a warning if not found).
- Load any previously saved vector store from ./vector_db.
- Scan the ./documents folder and process any new or modified PDFs.
- Start a watchdog observer to monitor the folder for future changes.
- Print system information and an extraction summary.
- Present the interactive prompt.

### 4. Ask questions

Type your question at the prompt and press Enter. The answer will stream
token-by-token.

```
Ask a question (or 'quit' to exit):
```

Type `quit` or `exit` to shut down the system cleanly.

---

## Running the API Server (for Open WebUI)

The API server exposes the same RAG system as an OpenAI-compatible HTTP
service. This is required if you want to use Open WebUI or any other
OpenAI-compatible client.

### 1. Start the server

```bash
poetry run python api_server.py
```

You will see output like this:

```
============================================================
  RAG Chat Assistant API Server
  Listening on http://0.0.0.0:9099
  Add this URL to Open WebUI -> Settings -> Connections
    -> OpenAI API: http://localhost:9099/v1
============================================================
```

The server listens on port 9099 by default.

### 2. Verifying the server is running

In a separate terminal, run:

```bash
curl http://localhost:9099/health
```

You should get a JSON response with system status information.

You can also check the model list:

```bash
curl http://localhost:9099/v1/models
```

This should return a list containing "rag-assistant".

---

## Connecting Open WebUI to the API Server

These instructions assume Open WebUI is running at http://localhost:8080 and the
API server is running at http://localhost:9099.

1. Open your browser and go to http://localhost:8080.

2. Click on your profile icon in the bottom-left corner, then click "Admin Panel".

3. In the Admin Panel, click "Settings" on top.

4. Click "Connections" in the settings menu.

5. Under the "OpenAI API" section, click the "+" button to add a new
   connection.

5. In the "URL" field, enter:

   ```
   http://localhost:9099/v1
   ```

   Important: include the /v1 at the end. Open WebUI appends paths like
   /models and /chat/completions to this base URL. Without /v1 the
   requests will go to the wrong endpoints.

   Important: if Open WebUI is running inside Docker and the API server is
   running on the host machine, use `http://host.docker.internal:9099/v1`
   instead of localhost. Docker containers cannot reach the host's localhost
   directly.

6. In the "API Key" field, enter any non-empty value (for example: `none`).
   The RAG API server does not check the key, but Open WebUI requires
   the field to be filled in.

7. Click the "Save" button.

8. Go back to the main chat view. Click the model dropdown at the top. You
   should see "rag-assistant" in the list. Select it.

9. Type a question and send it. The response will stream in from the RAG
   system.

If "rag-assistant" does not appear in the dropdown, check the following:
- The API server terminal shows no errors.
- The URL in the connection settings ends with /v1.
- The API key field is not empty.
- If using Docker for Open WebUI, you used host.docker.internal instead of
  localhost.

---

## Installing the Pipe Function in Open WebUI

The Pipe function is an optional but recommended addition. It creates a
dedicated "RAG Chat Assistant" model inside Open WebUI that can receive PDF
file uploads directly in the chat. When you attach a PDF, the Pipe function
sends it to the API server's /upload endpoint, which saves it to the documents
folder and processes it immediately.

### Step-by-step instructions

1. Make sure the API server is already running (see previous section).

2. Open Open WebUI in your browser (http://localhost:8080).

3. Click on your profile icon in the bottom-left corner, then click "Admin Panel".

4. At the top of the Admin Panel, click "Functions".

5. Click the "+" button in the top-right corner to create a new function.

6. You will see a form with a name field and a code editor.

7. In the "Name" field, type: RAG Chat Assistant
   (or any name you prefer).

8. Now open the file `openwebui_pipe.py` from this repository in any text
   editor on your computer. Select all of its contents and copy them to your
   clipboard.

9. Go back to the Open WebUI browser tab. Click inside the code editor area.
   Select all the placeholder code that is already there and delete it. Paste
   the contents you copied from openwebui_pipe.py.

   The code starts with a docstring block:

   ```
   """
   title: RAG Chat Assistant
   author: shar-mayank
   version: 3.0.0
   license: MIT
   description: >
       A Pipe function that connects Open WebUI to the local RAG Chat Assistant.
   ...
   """
   ```

   Make sure the entire file is pasted, from the opening triple-quote to the
   very last line of the Pipe class.

10. Click "Save" at the bottom of the form.

11. After saving, you will see the function listed on the Functions page. There
    is a toggle switch next to it. Make sure it is turned ON (enabled).

12. Go back to the main chat view. Click the model dropdown. You should now
    see a new entry called "RAG Chat Assistant" (separate from the
    "rag-assistant" model you may have connected earlier). Select it.

13. You can now type questions and also attach PDFs using the paperclip/attach
    button in the chat input area. Attached PDFs are automatically uploaded to
    the RAG server and processed.

### Configuring the Pipe function (Valves)

After creating the function, you can adjust its settings:

1. On the Functions page, find the RAG Chat Assistant function card.

2. Click the gear icon on the function card to open the Valves panel.

3. The only configurable valve is RAG_API_BASE_URL, which defaults to
   http://localhost:9099. Change this if your API server runs on a different
   host or port.

4. Click Save.

---

## Uploading PDFs

There are three ways to get PDFs into the system:

### Method 1: Drop files into the documents folder

Copy or move PDF files directly into the `./documents` directory. The watchdog
file monitor detects new and modified files automatically and processes them
within a few seconds. This works while either the CLI or the API server is
running.

### Method 2: Upload via the API

Send a multipart POST request to the /upload endpoint:

```bash
curl -X POST http://localhost:9099/upload -F "file=@/path/to/your-document.pdf"
```

The file is saved to the documents folder and processed immediately. The
response includes the filename and a status field.

### Method 3: Attach in Open WebUI chat (requires the Pipe function)

If you installed the Pipe function as described above, click the paperclip
icon in the Open WebUI chat input, select a PDF, and send your message. The
Pipe function handles the upload automatically.

---

## CLI Commands Reference

These commands are available at the interactive prompt when running
`poetry run python rag_system.py`:

| Command  | Description                                                        |
|----------|--------------------------------------------------------------------|
| help     | Print the list of available commands.                              |
| info     | Show current system configuration: models, paths, monitoring       |
|          | status, OCR availability, and extraction method statistics.        |
| summary  | Show a detailed summary of all processed files, extraction methods |
|          | used per file, and chunk counts.                                   |
| debug    | Toggle debug mode. When enabled, every PDF processing step prints  |
|          | detailed information: text previews, character and word counts,    |
|          | page-level metadata, OCR confidence scores, and chunk size         |
|          | distributions.                                                     |
| quit     | Shut down the file monitor and exit the program cleanly.           |
| exit     | Same as quit.                                                      |

---

## Configuration Reference

The system is configured through the RAGConfig dataclass in rag_system.py. The
defaults are:

| Parameter          | Default Value    | Description                                |
|--------------------|------------------|--------------------------------------------|
| pdf_folder         | ./documents      | Directory to watch for PDF files.          |
| vector_db_path     | ./vector_db      | Directory where the FAISS index is stored. |
| model_name         | gemma3:4b        | Ollama LLM model for answering questions.  |
| embedding_model    | bge-m3:latest    | Ollama model for generating embeddings.    |
| chunk_size         | 1500             | Maximum characters per text chunk.         |
| chunk_overlap      | 300              | Overlap between consecutive chunks.        |
| max_retrieval_docs | 6                | Number of chunks retrieved per query.      |
| temperature        | 0.0              | LLM temperature (0 = deterministic).       |
| top_p              | 0.1              | Nucleus sampling parameter.                |
| debug_mode         | False            | Enable verbose processing output.          |
| ocr_dpi            | 300              | DPI for converting PDF pages to images.    |
| ocr_lang           | eng              | Tesseract language code for OCR.           |

The API server also reads these environment variables:

| Variable  | Default   | Description                           |
|-----------|-----------|---------------------------------------|
| API_HOST  | 0.0.0.0   | Host address the server binds to.     |
| API_PORT  | 9099      | Port number the server listens on.    |

---

## Project Structure

```
RAG-based-Chat-Assistant/
|-- api_server.py          FastAPI server exposing an OpenAI-compatible API.
|-- rag_system.py          Core RAG logic: PDF processing, vector store,
|                          querying, streaming, and the interactive CLI.
|-- openwebui_pipe.py      Pipe function code to paste into Open WebUI.
|-- pyproject.toml         Poetry project definition and dependencies.
|-- documents/             Place your PDF files here. Monitored at runtime.
|-- vector_db/
|   |-- faiss_index/
|   |   |-- index.faiss    The serialized FAISS vector index.
|   |   |-- index.pkl      Metadata associated with the index.
|   |-- processed_files.pkl  Record of which files have been processed
|                             and their hashes.
|-- rag_system.log         Log file created at runtime.
|-- LICENSE                MIT license.
|-- README.md              This file.
```

---

## Troubleshooting

### "Address already in use" when starting the API server

This means a previous instance of the server is still running in the
background. Closing a terminal tab does not always kill the process.

Find the process using the port:

```bash
lsof -ti :9099
```

This prints one or more process IDs. Kill them:

```bash
kill -9 <PID>
```

Replace `<PID>` with the number from the previous command. For example:

```bash
kill -9 84974
```

Then start the server again. To avoid this in the future, always press Ctrl+C
in the terminal where the server is running before closing the terminal.

If you want to do it in one command:

```bash
lsof -ti :9099 | xargs kill -9 2>/dev/null; poetry run python api_server.py
```

### Ollama connection errors

If you see errors about failing to connect to Ollama or model not found:

1. Make sure the Ollama application is running. On macOS, check for the Ollama
   icon in the menu bar. On Linux, check with `systemctl status ollama`.

2. Confirm both models are pulled:

   ```bash
   ollama list
   ```

   You must see both `gemma3:4b` and `bge-m3:latest`. If either is missing,
   pull it:

   ```bash
   ollama pull gemma3:4b
   ollama pull bge-m3:latest
   ```

3. By default, Ollama listens on http://localhost:11434. If you changed the
   Ollama host or port, set the OLLAMA_HOST environment variable before
   running the system.

### OCR not available

If the system prints "OCR not available" at startup:

1. Verify Tesseract is installed and in your PATH:

   ```bash
   tesseract --version
   ```

2. Verify Poppler is installed (needed by pdf2image):

   macOS: `brew install poppler`
   Ubuntu: `sudo apt-get install poppler-utils`

3. If Tesseract is installed but not found, you may need to set the path
   explicitly. Create a .env file in the project root:

   ```
   TESSERACT_CMD=/usr/local/bin/tesseract
   ```

   Or on macOS with Homebrew, it is typically at:

   ```
   TESSERACT_CMD=/opt/homebrew/bin/tesseract
   ```

The system still works without OCR. It will process text-based PDFs normally
but skip image-based or scanned PDFs.

### "No documents have been processed yet"

This response means the vector store is empty. Check:

1. There are PDF files in the ./documents folder.
2. The files are actual PDFs (not renamed text files or corrupt downloads).
3. Look at the terminal output for processing errors during startup.
4. Try deleting the ./vector_db directory and restarting to force
   reprocessing:

   ```bash
   rm -rf vector_db
   poetry run python rag_system.py
   ```

### Poetry "command not found"

If `poetry` is not recognized after installation, your shell may not have
reloaded its PATH. Try:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add this line to your shell configuration file (~/.zshrc, ~/.bashrc, or
~/.bash_profile) to make it permanent.

### Open WebUI cannot reach the API server (Docker)

If Open WebUI is running inside a Docker container and the API server is
running directly on the host machine, localhost inside the container refers to
the container itself, not the host. Use the special Docker DNS name:

```
http://host.docker.internal:9099/v1
```

Enter this as the URL in Open WebUI's connection settings instead of
http://localhost:9099/v1.

### "rag-assistant" model not appearing in Open WebUI

1. Confirm the API server is running and accessible:

   ```bash
   curl http://localhost:9099/v1/models
   ```

2. In Open WebUI Settings -> Connections, verify the URL ends with /v1.

3. Verify the API Key field is not empty (enter any value like "none").

4. Click Save and refresh the page.

### Vector store corruption

If you encounter errors loading the vector store, delete it and let the system
rebuild it from the PDFs in the documents folder:

```bash
rm -rf vector_db
```

Then restart the CLI or API server. All PDFs in the documents folder will be
reprocessed.

---

## How It Works Internally

### Document Ingestion

The watchdog library monitors the ./documents directory for file creation and
modification events. When a PDF is detected, the PDFFileHandler class waits one
second (to ensure the file is fully written to disk) and then calls
process_single_file.

### Text Extraction (three-tier cascade)

The EnhancedDocumentProcessor tries three extraction methods in order:

1. PyMuPDF (fitz): Opens the PDF and calls get_text() on each page. This is
   the fastest method and works for most standard PDFs.

2. PyPDF (PyPDFLoader from LangChain): If PyMuPDF fails or returns no text,
   the system tries LangChain's PyPDFLoader as a fallback.

3. OCR (pytesseract + pdf2image): If both text extractors return nothing, the
   system converts each page to a PNG image at 300 DPI using pdf2image (which
   calls Poppler internally), then runs pytesseract on each image. The
   extracted text is cleaned to remove common OCR artifacts such as isolated
   single characters and lines with only special characters.

Each document records which extraction method produced its text in the
metadata.

### Chunking

The extracted text is split using LangChain's RecursiveCharacterTextSplitter
with a chunk size of 1500 characters and an overlap of 300 characters. Pages
with fewer than 50 characters are filtered out.

### Embedding and Vector Storage

Each text chunk is converted to a vector embedding using the Ollama bge-m3
model. The embeddings are stored in a FAISS index on disk under
./vector_db/faiss_index. A separate pickle file (processed_files.pkl) tracks
which files have been processed and their MD5 hashes, so files are not
reprocessed unless they change.

### Querying

When a question is submitted:

1. The question is validated (checked for empty input and prompt injection
   patterns).
2. The question is converted to an embedding and used to retrieve the top 6
   most similar chunks from the FAISS index.
3. Retrieved chunks are deduplicated (identical chunks from duplicate uploads
   are removed).
4. The chunks and the question are formatted into a prompt and sent to the
   gemma3:4b model through Ollama.
5. The response is streamed token-by-token.
6. A post-processing filter strips boilerplate phrases and checks for
   responses that indicate the model is using external knowledge rather than
   the provided context.
7. Valid responses are cached so that near-identical follow-up questions can
   be answered instantly without another LLM call.

### API Server

The api_server.py file wraps the RAG system in a FastAPI application that
implements the OpenAI chat completions API specification. It supports both
streaming and non-streaming responses. It also exposes a /upload endpoint for
PDF file uploads and a /health endpoint for status checks.

### Pipe Function

The openwebui_pipe.py file defines an Open WebUI Pipe class that runs inside
the Open WebUI process. When a user attaches a file in the chat, the Pipe reads
the file from Open WebUI's internal storage, uploads it to the RAG API server's
/upload endpoint, and then forwards the user's question to the
/v1/chat/completions endpoint. The response is returned to the user in the chat
window.

---

## License

This project is released under the MIT License. See the LICENSE file for the
full text.
