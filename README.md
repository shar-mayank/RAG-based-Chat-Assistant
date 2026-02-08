This repository contains a robust, local-first RAG (Retrieval-Augmented Generation) chat assistant. It is designed to process and understand your PDF documents, allowing you to ask questions and receive accurate, context-aware answers directly from your document library.

The system operates entirely on your local machine using Ollama, ensuring your data remains private. It features an advanced document processing pipeline that can handle a wide variety of PDF types, including text-based, image-based, and scanned documents, by automatically falling back to Optical Character Recognition (OCR) when needed.

## Key Features

* **Robust Multi-layered PDF Processing**: Employs a sophisticated, three-tiered approach to extract text from any PDF:
  1. **PyMuPDF**: For fast and accurate text extraction from standard PDFs.
  2. **PyPDF**: A reliable fallback for PDFs where PyMuPDF might fail.
  3. **OCR (Tesseract)**: Automatically processes image-based or scanned PDFs by converting pages to images and extracting text, ensuring no document is left behind.
* **Local & Private**: Runs completely locally using **Ollama**, meaning your documents and queries are never sent to the cloud.
* **Live Document Monitoring**: Actively watches a designated folder for new or modified PDFs and automatically processes them into the knowledge base without requiring a restart.
* **Streaming Responses**: The assistant streams answers token-by-token, providing a responsive and interactive user experience.
* **Intelligent Response Filtering**: Post-processes the LLM's output to remove conversational filler and boilerplate phrases (e.g., "Based on the context..."), providing clean, direct answers.
* **Interactive CLI**: Comes with a user-friendly command-line interface that includes helpful commands to inspect the system's state, view processing summaries, and toggle debug modes.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1. **Python 3.10+**
2. **Ollama**: The system relies on Ollama to run local LLMs. Download and install it from the [official Ollama website](https://ollama.com/).
3. **Tesseract OCR Engine**: Required for the OCR functionality.
   * Follow the installation instructions for your OS from the [Tesseract GitHub repository](https://github.com/tesseract-ocr/tesseract). Ensure the `tesseract` command is available in your system's PATH.

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/shar-mayank/RAG-based-Chat-Assistant.git
   cd RAG-based-Chat-Assistant
   ```
2. **Pull Required Ollama Models**
   Open your terminal and run the following commands to download the default language and embedding models:

   ```bash
   ollama pull gemma3:4b
   ollama pull bge-m3:latest
   ```

   Ensure the Ollama application is running in the background.
3. **Install Poetry and uv**
   This project uses Poetry for dependency management and uv for faster package installation.

   ```bash
   # Install uv (fast Python package installer)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Or using pip:
   ```bash
   pip install uv poetry
   ```

4. **Install Project Dependencies**
   Use Poetry with uv for faster dependency resolution and installation:

   ```bash
   # Install dependencies using Poetry (with uv acceleration)
   poetry install
   ```

   Poetry will automatically create a virtual environment and install all dependencies.

5. **Create the Documents Folder**
   The application monitors a `./documents` folder for your PDFs.

   ```bash
   mkdir documents
   ```

## Usage

There are two ways to use this system: the **CLI** (terminal) or through **Open WebUI** (web interface).

### Option 1: CLI (Terminal)

1. **Add PDFs**: Place any PDF files you want to chat with into the `./documents` folder. The system will detect and process them automatically.
2. **Run the Assistant**: Start the main application script using Poetry.

   ```bash
   poetry run python rag_system.py
   ```

   Alternatively, activate the Poetry environment first (Poetry 2.0+):
   ```bash
   source $(poetry env activate)
   python rag_system.py
   ```
3. **Interact with the CLI**:
   Upon startup, the system will process any existing PDFs in the `documents` folder and then begin monitoring for changes. You will see a prompt to ask questions.

   ```
   Ask a question (or 'quit' to exit):
   ```

   You can ask any question related to the content of your PDFs.

### Option 2: Open WebUI Integration

This project includes an OpenAI-compatible API server that integrates with [Open WebUI](https://openwebui.com/), giving you a full web-based chat interface for your RAG system.

1. **Start the API Server**

   ```bash
   poetry run python api_server.py
   ```

   The server starts on `http://localhost:9099` by default. You can change the port with:
   ```bash
   API_PORT=8080 poetry run python api_server.py
   ```

2. **Connect Open WebUI**

   Open your Open WebUI instance (usually `http://localhost:3000`) and:
   1. Go to **Settings** → **Connections**
   2. Under **OpenAI API**, click the **+** button to add a new connection
   3. Set the **URL** to: `http://localhost:9099/v1`
   4. Set the **API Key** to any value (e.g., `none`) — it's not required but the field cannot be empty
   5. Click **Save**

3. **Start Chatting**

   - Select the **rag-assistant** model from the model dropdown in Open WebUI
   - Ask questions about your documents
   - The system uses your locally processed PDFs from the `./documents` folder

4. **Upload PDFs Directly from Open WebUI (Recommended)**

   Install the **RAG Chat Assistant Pipe Function** so that PDFs you attach in the
   Open WebUI chat are automatically sent to your local `/documents` folder.

   1. In Open WebUI, go to **Workspace** → **Functions** → click **+** (Add Function)
   2. Give the function a name (e.g. `RAG Chat Assistant`)
   3. Open the file `openwebui_pipe.py` from this repo and **copy-paste its entire
      contents** into the code editor in Open WebUI
   4. Click **Save**
   5. Enable the function using the toggle
   6. A new model called **RAG Chat Assistant** will appear in the model selector —
      select it and start chatting
   7. When you attach a PDF using the 📎 button, the Pipe downloads it, saves it to
      your `/documents` folder, processes it through the RAG pipeline, and then
      answers your questions

   > **Tip:** After creating the function, click the ⚙️ (gear icon) on the function
   > card to open **Valves** (settings). You can change the `RAG_API_BASE_URL` if
   > your API server runs on a different port.

5. **Other Ways to Upload PDFs**

   - **Drop files** directly into the `./documents` folder — the watchdog monitor picks them up automatically
   - **Upload via API** — send a POST request to `http://localhost:9099/upload`:
     ```bash
     curl -X POST http://localhost:9099/upload -F "file=@your-document.pdf"
     ```

### CLI Commands

The CLI also accepts special commands for managing the system:

* `help`: Displays a list of all available commands.
* `info`: Shows the current system configuration and status, including loaded models and monitoring status.
* `summary`: Provides a detailed summary of all processed files, including which extraction methods were used and how many chunks were created per file.
* `debug`: Toggles debug mode on or off. When enabled, it provides verbose output during the PDF extraction process.
* `quit` or `exit`: Shuts down the application gracefully.

## How It Works

The system is built around a robust RAG pipeline orchestrated by LangChain.

1. **Document Ingestion & Monitoring**: The `watchdog` library monitors the `./documents` directory. When a PDF is added or modified, the `PDFFileHandler` triggers the processing workflow.
2. **Enhanced Document Processing**: The `EnhancedDocumentProcessor` attempts to extract text using a cascading strategy for maximum compatibility:
   * It first tries `PyMuPDF` for its speed and accuracy.
   * If that fails, it falls back to `PyPDFLoader`.
   * If no text is extracted, and the PDF appears to be image-based, it uses `pdf2image` and `pytesseract` to perform OCR on each page.
3. **Chunking and Vectorization**: The extracted text is split into smaller, overlapping chunks using `RecursiveCharacterTextSplitter`. These chunks are then converted into vector embeddings by the local Ollama `bge-m3:latest` model.
4. **Vector Storage**: The generated embeddings are stored locally in a `FAISS` vector store, located in the `./vector_db` directory. This allows for efficient similarity searches.
5. **Query and Retrieval**: When you ask a question:
   * Your query is converted into an embedding.
   * The `FAISS` retriever finds the most relevant document chunks based on vector similarity.
   * These chunks (context) and your original question are inserted into a prompt.
6. **Generation and Streaming**: The complete prompt is sent to the local `gemma3:4b` LLM via Ollama. The model generates a response, which is streamed back to you in real-time. The final response is cleaned by `ResponseFilter` to ensure it is direct and relevant.
