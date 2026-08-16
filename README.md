# Document Ingestion and RAG Service

A backend service for ingesting PDF documents and querying them with AI. Upload any PDF, and the
service chunks and embeds it. Ask questions in natural language and get cited answers generated
from the document content using hybrid vector + lexical retrieval and an LLM.

## Architecture

```
PDF upload → chunk → embed → store (PostgreSQL + pgvector)
                                      ↑
Question → embed → vector search ─────┤
                 → lexical search ────┤ RRF fusion → top chunks → LLM → cited answer
```

## Quick start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Configure environment

Copy the example file and fill in your own keys — the `.env` file is gitignored and never committed:

```bash
cp .env.example .env        # Linux/Mac
copy .env.example .env      # Windows
```

Then edit `.env`:

```env
# Gemini
PROVIDER_MODE=gemini
GEMINI_API_KEY=your-key-here
DATABASE_URL=sqlite:///./copilot.db
```

Or for OpenAI:

```env
PROVIDER_MODE=openai
OPENAI_API_KEY=your-key-here
DATABASE_URL=sqlite:///./copilot.db
```

> SQLite works for local dev. For PostgreSQL use `postgresql+psycopg://user:pass@localhost:5432/copilot`.

### 3. Start the API

```bash
uvicorn kube_copilot.api:app --reload --port 8000
```

The API is ready when you see `Application startup complete`.

### 4. Check health

**curl:**
```bash
curl http://localhost:8000/health/live
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health/live"
```

### 5. Upload a document

**curl:**
```bash
curl -X POST http://localhost:8000/v1/ingestion/upload \
    -F "file=@/path/to/<your-document>.pdf" \
    -F "product_version=latest"
```

**PowerShell 5.1** (Windows PowerShell — no `-Form` support):
```powershell
$filePath = "C:\path\to\<your-document>.pdf"
$boundary = [System.Guid]::NewGuid().ToString()
$fileName = [System.IO.Path]::GetFileName($filePath)
$fileBytes = [System.IO.File]::ReadAllBytes($filePath)
$enc = [System.Text.Encoding]::UTF8

$ms = New-Object System.IO.MemoryStream
$p1 = $enc.GetBytes("--$boundary`r`nContent-Disposition: form-data; name=`"product_version`"`r`n`r`nlatest`r`n")
$ms.Write($p1, 0, $p1.Length)
$p2 = $enc.GetBytes("--$boundary`r`nContent-Disposition: form-data; name=`"file`"; filename=`"$fileName`"`r`nContent-Type: application/pdf`r`n`r`n")
$ms.Write($p2, 0, $p2.Length)
$ms.Write($fileBytes, 0, $fileBytes.Length)
$p3 = $enc.GetBytes("`r`n--$boundary--`r`n")
$ms.Write($p3, 0, $p3.Length)

Invoke-RestMethod -Uri "http://localhost:8000/v1/ingestion/upload" `
    -Method Post `
    -ContentType "multipart/form-data; boundary=$boundary" `
    -Body $ms.ToArray()
```

Expected response:
```json
{
  "filename": "your-document.pdf",
  "title": "Your Document",
  "corpus_id": "abc123...",
  "chunks": 47,
  "product_version": "latest"
}
```

### 6. Ask a question

**curl:**
```bash
curl -X POST http://localhost:8000/v1/query \
    -H "Content-Type: application/json" \
    -d '{"question": "<your question here>"}'
```

**PowerShell — get full answer text:**
```powershell
$result = Invoke-RestMethod -Uri "http://localhost:8000/v1/query" `
    -Method Post `
    -ContentType "application/json" `
    -Body '{"question": "<your question here>"}'

$result.answer
```

**PowerShell — see all fields (answer + sources):**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/v1/query" `
    -Method Post `
    -ContentType "application/json" `
    -Body '{"question": "<your question here>"}' | Format-List
```

Expected response shape:
```json
{
  "answer": "<answer>",
  "sources": [
    {
      "citation": 1,
      "title": "Your Document",
      "heading": "<answer heading>",
      "excerpt": "...",
      "score": 0.83,
      "url": "file://your-document.pdf"
    }
  ],
  "hit_count": 5
}
```

## Notes

- Upload the same filename again to re-index it — old chunks are replaced automatically.
- `product_version` is optional (defaults to `"latest"`). Use it to tag and filter documents by version.
- Interactive API docs are available at `http://localhost:8000/docs`.

## Development

Requires Python 3.12.

```bash
pytest          # run tests
ruff check .    # lint
```
