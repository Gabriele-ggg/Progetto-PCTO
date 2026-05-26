from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from pathlib import Path
import httpx
import json
import asyncio

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"  
BASE_DIR = Path(__file__).parent


# ── Robots.txt ────────────────────────────────────────────────────────────────

@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots():
    robots_path = BASE_DIR / "Robots.txt"
    if robots_path.exists():
        return robots_path.read_text(encoding="utf-8")
    return "User-agent: *\nDisallow: /"


# ── Context loader con sistema di CACHE per i PDF (Solo pypdf) ────────────────

def load_context_from_files() -> str:
    """
    Scansiona le cartelle per cercare .json e .pdf.
    Se un PDF è già stato analizzato in passato, legge il testo dalla cache (.txt) 
    evitando di ri-elaborare il PDF con pypdf, velocizzando l'avvio del 99%.
    """
    context_parts: list[str] = []
    print("=== AVVIO: Caricamento e indicizzazione del contesto ===")

    for path in sorted(BASE_DIR.rglob("*")):
        # Salta il file principale e i file nascosti/cache generati
        if path == BASE_DIR / "main.py" or path.name.startswith(".") or path.suffix.lower() == ".txt" and path.name != "Robots.txt":
            continue

        # ── Lettura JSON ──────────────────────────────────────────────────────
        if path.suffix.lower() == ".json" and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                pretty = json.dumps(data, ensure_ascii=False, indent=2)
                context_parts.append(f"[FILE JSON: {path.relative_to(BASE_DIR)}]\n{pretty}")
                print(f"✔️ JSON caricato: {path.name}")
            except Exception as e:
                context_parts.append(f"[FILE JSON: {path.relative_to(BASE_DIR)}] — Errore: {e}")
                print(f"❌ Errore JSON {path.name}: {e}")

        # ── Lettura PDF (Solo pypdf con CACHE ottimizzata) ────────────────────
        elif path.suffix.lower() == ".pdf" and path.is_file():
            # Definiamo il percorso del file di cache (es: documento.pdf -> documento.pdf.cache.txt)
            cache_path = path.with_suffix(path.suffix + ".cache.txt")
            
            # Se esiste già la cache, leggiamo il testo direttamente (Istantaneo)
            if cache_path.exists():
                try:
                    combined = cache_path.read_text(encoding="utf-8")
                    context_parts.append(f"[FILE PDF: {path.relative_to(BASE_DIR)}]\n{combined}")
                    print(f"⚡ PDF caricato dalla CACHE (Istantaneo): {path.name}")
                    continue
                except Exception as e:
                    print(f"⚠️ Impossibile leggere cache per {path.name}, rigenero...")

            # Se la cache non esiste, usiamo pypdf per estrarre il testo
            try:
                import pypdf
                text_pages: list[str] = []
                with open(path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for i, page in enumerate(reader.pages, 1):
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text_pages.append(f"  [Pagina {i}]\n{page_text.strip()}")
                
                combined = "\n".join(text_pages) if text_pages else "(nessun testo estraibile)"
                context_parts.append(f"[FILE PDF: {path.relative_to(BASE_DIR)}]\n{combined}")
                
                # Salviamo il testo estratto nella cache per i prossimi avvii
                cache_path.write_text(combined, encoding="utf-8")
                print(f"✔️ PDF analizzato con pypdf e salvato in cache: {path.name} ({len(reader.pages)} pagine)")
                
            except ImportError:
                context_parts.append(f"[FILE PDF: {path.relative_to(BASE_DIR)}] — Errore: pypdf non installato.")
                print("⚠️ Errore: Installa pypdf eseguendo: pip install pypdf")
            except Exception as e:
                context_parts.append(f"[FILE PDF: {path.relative_to(BASE_DIR)}] — Errore: {e}")
                print(f"❌ Errore PDF {path.name}: {e}")

    print("=== COMPLETATO: Tutti i dati sono pronti in RAM ===\n")
    return "\n\n".join(context_parts)


# Carichiamo tutto globalmente all'avvio sfruttando la cache velocizzata
GLOBAL_CONTEXT = load_context_from_files()


# ── API endpoint ──────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(request: QuestionRequest):
    context = GLOBAL_CONTEXT

    if context:
        prompt = (
            "Hai accesso ai seguenti file presenti nella directory del server.\n"
            "Usali come contesto per rispondere alla domanda dell'utente in italiano.\n\n"
            f"{context}\n\n"
            f"Domanda dell'utente: {request.question}"
        )
    else:
        prompt = request.question

    async def stream_response():
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                async with client.stream(
                    "POST",
                    OLLAMA_URL,
                    json={"model": MODEL, "prompt": prompt, "stream": True},
                ) as response:
                    
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f"⚠️ Errore dal server Ollama (Codice {response.status_code})."
                        print(f"Errore Ollama {response.status_code}: {error_text.decode()}")
                        return

                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            yield data.get("response", "")
                            if data.get("done"):
                                break
                                
            except httpx.ConnectError:
                yield "⚠️ Impossibile connettersi. Assicurati che Ollama sia avviato."
            except httpx.ReadTimeout:
                yield "⚠️ Timeout. Ollama sta impiegando troppo tempo a elaborare il prompt."
            except Exception as e:
                yield f"⚠️ Errore: {e}"

    return StreamingResponse(stream_response(), media_type="text/plain")


# ── Frontend HTML ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Chat con LLaMA 3</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #0f0f1a;
            color: #e0e0f0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 40px 16px;
        }
        h1 { font-size: 1.8rem; margin-bottom: 8px; color: #a78bfa; }
        p.subtitle { color: #888; margin-bottom: 32px; font-size: 0.95rem; }

        .chat-box {
            width: 100%;
            max-width: 720px;
            background: #1a1a2e;
            border-radius: 16px;
            padding: 24px;
            border: 1px solid #2e2e4e;
            min-height: 200px;
            max-height: 420px;
            overflow-y: auto;
            margin-bottom: 24px;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .chat-box p { margin-bottom: 12px; }
        .chat-box .user    { color: #a78bfa; font-weight: 600; }
        .chat-box .assistant { color: #e0e0f0; }
        .chat-box .hint    { color: #555; font-style: italic; }

        .input-row {
            display: flex;
            width: 100%;
            max-width: 720px;
            gap: 12px;
        }
        textarea {
            flex: 1;
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid #3a3a5e;
            background: #1a1a2e;
            color: #e0e0f0;
            font-size: 1rem;
            resize: none;
            height: 56px;
            outline: none;
            transition: border 0.2s;
        }
        textarea:focus { border-color: #a78bfa; }

        button {
            padding: 0 20px;
            border-radius: 12px;
            border: none;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            height: 56px;
            white-space: nowrap;
        }
        #sendBtn  { background: #7c3aed; }
        #sendBtn:hover  { background: #6d28d9; }
        #sendBtn:disabled { background: #3a2a5e; cursor: not-allowed; }

        #stopBtn  { background: #be123c; display: none; }
        #stopBtn:hover  { background: #9f1239; }

        .spinner-wrap {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #a78bfa;
            font-style: italic;
            font-size: 0.9rem;
            margin-bottom: 12px;
        }
        .spinner {
            width: 18px; height: 18px;
            border: 3px solid #3a2a5e;
            border-top-color: #a78bfa;
            border-radius: 50%;
            animation: spin 0.75s linear infinite;
            flex-shrink: 0;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .stopped-badge {
            display: inline-block;
            margin-left: 8px;
            font-size: 0.75rem;
            color: #f87171;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>🦙 Chat con LLaMA 3</h1>
    <p class="subtitle">Modello: llama3:8b — powered by Ollama</p>

    <div class="chat-box" id="chat">
        <p class="hint">Fai una domanda per iniziare...</p>
    </div>

    <div class="input-row">
        <textarea id="question" placeholder="Scrivi la tua domanda..." onkeydown="handleKey(event)"></textarea>
        <button id="sendBtn" onclick="sendQuestion()">Invia</button>
        <button id="stopBtn" onclick="stopGeneration()">⏹ Ferma</button>
    </div>

    <script>
        let currentReader = null;
        let abortController = null;

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        }

        function setSending(active) {
            document.getElementById('sendBtn').disabled = active;
            document.getElementById('stopBtn').style.display = active ? 'inline-flex' : 'none';
        }

        function stopGeneration() {
            if (abortController) abortController.abort();
            if (currentReader)   currentReader.cancel();
        }

        async function sendQuestion() {
            const input    = document.getElementById('question');
            const chat     = document.getElementById('chat');
            const question = input.value.trim();
            if (!question) return;

            if (chat.querySelector('.hint')) chat.innerHTML = '';

            chat.innerHTML += `<p><span class="user">Tu:</span> ${escapeHtml(question)}</p>`;
            input.value = '';
            setSending(true);

            const spinnerEl = document.createElement('div');
            spinnerEl.className = 'spinner-wrap';
            spinnerEl.innerHTML = '<div class="spinner"></div><span>LLaMA sta elaborando la richiesta...</span>';
            chat.appendChild(spinnerEl);
            chat.scrollTop = chat.scrollHeight;

            const answerEl  = document.createElement('p');
            answerEl.style.display = 'none';
            answerEl.innerHTML = '<span class="assistant"><strong>LLaMA:</strong> </span>';
            const textNode  = document.createTextNode('');
            answerEl.appendChild(textNode);
            chat.appendChild(answerEl);

            abortController = new AbortController();

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question }),
                    signal: abortController.signal
                });

                currentReader = res.body.getReader();
                const decoder = new TextDecoder();
                let firstChunk = true;

                while (true) {
                    const { done, value } = await currentReader.read();
                    if (done) break;
                    if (firstChunk) {
                        spinnerEl.remove();
                        answerEl.style.display = '';
                        firstChunk = false;
                    }
                    textNode.textContent += decoder.decode(value, { stream: true });
                    chat.scrollTop = chat.scrollHeight;
                }
            } catch (err) {
                spinnerEl.remove();
                answerEl.style.display = '';
                if (err.name === 'AbortError') {
                    if (!textNode.textContent) {
                        textNode.textContent = '(Generazione annullata)';
                    } else {
                        const badge = document.createElement('span');
                        badge.className = 'stopped-badge';
                        badge.textContent = '[Interrotto]';
                        answerEl.appendChild(badge);
                    }
                } else {
                    textNode.textContent = '⚠️ Errore di connessione.';
                }
            }

            currentReader    = null;
            abortController  = null;
            setSending(false);
        }

        function escapeHtml(t) {
            return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }
    </script>
</body>
</html>
"""