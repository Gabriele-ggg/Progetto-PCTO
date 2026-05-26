from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List
import httpx
import json
import asyncio

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:9b"  
BASE_DIR = Path(__file__).parent

# Cartelle selezionate dall'utente (None = tutte le sottocartelle)
SELECTED_FOLDERS: List[Path] = []
GLOBAL_CONTEXT: str = ""


# ── Robots.txt ────────────────────────────────────────────────────────────────

@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots():
    robots_path = BASE_DIR / "Robots.txt"
    if robots_path.exists():
        return robots_path.read_text(encoding="utf-8")
    return "User-agent: *\nDisallow: /"


# ── Context loader con Cache (Solo pypdf) ─────────────────────────────────────

def load_context_from_files(folders: List[Path] | None = None) -> str:
    """
    Scansiona le cartelle selezionate per cercare dati aziendali TPL FVG (.json e .pdf).
    Se folders è None o vuoto, scansiona tutta BASE_DIR.
    Usa la cache .txt per evitare di ri-analizzare i PDF a ogni avvio.
    """
    context_parts: list[str] = []
    print("\n==========================================================")
    print("🚌 [SISTEMA] Caricamento database orari e linee TPL FVG...")
    if folders:
        names = [f.name for f in folders]
        print(f"   Cartelle selezionate: {', '.join(names)}")
    else:
        print("   Scansione completa della directory base.")
    print("==========================================================")

    # Determina i root di scansione
    if folders:
        scan_roots = folders
    else:
        scan_roots = [BASE_DIR]

    # Raccogli tutti i file da scansionare
    all_files: list[Path] = []
    for root in scan_roots:
        if root.is_dir():
            all_files.extend(sorted(root.rglob("*")))
        elif root.is_file():
            all_files.append(root)

    for path in all_files:
        if path == BASE_DIR / "main.py" or path.name.startswith(".") or path.suffix.lower() == ".txt" and path.name != "Robots.txt":
            continue

        # JSON
        if path.suffix.lower() == ".json" and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                pretty = json.dumps(data, ensure_ascii=False, indent=2)
                context_parts.append(f"[DATI SOCIETARI JSON: {path.relative_to(BASE_DIR)}]\n{pretty}")
                print(f"  ✔️ Data-JSON indicizzato: {path.name}")
            except Exception as e:
                print(f"  ❌ Errore JSON {path.name}: {e}")

        # PDF
        elif path.suffix.lower() == ".pdf" and path.is_file():
            cache_path = path.with_suffix(path.suffix + ".cache.txt")
            
            if cache_path.exists():
                try:
                    combined = cache_path.read_text(encoding="utf-8")
                    context_parts.append(f"[ORARI PDF: {path.relative_to(BASE_DIR)}]\n{combined}")
                    print(f"  ⚡ Orari PDF da cache (Istantaneo): {path.name}")
                    continue
                except Exception as e:
                    print(f"  ⚠️ Errore lettura cache per {path.name}, rigenero...")

            try:
                import pypdf
                text_pages: list[str] = []
                with open(path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for i, page in enumerate(reader.pages, 1):
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text_pages.append(f"  [Pagina {i}]\n{page_text.strip()}")
                
                combined = "\n".join(text_pages) if text_pages else "(testo non estraibile)"
                context_parts.append(f"[ORARI PDF: {path.relative_to(BASE_DIR)}]\n{combined}")
                cache_path.write_text(combined, encoding="utf-8")
                print(f"  ✔️ PDF analizzato con pypdf: {path.name} ({len(reader.pages)} pagine)")
                
            except Exception as e:
                print(f"  ❌ Errore PDF {path.name}: {e}")

    print("==========================================================")
    print("✔️ [SISTEMA] Database TPL FVG pronto in memoria RAM.")
    print("==========================================================\n")
    return "\n\n".join(context_parts)


# Inizializzazione del contesto globale (a caldo, senza cartelle selezionate)
GLOBAL_CONTEXT = load_context_from_files()


# ── Endpoint per scaricare il modello dalla RAM immediatamente ────────────────

@app.post("/unload")
async def unload_model():
    """
    Invia un comando a Ollama per forzare lo scaricamento immediato del modello dalla RAM/VRAM.
    """
    print("\n⏹️ [SISTEMA] Ricevuto comando 'Interrompi' dal browser.")
    print("♻️ [SISTEMA] Invio richiesta di sblocco memoria a Ollama...")
    async with httpx.AsyncClient() as client:
        try:
            await client.post(OLLAMA_URL, json={"model": MODEL, "keep_alive": 0})
            print("✔️ [RAM LIBERA] Il modello è stato rimosso istantaneamente dalla memoria.\n")
            return {"status": "success", "message": "Modello rimosso dalla RAM"}
        except Exception as e:
            print(f"❌ [ERRORE] Impossibile liberare la RAM: {e}\n")
            return {"status": "error", "detail": str(e)}


# ── Endpoint per elencare e selezionare le cartelle ──────────────────────────

@app.get("/folders")
async def list_folders():
    """
    Restituisce le sottocartelle disponibili nella stessa directory di main.py.
    """
    folders = []
    for item in sorted(BASE_DIR.iterdir()):
        if item.is_dir() and not item.name.startswith(".") and item.name != "__pycache__":
            # Conta quanti PDF e JSON contiene (ricorsivamente)
            pdf_count  = len(list(item.rglob("*.pdf")))
            json_count = len(list(item.rglob("*.json")))
            folders.append({
                "name": item.name,
                "pdf_count":  pdf_count,
                "json_count": json_count,
            })
    return {"folders": folders, "selected": [f.name for f in SELECTED_FOLDERS]}


class FolderSelection(BaseModel):
    folders: List[str]   # lista di nomi di cartella relativi a BASE_DIR


@app.post("/set-folders")
async def set_folders(selection: FolderSelection):
    """
    Imposta le cartelle da cui caricare il contesto e ricarica il database.
    """
    global SELECTED_FOLDERS, GLOBAL_CONTEXT

    resolved: List[Path] = []
    errors: List[str] = []

    for name in selection.folders:
        candidate = (BASE_DIR / name).resolve()
        # Sicurezza: la cartella deve stare dentro BASE_DIR
        if not str(candidate).startswith(str(BASE_DIR.resolve())):
            errors.append(f"Percorso non consentito: {name}")
            continue
        if not candidate.is_dir():
            errors.append(f"Cartella non trovata: {name}")
            continue
        resolved.append(candidate)

    SELECTED_FOLDERS = resolved
    print(f"\n📁 [SISTEMA] Cartelle aggiornate: {[f.name for f in SELECTED_FOLDERS] or 'tutte'}")
    GLOBAL_CONTEXT = load_context_from_files(SELECTED_FOLDERS if SELECTED_FOLDERS else None)

    return {
        "status": "ok",
        "loaded_folders": [f.name for f in SELECTED_FOLDERS],
        "errors": errors,
        "context_length": len(GLOBAL_CONTEXT),
    }


# ── API endpoint per la generazione ───────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(request: QuestionRequest):
    # Log della domanda ricevuta nel terminale
    print("\n" + "─"*60)
    print(f"▶️ [RICHIESTA] Nuova ricerca itinerario dal passeggero:")
    print(f"    ↳ \"{request.question}\"")
    print("─"*60)

    context = GLOBAL_CONTEXT

    if context:
        prompt = (
            "Sei l'Assistente Virtuale di TPL FVG (Trasporto Pubblico Locale del Friuli Venezia Giulia).\n"
            "Il tuo compito è aiutare i viaggiatori a trovare il percorso, la linea, l'autobus o l'orario ideale.\n"
            "Rispondi in modo chiaro, cortese e preciso basandoti ESCLUSIVAMENTE sui seguenti dati ufficiali aziendali (orari, fermate, tariffe):\n\n"
            f"{context}\n\n"
            f"Richiesta del passeggero: {request.question}"
        )
    else:
        prompt = request.question

    async def stream_response():
        print("🧠 [AI THINKING] LLaMA sta elaborando il contesto dei file e calcolando la rotta...")
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            try:
                async with client.stream(
                    "POST",
                    OLLAMA_URL,
                    json={"model": MODEL, "prompt": prompt, "stream": True},
                ) as response:
                    
                    if response.status_code != 200:
                        print("❌ [ERRORE HTTP] Il server Ollama ha rifiutato la richiesta.")
                        yield "⚠️ Errore di comunicazione con il motore AI aziendale."
                        return

                    print("✍️ [AI STREAMING] Risposta in generazione sul browser (anteprima terminale):")
                    print("─"*40)
                    
                    first_chunk = True
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            token = data.get("response", "")
                            
                            # Stampiamo il token sul terminale in tempo reale senza andare a capo
                            print(token, end="", flush=True)
                            
                            yield token
                            if data.get("done"):
                                print("\n" + "─"*40)
                                print("✔️ [AI DONE] Risposta completata e inviata con successo al passeggero.\n")
                                break
                                
            except asyncio.CancelledError:
                print("\n⚠️ [AI CANCELLED] Lo streaming è stato interrotto bruscamente dall'utente.")
            except Exception as e:
                print(f"\n❌ [ERRORE FLUSSO] Si è verificato un problema: {e}\n")
                yield f"⚠️ Servizio momentaneamente non disponibile: {e}"

    return StreamingResponse(stream_response(), media_type="text/plain")


# ── Frontend HTML/CSS (Tema Istituzionale TPL FVG) ────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>TPL FVG - Assistente di Viaggio AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: '-apple-system', BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #f4f6f9;
            color: #1e293b;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }

        header {
            background: linear-gradient(135deg, #003b71 0%, #00569d 100%);
            color: white;
            width: 100%;
            padding: 20px 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .brand-container {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
        }

        header h1 {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        header p {
            font-size: 0.9rem;
            color: #cbd5e1;
        }

        main {
            flex: 1;
            width: 100%;
            max-width: 800px;
            margin: 24px auto;
            padding: 0 16px;
            display: flex;
            flex-direction: column;
        }

        .chat-container {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 400px;
            max-height: 550px;
            overflow-y: auto;
            padding: 24px;
            margin-bottom: 16px;
        }

        .message {
            margin-bottom: 16px;
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .message.user {
            background-color: #e2f0fd;
            color: #003b71;
            align-self: flex-end;
            border-bottom-right-radius: 2px;
            border-left: 4px solid #00569d;
        }

        .message.assistant {
            background-color: #f8fafc;
            color: #334155;
            align-self: flex-start;
            border-bottom-left-radius: 2px;
            border-left: 4px solid #00a896;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }

        .hint {
            color: #64748b;
            font-style: italic;
            text-align: center;
            margin: auto;
            font-size: 0.95rem;
        }

        .input-container {
            display: flex;
            gap: 12px;
            background: white;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }

        textarea {
            flex: 1;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 12px;
            font-size: 0.95rem;
            font-family: inherit;
            resize: none;
            height: 48px;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: #00569d;
            box-shadow: 0 0 0 3px rgba(0, 86, 157, 0.15);
        }

        button {
            border: none;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            padding: 0 24px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            height: 48px;
        }

        #sendBtn {
            background-color: #00569d;
            color: white;
        }

        #sendBtn:hover {
            background-color: #003b71;
        }

        #sendBtn:disabled {
            background-color: #cbd5e1;
            cursor: not-allowed;
        }

        #stopBtn {
            background-color: #ef4444;
            color: white;
            display: none;
        }

        #stopBtn:hover {
            background-color: #dc2626;
        }

        .loading-wrap {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #00569d;
            font-weight: 500;
            font-size: 0.9rem;
            margin-bottom: 16px;
            align-self: flex-start;
        }

        .pulse-icon {
            width: 10px;
            height: 10px;
            background-color: #00a896;
            border-radius: 50%;
            animation: pulse 1.2s infinite ease-in-out;
        }

        @keyframes pulse {
            0% { transform: scale(0.8); opacity: 0.5; }
            50% { transform: scale(1.3); opacity: 1; }
            100% { transform: scale(0.8); opacity: 0.5; }
        }

        footer {
            text-align: center;
            padding: 16px;
            font-size: 0.8rem;
            color: #94a3b8;
            border-top: 1px solid #e2e8f0;
            background: white;
        }
        /* ── Folder selector ─────────────────────────────── */
        .folder-panel {
            background: white;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            padding: 16px 20px;
            margin-bottom: 16px;
        }

        .folder-panel summary {
            cursor: pointer;
            font-weight: 600;
            color: #003b71;
            font-size: 0.95rem;
            user-select: none;
            list-style: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .folder-panel summary::after {
            content: '▸';
            transition: transform 0.2s;
            margin-left: auto;
            color: #94a3b8;
        }

        .folder-panel[open] summary::after {
            transform: rotate(90deg);
        }

        .folder-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }

        .folder-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            border: 2px solid #cbd5e1;
            background: #f8fafc;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.15s;
            color: #334155;
        }

        .folder-chip.selected {
            border-color: #00569d;
            background: #e2f0fd;
            color: #003b71;
            font-weight: 600;
        }

        .folder-chip .badge {
            font-size: 0.75rem;
            background: #e2e8f0;
            border-radius: 10px;
            padding: 1px 6px;
            color: #64748b;
        }

        .folder-chip.selected .badge {
            background: #bfdbfe;
            color: #1d4ed8;
        }

        .folder-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            align-items: center;
        }

        #applyFoldersBtn {
            background-color: #00569d;
            color: white;
            font-size: 0.85rem;
            height: 36px;
            padding: 0 16px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-weight: 600;
        }

        #applyFoldersBtn:hover { background-color: #003b71; }
        #applyFoldersBtn:disabled { background-color: #cbd5e1; cursor: not-allowed; }

        .folder-status {
            font-size: 0.8rem;
            color: #64748b;
        }
    </style>
</head>
<body>

    <header>
        <div class="brand-container">
            <h1>🚌 TPL FVG — Assistente Virtuale</h1>
            <p>Servizi automobilistici e marittimi integrati del Friuli Venezia Giulia</p>
        </div>
    </header>

    <main>
        <details class="folder-panel" id="folderPanel">
            <summary>📁 Cartelle dati — seleziona le sorgenti da ricercare</summary>
            <div class="folder-list" id="folderList">
                <span style="color:#94a3b8; font-size:0.85rem;">Caricamento cartelle...</span>
            </div>
            <div class="folder-actions">
                <button id="applyFoldersBtn" onclick="applyFolders()">✔ Applica selezione</button>
                <span class="folder-status" id="folderStatus"></span>
            </div>
        </details>

        <div class="chat-container" id="chat">
            <p class="hint">Indica il punto di partenza e la destinazione desiderata (es: "Come arrivo da Udine a Grado sabato pomeriggio?") per calcolare il percorso basato sui dati ufficiali.</p>
        </div>

        <div class="input-container">
            <textarea id="question" placeholder="Pianifica il tuo viaggio... Inserisci partenza e destinazione" onkeydown="handleKey(event)"></textarea>
            <button id="sendBtn" onclick="sendQuestion()">Cerca Percorso</button>
            <button id="stopBtn" onclick="stopGeneration()">⏹ Interrompi</button>
        </div>
    </main>

    <footer>
        Risposte generate dall'AI aziendale basate sui libretti orari e documenti ufficiali di TPL FVG.
    </footer>

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

        async function stopGeneration() {
            if (abortController) abortController.abort();
            if (currentReader)   currentReader.cancel();

            try {
                await fetch('/unload', { method: 'POST' });
                console.log("Richiesta di svuotamento RAM inviata.");
            } catch (err) {
                console.error("Errore nell'invio a /unload:", err);
            }
        }

        async function sendQuestion() {
            const input    = document.getElementById('question');
            const chat     = document.getElementById('chat');
            const question = input.value.trim();
            if (!question) return;

            if (chat.querySelector('.hint')) chat.innerHTML = '';

            chat.innerHTML += `<div class="message user"><strong>Mio Percorso:</strong><br>${escapeHtml(question)}</div>`;
            input.value = '';
            setSending(true);

            const loadingEl = document.createElement('div');
            loadingEl.className = 'loading-wrap';
            loadingEl.innerHTML = '<div class="pulse-icon"></div><span>Verifica coincidenze e orari ufficiali in corso...</span>';
            chat.appendChild(loadingEl);
            chat.scrollTop = chat.scrollHeight;

            const answerEl  = document.createElement('div');
            answerEl.className = 'message assistant';
            answerEl.style.display = 'none';
            answerEl.innerHTML = '<strong>Soluzione di viaggio consigliata:</strong><br>';
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
                        loadingEl.remove();
                        answerEl.style.display = 'block';
                        firstChunk = false;
                    }
                    textNode.textContent += decoder.decode(value, { stream: true });
                    chat.scrollTop = chat.scrollHeight;
                }
            } catch (err) {
                loadingEl.remove();
                answerEl.style.display = 'block';
                if (err.name === 'AbortError') {
                    if (!textNode.textContent) textNode.textContent = '(Ricerca itinerario annullata e RAM liberata)';
                } else {
                    textNode.textContent = '⚠️ Connessione ai sistemi informativi di bordo momentaneamente interrotta.';
                }
            }

            currentReader    = null;
            abortController  = null;
            setSending(false);
        }

        // ── Folder management ─────────────────────────────────────────────
        let availableFolders = [];
        let selectedFolders  = new Set();

        async function loadFolders() {
            try {
                const res  = await fetch('/folders');
                const data = await res.json();
                availableFolders = data.folders;
                selectedFolders  = new Set(data.selected);
                renderFolders();
            } catch (err) {
                document.getElementById('folderList').innerHTML =
                    '<span style="color:#ef4444;font-size:0.85rem;">⚠️ Impossibile caricare le cartelle.</span>';
            }
        }

        function renderFolders() {
            const container = document.getElementById('folderList');
            if (availableFolders.length === 0) {
                container.innerHTML = '<span style="color:#94a3b8;font-size:0.85rem;">Nessuna sottocartella trovata nella directory.</span>';
                return;
            }
            container.innerHTML = availableFolders.map(f => {
                const sel   = selectedFolders.has(f.name);
                const badge = [f.pdf_count  ? `${f.pdf_count} PDF`  : '',
                               f.json_count ? `${f.json_count} JSON` : '']
                              .filter(Boolean).join(' · ') || 'vuota';
                return `<div class="folder-chip ${sel ? 'selected' : ''}"
                             onclick="toggleFolder('${escapeHtml(f.name)}')">
                            📂 ${escapeHtml(f.name)}
                            <span class="badge">${badge}</span>
                        </div>`;
            }).join('');
        }

        function toggleFolder(name) {
            if (selectedFolders.has(name)) selectedFolders.delete(name);
            else selectedFolders.add(name);
            renderFolders();
        }

        async function applyFolders() {
            const btn    = document.getElementById('applyFoldersBtn');
            const status = document.getElementById('folderStatus');
            btn.disabled = true;
            status.textContent = 'Caricamento in corso...';

            try {
                const res  = await fetch('/set-folders', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify({ folders: [...selectedFolders] })
                });
                const data = await res.json();

                const loaded = data.loaded_folders.length
                    ? data.loaded_folders.join(', ')
                    : 'tutte le cartelle';
                const chars = data.context_length.toLocaleString('it-IT');
                status.textContent = `✔ Contesto aggiornato: ${loaded} — ${chars} caratteri indicizzati`;
                if (data.errors.length)
                    status.textContent += ' | ⚠️ ' + data.errors.join('; ');
            } catch (err) {
                status.textContent = '⚠️ Errore durante il caricamento.';
            } finally {
                btn.disabled = false;
            }
        }

        // Carica subito le cartelle all'avvio
        loadFolders();

        function escapeHtml(t) {
            return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }
    </script>
</body>
</html>
"""