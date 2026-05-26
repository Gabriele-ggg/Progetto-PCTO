from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import httpx

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"


class QuestionRequest(BaseModel):
    question: str


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
        h1 {
            font-size: 1.8rem;
            margin-bottom: 8px;
            color: #a78bfa;
        }
        p.subtitle {
            color: #888;
            margin-bottom: 32px;
            font-size: 0.95rem;
        }
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
        .chat-box .user { color: #a78bfa; font-weight: 600; }
        .chat-box .assistant { color: #e0e0f0; }
        .chat-box .thinking { color: #555; font-style: italic; }
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
            padding: 0 24px;
            border-radius: 12px;
            border: none;
            background: #7c3aed;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            height: 56px;
        }
        button:hover { background: #6d28d9; }
        button:disabled { background: #3a2a5e; cursor: not-allowed; }
    </style>
</head>
<body>
    <h1>🦙 Chat con LLaMA 3</h1>
    <p class="subtitle">Modello: llama3:8b — powered by Ollama</p>

    <div class="chat-box" id="chat">
        <p class="thinking">Fai una domanda per iniziare...</p>
    </div>

    <div class="input-row">
        <textarea id="question" placeholder="Scrivi la tua domanda..." onkeydown="handleKey(event)"></textarea>
        <button id="sendBtn" onclick="sendQuestion()">Invia</button>
    </div>

    <script>
        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        }

        async function sendQuestion() {
            const input = document.getElementById('question');
            const chat = document.getElementById('chat');
            const btn = document.getElementById('sendBtn');
            const question = input.value.trim();
            if (!question) return;

            // Clear placeholder
            if (chat.querySelector('.thinking')) chat.innerHTML = '';

            // Show user message
            chat.innerHTML += `<p><span class="user">Tu:</span> ${escapeHtml(question)}</p>`;
            input.value = '';
            btn.disabled = true;

            // Add assistant placeholder
            const answerEl = document.createElement('p');
            answerEl.innerHTML = '<span class="assistant"><strong>LLaMA:</strong> </span>';
            const textNode = document.createTextNode('');
            answerEl.appendChild(textNode);
            chat.appendChild(answerEl);
            chat.scrollTop = chat.scrollHeight;

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const reader = res.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    textNode.textContent += decoder.decode(value, { stream: true });
                    chat.scrollTop = chat.scrollHeight;
                }
            } catch (err) {
                textNode.textContent = '⚠️ Errore nella connessione con Ollama.';
            }

            btn.disabled = false;
        }

        function escapeHtml(text) {
            return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }
    </script>
</body>
</html>
"""


@app.post("/ask")
async def ask(request: QuestionRequest):
    async def stream_response():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={"model": MODEL, "prompt": request.question, "stream": True},
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        yield data.get("response", "")
                        if data.get("done"):
                            break

    return StreamingResponse(stream_response(), media_type="text/plain")