import subprocess
import time
import os
import sys
import webbrowser
import signal
import threading
import requests
from pathlib import Path
import socket

# Aggiungi la root directory al path Python
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from backend.services import rag_service

# Lista per tracciare i processi (dict with proc, name, threads, logs)
processes = []

def run_command_async(cmd: str, name: str, cwd: str = None, log_dir: str = None, env: dict = None):
    """Lancia un comando in background e streama stdout/stderr."""
    print(f"\n🚀 Avvio: {name}...\n    -> cmd: {cmd}")
    try:
        if cwd is None:
            cwd = ROOT_DIR
        if log_dir is None:
            log_dir = os.path.join(ROOT_DIR, 'data', 'logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        stdout_log_path = os.path.join(log_dir, f"{name.replace(' ', '_')}_stdout.log")
        stderr_log_path = os.path.join(log_dir, f"{name.replace(' ', '_')}_stderr.log")

        # Eredita l'ambiente corrente e sovrascrive con le variabili extra (cross-platform)
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)

        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        info = {
            'proc': process,
            'name': name,
            'stdout_lines': [],
            'stderr_lines': [],
            'stdout_log': stdout_log_path,
            'stderr_log': stderr_log_path,
            'threads': []
        }

        def _stream_output(stream, log_path, collector, stream_name):
            with open(log_path, 'a', encoding='utf-8') as fh:
                while True:
                    line = stream.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        text = line.rstrip('\n')
                        collector.append(text)
                        print(f"[{name}][{stream_name}] {text}")
                        fh.write(text + "\n")
                        fh.flush()

        t_out = threading.Thread(target=_stream_output, args=(process.stdout, stdout_log_path, info['stdout_lines'], 'OUT'), daemon=True)
        t_err = threading.Thread(target=_stream_output, args=(process.stderr, stderr_log_path, info['stderr_lines'], 'ERR'), daemon=True)

        t_out.start()
        t_err.start()
        info['threads'] = [t_out, t_err]

        processes.append(info)
        print(f"✅ {name} avviato (PID: {process.pid}), log: {stdout_log_path}, {stderr_log_path}")
        return info

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ ERRORE nell'avvio di {name}: {e}")
        return None


def cleanup():
    """Termina tutti i processi."""
    print("\n\n🛑 Arresto dei processi...")
    for info in processes:
        proc = info.get('proc')
        name = info.get('name')
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3)
            print(f"✅ {name} terminato")
        except Exception:
            try:
                proc.kill()
                print(f"❌ {name} forzatamente terminato")
            except Exception:
                print(f"❌ Impossibile terminare {name}")

def signal_handler(sig, frame):
    """Gestisce CTRL+C."""
    print("\n\n⚠️ Interruzione richiesta dall'utente")
    cleanup()
    sys.exit(0)


def check_backend_ready(url: str, timeout: int = 30):
    """Polling dell'endpoint di stato del backend."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as e:
                    print(f"[WARN] JSON non valido: {e}")
                    return {'raw': r.text}
        except Exception as e:
            print(f"[WARN] Errore chiamata a {url}: {e}")

        time.sleep(0.5)
    return None


def diagnostic_check_api(base_url: str):
    """Esegue chiamate diagnostiche al backend per verificare errori comuni (422 ecc.)."""
    status_url = f"{base_url}/api/status"
    ask_url = f"{base_url}/ask"
    print(f"🔎 Diagnostic: controllo {status_url} e test POST su {ask_url}")
    status = None
    try:
        status = requests.get(status_url, timeout=5).json()
        print("✅ /api/status:", status)
    except Exception as e:
        print("❌ Errore chiamando /api/status:", e)

    # Test POST corretto
    try:
        r = requests.post(ask_url, json={"question": "Orario linea 1"}, timeout=10)
        print(f"POST /ask -> status {r.status_code}")
        try:
            print("Response:", r.json())
        except Exception:
            print("Response text:", r.text)
    except Exception as e:
        print("❌ Errore chiamando /ask:", e)

    # Test POST malformato per simulare 422
    try:
        r2 = requests.post(ask_url, data="invalid", timeout=5)
        print(f"POST /ask malformed -> status {r2.status_code}")
        print("Response text:", r2.text[:400])
    except Exception as e:
        print("❌ Errore chiamando /ask con payload malformato:", e)

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════╗")
    print("║         🚀 AVVIO SISTEMA COMPLETO TRAVEL ASSISTANT  🚀 ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    # Registra il gestore del segnale per CTRL+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # --- FASE 0: SETUP INIZIALE ---
    print("\n" + "="*60)
    print("📋 FASE 0: Setup iniziale del sistema (JSON & ChromaDB)")
    print("="*60)
    try:
        rag_service.initialize_system()
        print("✅ Setup dei dati completato!")
    except Exception as e:
        print(f"❌ ERRORE durante il setup: {e}")
        sys.exit(1)
    
    # --- FASE 1: AVVIO BACKEND API ---
    print("\n" + "="*60)
    print("🔧 FASE 1: Avvio Backend API Server (porta 8000)")
    print("="*60)
    # Modello LLM da usare: esclusivamente Ollama.
    # Legge RAG_LLM_MODEL dall'ambiente se già impostato (es. "mistral:7b"),
    # altrimenti usa "rule-based" come fallback sicuro.
    if not os.environ.get('RAG_LLM_MODEL'):
        os.environ['RAG_LLM_MODEL'] = 'rule-based'
        print("⚠️  RAG_LLM_MODEL non impostato → rule-based (fallback)")
        print("   Per usare Ollama: set RAG_LLM_MODEL=mistral:7b  (o altro modello installato)")
    else:
        print(f"🤖 Modello LLM (Ollama): {os.environ['RAG_LLM_MODEL']}")

    # Controlla se la porta 8000 è già occupata; evita il binding doppio che genera WinError 10048
    def port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                return s.connect_ex((host, port)) == 0
            except Exception:
                return False

    # Passa RAG_LLM_MODEL al sottoprocesso uvicorn tramite env (cross-platform)
    _llm_model  = os.environ.get('RAG_LLM_MODEL', 'rule-based')
    backend_env = {'RAG_LLM_MODEL': _llm_model}
    backend_cmd = "python -m uvicorn backend.api_server:app --host 127.0.0.1 --port 8000"
    if port_in_use('127.0.0.1', 8000):
        print("⚠️ Porta 8000 già in uso: salto l'avvio del backend e continuo con il demo.")
        backend_info = None
    else:
        backend_info = run_command_async(backend_cmd, "Backend API", ROOT_DIR, env=backend_env)

    if backend_info is None and not port_in_use('127.0.0.1', 8000):
        cleanup()
        sys.exit(1)

    # Aspetta che il backend sia pronto interrogando /api/status
    print("⏳ Attendendo l'avvio del backend (controllo /api/status)...")
    status = check_backend_ready("http://127.0.0.1:8000/api/status", timeout=30)
    if status:
        print("✅ Backend risponde:", status)
        # Apri API docs
        #webbrowser.open("http://127.0.0.1:8000/docs")
        #webbrowser.open("http://127.0.0.1:8000/api/status")
        # Esegui controlli diagnostici rapidi
        #diagnostic_check_api("http://127.0.0.1:8000")
    else:
        print("❌ Timeout: il backend non ha risposto entro il tempo previsto. Controlla i log.")
    
    # --- FASE 2: AVVIO FRONTEND HTML ---
    print("\n" + "="*60)
    print("🎨 FASE 2: Apertura Frontend Web (HTML)")
    print("="*60)
    
    html_path = os.path.join(ROOT_DIR, "frontend", "index.html")
    if os.path.exists(html_path):
        print(f"✅ File HTML trovato: {html_path}")
        webbrowser.open(f"file:///{html_path}")
        print("✅ Browser aperto!")
    else:
        print(f"⚠️ File HTML non trovato in {html_path}")
        print("💡 Creazione file HTML...")
        # Il file HTML verrà creato automaticamente da questo script
    
    # --- FASE 3: STREAMLIT (OPZIONALE) ---
    print("\n" + "="*60)
    print("✨ Sistema completamente avviato!")
    print("="*60)
    print("\n📍 URL disponibili:")
    print("   🔗 API Documentation: http://127.0.0.1:8000/docs")
    print(f"   🌐 Frontend HTML: file:///{html_path}")
    print("\n⚠️  Premi CTRL+C per arrestare tutti i servizi...\n")
    
    # Mantieni il programma in esecuzione
    try:
        while True:
            time.sleep(1)
            # Verifica se i processi sono ancora in esecuzione
            for info in processes[:]:
                proc = info.get('proc')
                name = info.get('name')
                if proc.poll() is not None:
                    print(f"⚠️  {name} si è arrestato. Return code: {proc.returncode}")
                    # Non riavviare automaticamente il backend: lasciare il problema visibile nei log
                    processes.remove(info)
    except KeyboardInterrupt:
        signal_handler(None, None)