"""
Chat CLI con Ollama — legge PDF e JSON dalla directory corrente.
Uso: python chat_cli.py
Dipendenze: pip install pypdf httpx
"""

import json
import httpx
import sys
from pathlib import Path

# ── Configurazione ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3:8b"
BASE_DIR   = Path(__file__).parent   # cartella dove si trova questo script


# ── Lettura dei file di contesto ───────────────────────────────────────────────

def carica_contesto() -> str:
    """
    Cerca tutti i .json e .pdf nella directory dello script
    e li legge per creare un contesto da passare al modello.
    """
    parti = []

    for path in sorted(BASE_DIR.rglob("*")):
        # Salta file nascosti e lo script stesso
        if path.name.startswith(".") or path.name == "chat_cli.py":
            continue

        # ── JSON ──────────────────────────────────────────────────────────────
        if path.suffix.lower() == ".json" and path.is_file():
            try:
                data   = json.loads(path.read_text(encoding="utf-8"))
                pretty = json.dumps(data, ensure_ascii=False, indent=2)
                parti.append(f"[JSON: {path.name}]\n{pretty}")
                print(f"  ✔ JSON: {path.name}")
            except Exception as e:
                print(f"  ✗ Errore JSON {path.name}: {e}")

        # ── PDF ───────────────────────────────────────────────────────────────
        elif path.suffix.lower() == ".pdf" and path.is_file():
            try:
                import pypdf
                testo = []
                with open(path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for i, pagina in enumerate(reader.pages, 1):
                        t = pagina.extract_text() or ""
                        if t.strip():
                            testo.append(f"[Pagina {i}]\n{t.strip()}")
                contenuto = "\n".join(testo) if testo else "(nessun testo estraibile)"
                parti.append(f"[PDF: {path.name}]\n{contenuto}")
                print(f"  ✔ PDF: {path.name} ({len(reader.pages)} pagine)")
            except ImportError:
                print("  ✗ pypdf non trovato. Installa con: pip install pypdf")
            except Exception as e:
                print(f"  ✗ Errore PDF {path.name}: {e}")

    return "\n\n".join(parti)


# ── Invio domanda a Ollama con streaming ───────────────────────────────────────

def chiedi(domanda: str, contesto: str) -> None:
    """
    Manda la domanda ad Ollama (con contesto) e stampa
    la risposta in streaming direttamente nel terminale.
    """
    if contesto:
        prompt = (
            "Hai accesso ai seguenti file. "
            "Usali per rispondere alla domanda in italiano.\n\n"
            f"{contesto}\n\n"
            f"Domanda: {domanda}"
        )
    else:
        prompt = domanda

    print("\nLLaMA: ", end="", flush=True)

    try:
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                OLLAMA_URL,
                json={"model": MODEL, "prompt": prompt, "stream": True},
            ) as risposta:

                if risposta.status_code != 200:
                    print(f"⚠ Errore Ollama (codice {risposta.status_code})")
                    return

                for riga in risposta.iter_lines():
                    if riga:
                        data = json.loads(riga)
                        print(data.get("response", ""), end="", flush=True)
                        if data.get("done"):
                            break

    except httpx.ConnectError:
        print("⚠ Impossibile connettersi a Ollama. Assicurati che sia avviato.")
    except httpx.ReadTimeout:
        print("⚠ Timeout: Ollama sta impiegando troppo tempo.")
    except Exception as e:
        print(f"⚠ Errore: {e}")

    print("\n")  # a capo dopo la risposta


# ── Loop principale ────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Chat CLI con LLaMA 3 — powered by Ollama")
    print("=" * 50)
    print(f"\nCaricamento file da: {BASE_DIR}\n")

    contesto = carica_contesto()

    if contesto:
        print(f"\nContesto caricato ({len(contesto)} caratteri).")
    else:
        print("\nNessun file trovato. Rispondo senza contesto.")

    print("\nScrivi la tua domanda (o 'esci' per uscire).\n")

    while True:
        try:
            domanda = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nArrivederci!")
            sys.exit(0)

        if not domanda:
            continue

        if domanda.lower() in ("esci", "exit", "quit"):
            print("Arrivederci!")
            break

        chiedi(domanda, contesto)


if __name__ == "__main__":
    main()