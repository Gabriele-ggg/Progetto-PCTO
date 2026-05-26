import httpx
import json
import asyncio
import sys
import hashlib
from pathlib import Path
from typing import List, Dict

# ── CONFIGURAZIONE SISTEMA ────────────────────────────────────────────────────
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3:8b"           
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdf"
DB_JSON_PATH = BASE_DIR / "database_orari.json"

# Massimo testo grezzo da dare a Llama per singola conversione (circa 12.000 caratteri)
MAX_RAW_CHARS = 12000 


# ── FUNZIONI DI UTTIMIZZAZIONE ────────────────────────────────────────────────

def calculate_md5(file_path: Path) -> str:
    """Calcola l'hash MD5 di un file per verificare se è stato modificato."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_existing_database() -> Dict[str, dict]:
    """Carica il database JSON precedente (se esiste) per usarlo come cache."""
    if not DB_JSON_PATH.exists():
        return {}
    try:
        data = json.loads(DB_JSON_PATH.read_text(encoding="utf-8"))
        # Mappa i documenti usando il nome del file come chiave
        return {doc["file_nome"]: doc for doc in data.get("autobus", []) + data.get("treni", [])}
    except Exception:
        return {}


async def struttura_testo_con_ai(testo_grezzo: str, nome_file: str) -> dict:
    """
    Invia il testo grezzo del PDF a Llama chiedendogli di estrarre 
    le informazioni in un formato JSON rigido e pulito.
    """
    prompt_strutturazione = (
        "Analizza il seguente testo estratto da un PDF di orari di trasporto. "
        "Il tuo compito è trasformarlo in un oggetto JSON strutturato.\n"
        "REGOLE RIGIDE:\n"
        "1. Rispondi ESCLUSIVAMENTE con il codice JSON, senza testo prima o dopo, senza blocchi ```json.\n"
        "2. Identifica se si tratta di AUTOBUS o TRENO.\n"
        "3. Estrai: il numero univoco del mezzo (o codice linea), la direzione, le fermate e gli orari di ciascuna fermata.\n\n"
        "Formato richiesto:\n"
        "{\n"
        "  'tipo': 'autobus' o 'treno',\n"
        "  'numero_univoco_mezzo': 'stringa',\n"
        "  'direzione': 'stringa',\n"
        "  'tabelle_orari': [\n"
        "    {\n"
        "      'fermata': 'Nome Fermata',\n"
        "      'orari': ['08:30', '09:15']\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Testo del file '{nome_file}':\n{testo_grezzo}"
    )

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt_strutturazione,
                    "stream": False, # Qui non usiamo lo stream perché ci serve il JSON intero
                    "options": {"temperature": 0.0} # Massima precisione, niente fantasia
                }
            )
            if response.status_code == 200:
                risposta_ai = response.json().get("response", "").strip()
                # Pulizia da eventuali blocchi markdown inseriti dal modello per errore
                if risposta_ai.startswith("```"):
                    risposta_ai = risposta_ai.strip("```json").strip("```").strip()
                return json.loads(risposta_ai)
        except Exception as e:
            print(f"  ⚠️ Errore AI durante la strutturazione di {nome_file}: {e}")
    
    # Fallback in caso di errore dell'AI
    return {"tipo": "autobus", "numero_univoco_mezzo": "Sconosciuto", "direzione": "Sconosciuta", "tabelle_orari": [{"fermata": "Dati Grezzi", "orari": [testo_grezzo[:200]]}]}


# ── CORE: COSTRUZIONE DATABASE STRUTTURATO ────────────────────────────────────

async def build_json_database():
    """
    Scansiona i PDF e usa Llama3 all'avvio per mapparli nel nuovo formato strutturato.
    Divide automaticamente i mezzi in due macro-categorie: 'autobus' e 'treni'.
    """
    print("\n==========================================================")
    print("⚙️ [INDICIZZAZIONE] Generazione Database Strutturato (Autobus vs Treni)...")
    print("==========================================================")

    if not PDF_DIR.exists() or not PDF_DIR.is_dir():
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📁 Creazione cartella '{PDF_DIR.name}/'. Inserisci i PDF e riavvia.")
        return

    pdf_files = sorted(list(PDF_DIR.rglob("*.pdf")))
    if not pdf_files:
        print("⚠️ Nessun file PDF trovato. Genero un database vuoto.")
        DB_JSON_PATH.write_text(json.dumps({"autobus": [], "treni": []}), encoding="utf-8")
        return

    cache_database = load_existing_database()
    
    # Nuova struttura divisa richiesta
    nuovo_database = {
        "info": "Database TPL FVG diviso per tipologia di mezzo, fermate ed orari",
        "autobus": [],
        "treni": []
    }

    for path in pdf_files:
        try:
            current_md5 = calculate_md5(path)
            
            # 1. Uso della Cache per non risvegliare Llama sui file già fatti
            if path.name in cache_database and cache_database[path.name].get("md5") == current_md5:
                print(f"⚡ [CACHE] {path.name} già strutturato.")
                doc_salvato = cache_database[path.name]
                tipo_lista = "treni" if doc_salvato.get("tipo") == "treno" else "autobus"
                nuovo_database[tipo_lista].append(doc_salvato)
                continue

            # 2. Lettura del PDF
            print(f"📄 [ESTRAZIONE] Lettura testo da: {path.name}...")
            import pypdf
            text_pages = []
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text_pages.append(page.extract_text() or "")
            
            testo_grezzo = "\n".join(text_pages)[:MAX_RAW_CHARS]

            # 3. L'AI trasforma il testo disordinato nel JSON perfetto
            print(f"🧠 [AI STRUCTURING] Llama3 sta classificando e strutturando {path.name}...")
            dati_strutturati = await struttura_testo_con_ai(testo_grezzo, path.name)
            
            # Arricchiamo l'entry con i metadati del file per la cache
            dati_strutturati["file_nome"] = path.name
            dati_strutturati["md5"] = current_md5
            
            # 4. Smistamento automatico in base al tipo restituito dall'AI
            if dati_strutturati.get("tipo") == "treno":
                nuovo_database["treni"].append(dati_strutturati)
                print(f"  ✔️ Classificato come TRENO e aggiunto al database.")
            else:
                nuovo_database["autobus"].append(dati_strutturati)
                print(f"  ✔️ Classificato come AUTOBUS e aggiunto al database.")

        except Exception as e:
            print(f"  ❌ Errore durante l'elaborazione di {path.name}: {e}")

    # Scrittura finale sul file JSON
    try:
        DB_JSON_PATH.write_text(json.dumps(nuovo_database, indent=2, ensure_ascii=False), encoding="utf-8")
        print("==========================================================")
        print(f"💾 [COMPLETATO] '{DB_JSON_PATH.name}' salvato con la nuova struttura!")
        print("==========================================================\n")
    except Exception as e:
        print(f"❌ Impossibile salvare il file JSON: {e}")


def load_context_from_generated_json() -> str:
    """Prepara il contesto leggendo dal nuovo JSON diviso."""
    if not DB_JSON_PATH.exists():
        return ""
    try:
        data = json.loads(DB_JSON_PATH.read_text(encoding="utf-8"))
        contesto_linee = []
        
        # Carica sezione Autobus
        for bus in data.get("autobus", []):
            contesto_linee.append(
                f"[AUTOBUS] Linea/Mezzo N°: {bus.get('numero_univoco_mezzo')}\n"
                f"Direzione: {bus.get('direzione')}\n"
                f"Fermate e Orari:\n{json.dumps(bus.get('tabelle_orari'), ensure_ascii=False)}"
            )
            
        # Carica sezione Treni
        for treno in data.get("treni", []):
            contesto_linee.append(
                f"[TRENO] Codice: {treno.get('numero_univoco_mezzo')}\n"
                f"Direzione: {treno.get('direzione')}\n"
                f"Fermate e Orari:\n{json.dumps(treno.get('tabelle_orari'), ensure_ascii=False)}"
            )
            
        return "\n\n".join(contesto_linee)
    except Exception as e:
        print(f"❌ Errore caricamento contesto JSON: {e}")
        return ""


# ── INTERAZIONE CHAT CON OLLAMA ───────────────────────────────────────────────

async def ask_ollama(question: str, context: str):
    if context:
        prompt = (
            "Sei l'Assistente Virtuale di TPL FVG. Rispondi alle domande dei passeggeri "
            "basandoti esclusivamente su questo database strutturato di autobus e treni:\n\n"
            f"{context}\n\n"
            f"Richiesta del passeggero: {question}"
        )
    else:
        prompt = question

    print("\n" + "─" * 60)
    print(f"▶️ [RICHIESTA] {question}")
    print("─" * 60)
    print("🧠 [AI THINKING] Ricerca nel database strutturato e generazione risposta...\n")

    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": MODEL, 
                    "prompt": prompt, 
                    "stream": True,
                    "options": {"num_ctx": 950000000} 
                },
            ) as response:
                
                if response.status_code != 200:
                    print(f"❌ Errore Ollama: {response.status_code}")
                    return

                print("🚌 Risposta per il viaggio:")
                print("─" * 40)
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            print(data.get("response", ""), end="", flush=True)
                            if data.get("done"):
                                print("\n" + "─" * 40 + "\n")
                                break
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            print(f"\n❌ Errore di flusso: {e}\n")


async def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      🚌 TPL FVG — Assistente Avanzato Strutturato       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Avvio asincrono corretto per la chiamata AI di strutturazione
    await build_json_database()
    global_context = load_context_from_generated_json()

    print("Esplora gli orari inserendo una domanda (es. 'A che ora passa il bus per Udine?'):")
    print("  /rigenera  — Svuota la cache e riclassifica i PDF")
    print("  /esci      — Chiudi il programma\n")

    while True:
        try:
            question = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("/esci", "exit", "quit"):
            break
        elif question.lower() == "/rigenera":
            if DB_JSON_PATH.exists():
                DB_JSON_PATH.unlink()
            await build_json_database()
            global_context = load_context_from_generated_json()
        else:
            await ask_ollama(question, global_context)


if __name__ == "__main__":
    asyncio.run(main())