# Guida Estrazione PDF → JSON

## Panoramica
Il file `pipeline_completa.py` è stato aggiornato per generare automaticamente il file JSON leggendo i dati dai file PDF presenti nelle cartelle:
- `./pdf/urbani/` - per gli autobus urbani
- `./pdf/extraurbani/` - per gli autobus extraurbani

## Struttura Cartelle
```
Prova_sito2_prova/
├── pipeline_completa.py
├── index.html
├── requirements.txt
└── pdf/
    ├── urbani/
    │   ├── linea1.pdf
    │   ├── linea2.pdf
    │   └── ...
    └── extraurbani/
        ├── linea380.pdf
        ├── linea390.pdf
        └── ...
```

## Funzionamento
### STEP 0 — Estrazione e Generazione JSON
Il nuovo `step0_genera_json_strutturato()` esegue:

1. **Scansione Cartelle PDF**
   - Legge tutti i file `.pdf` da `./pdf/urbani/`
   - Legge tutti i file `.pdf` da `./pdf/extraurbani/`

2. **Estrazione Testo**
   - Utilizza `pdfplumber` (libreria primaria) per estrazione di qualità
   - Fallback su `PyPDF2` se pdfplumber non disponibile

3. **Parsing Intelligente**
   - Rileva numeri di linea (regex: `Linea 1`, `Line 1`, `1`, `81`)
   - Estrae orari nel formato `HH:MM` o `HH.MM`
   - Identifica fermate (vie, piazze, stazioni, ecc.)
   - Divide in Andata/Ritorno automaticamente

4. **Generazione JSON**
   - Crea struttura identica a `trasporti_non_copiare_prendere_spunto.json`
   - Salva in `trasporti_generato.json`

## Utilizzo

### Esecuzione Automatica
```bash
python pipeline_completa.py
```
- Genera JSON dai PDF in `./pdf/`
- Avvia il server FastAPI su `http://127.0.0.1:8000/`

### Esecuzione con Cartelle Personalizzate
```bash
python pipeline_completa.py --pdf /path/to/pdf/folder --path custom_output.json
```

### Parametri Disponibili
```bash
--path   : Percorso output JSON (default: trasporti_generato.json)
--pdf    : Cartella base con sottocartelle urbani/ e extraurbani/ (default: ./pdf)
--model  : Modello LLM Ollama da usare (default: qwen3.5:9b)
```

## Struttura JSON Generato
```json
{
  "servizio": "Servizio Urbano e Extraurbano",
  "orario_dal": "2026-05-31",
  "linee": [
    {
      "n": "1",
      "tipo": "urbano",
      "a": {
        "p": "Via Chiusaforte - Ospedale - Chiavris - 1°Maggio - FS - Gervasutta",
        "f": [
          {
            "c": "UD100",
            "n": "Via Chiusaforte",
            "ci": "UDINE",
            "v": "Via Chiusaforte",
            "o": ["06:10", "06:25", "06:36", "06:54", "07:05"]
          },
          ...
        ]
      },
      "r": {
        "p": "Gervasutta - FS - 1°Maggio - Chiavris - Ospedale - Via Chiusaforte",
        "f": [...]
      }
    }
  ]
}
```

### Campi JSON
- **n**: Numero linea
- **tipo**: `urbano` o `extraurbano`
- **a**: Direzione Andata
- **r**: Direzione Ritorno (opzionale)
- **p**: Percorso (concatenazione fermate)
- **f**: Array fermate
  - **c**: Codice fermata (es. UD100)
  - **n**: Nome fermata
  - **ci**: Città
  - **v**: Via
  - **o**: Array orari

## Installazione Dipendenze
```bash
pip install -r requirements.txt
```

### Dipendenze Principali
- `pdfplumber` - Estrazione testo da PDF (primaria)
- `pypdf` - Fallback per estrazione PDF
- `langchain*` - Processing testo e embeddings
- `chromadb` - Vector database
- `fastapi` - Server web

## Endpoint API

### `/api/reload` (POST)
Rigenerazione del JSON dai PDF e ricarica del database vettoriale:
```bash
curl -X POST http://localhost:8000/api/reload
```

### `/api/status` (GET)
Stato attuale del database:
```bash
curl http://localhost:8000/api/status
```

Risposta:
```json
{
  "totale_corse": 156,
  "totale_linee": 25,
  "db_pronto": true,
  "embedding": "nomic-embed-text:latest",
  "llm": "qwen3.5:9b",
  "linee": [...]
}
```

### `/api/chat` (POST)
Query RAG per itinerari:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Come raggiungo il centro da via Chiusaforte?"}'
```

## Formati PDF Supportati
Il parser è progettato per estrarre dati da PDF di orari comuni:

✅ Supportato:
- Linee numerate (1, 2, 3, 81, 380, 390, ecc.)
- Fermate come: "Via XXXX", "Piazza YYYY", "Stazione FS"
- Orari in formato HH:MM o HH.MM
- Testo fluido o tabellare

⚠️ Limitazioni:
- PDF scansionati (immagini) richiedono OCR
- Formati molto particolari potrebbero richiedere parsing personalizzato

## Troubleshooting

### "pdfplumber non installato"
```bash
pip install pdfplumber
```

### "Nessuna linea estratta dai PDF"
Possibili cause:
1. PDF sono immagini (richiedono OCR)
2. Formato diverso da quanto atteso
3. Cartella pdf/ non esiste

Soluzione: Controllare il contenuto del PDF e i dati di debug nel output console.

### JSON generato ma vuoto
- Verificare che i PDF contengono testo estraibile
- Controllare la regex di pattern_linea e pattern_orario
- Aggiungere print() nel parsing per debug

## Estensioni Future
- [ ] Supporto OCR per PDF scansionati (Tesseract)
- [ ] Parsing tabellare automatico
- [ ] Validazione dati estratti
- [ ] Export formati alternativi (XML, CSV)
- [ ] Web UI per caricamento PDF

## Notes Sviluppatore
- Modifica `pattern_linea` e `pattern_orario` in `_estrai_linee_da_testo()` per formati personalizzati
- Aggiungi keywords fermate in `_estrai_linee_da_testo()` se necessario
- Usa `_estrai_testo_da_pdf()` direttamente per debug PDF
