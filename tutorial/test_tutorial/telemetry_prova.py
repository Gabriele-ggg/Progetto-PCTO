import csv
import time
from datetime import datetime

# ... dentro il gestore della rotta POST /ask ...

# 1. Prendi il timestamp iniziale e l'orario da iniettare
start_time = time.time()
current_time_str = datetime.now().strftime("%H:%M:%S")

# 2. (Qui avviene la manipolazione: Iniezione nel ChatPromptTemplate e filtraggio JSON)
# risposta_llm = catena_rag.invoke({"question": prompt_utente, "current_time": current_time_str})

# 3. Calcola la latenza a fine processo
latency = round(time.time() - start_time, 2)



# log errori azioni...circa
# auditory = Chi ha scritto, utente...circa


# 4. Scrivi la telemetria con csv.writer()
with open("telemetry.csv", mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp log
            "/ask",  # Endpoint
            prompt_utente,  # Prompt salvato per auditing
            current_time_str,  # Ora iniettata nel prompt
            latency,  # Telemetria delle latenze
            0.0,  # Temperatura di inferenza manipolata
            "SUCCESS",  # Status dell'azione
        ]
    )