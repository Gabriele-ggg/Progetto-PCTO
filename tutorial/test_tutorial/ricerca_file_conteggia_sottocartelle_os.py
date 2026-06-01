import os

percorso_partenza = "il/percorso/della/tua/cartella"
conteggio_file = 0

# os.walk restituisce una tupla per OGNI sotto-cartella che trova
for cartella_corrente, sotto_cartelle, files in os.walk(percorso_partenza):
    # 'files' contiene la lista di tutti i file presenti nella cartella_corrente
    conteggio_file += len(files)

print(f"In totale ci sono {conteggio_file} file (incluse le sotto-cartelle).")