import os

# Inserisci qui il percorso della tua cartella
percorso_cartella = "il/percorso/della/tua/cartella"

# Inizializziamo il contatore
conteggio_file = 0

# Ciclo for per scorrere gli elementi nella cartella
for elemento in os.listdir(percorso_cartella):
    # Uniamo il percorso della cartella con il nome dell'elemento
    percorso_completo_elemento = os.path.join(percorso_cartella, elemento)
    
    # Controlliamo se l'elemento è un file
    if os.path.isfile(percorso_completo_elemento):
        conteggio_file += 1

print(f"Ci sono {conteggio_file} file nella cartella.")