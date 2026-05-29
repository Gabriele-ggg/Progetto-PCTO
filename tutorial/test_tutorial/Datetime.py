from datetime import datetime
from zoneinfo import ZoneInfo  # Modulo integrato

# Data e ora attuale in un fuso orario specifico
roma_time = datetime.now(ZoneInfo("Europe/Rome"))

day = roma_time.strftime("%A")
hour = roma_time.strftime("%H")
mint = roma_time.strftime("%M")