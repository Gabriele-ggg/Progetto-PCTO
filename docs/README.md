README - Documentazione aggiuntiva

Esempio: controllo di `robots.txt` prima di effettuare scraping su siti approvati dallo sviluppatore.

1) Importa l'helper:

	from backend.services.robots import fetch_with_robots, DisallowedByRobots

2) Esempio di utilizzo:

	allowed = ['example.com', 'altrousito.it']
	try:
		resp = fetch_with_robots('https://example.com/some/page', allowed_hosts=allowed, user_agent='PCTO-Scraper')
		content = resp.text
	except DisallowedByRobots:
		print('Accesso vietato da robots.txt')
	except ValueError as e:
		print('Host non autorizzato:', e)

Nota: Lo sviluppatore deve definire esplicitamente `allowed_hosts` per evitare scraping accidentale di domini non approvati.
