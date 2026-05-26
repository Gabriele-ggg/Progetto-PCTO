from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("D:\prova_git\pdf\CARNIA_orario_valido_dal_25_maggio_2026.pdf")

pagine = loader.load()

for numero, pagina in enumerate(pagine, start=1):
    print(f"--- PAGINA {numero} ---")
    print(pagina.page_content)
    print("\n")  