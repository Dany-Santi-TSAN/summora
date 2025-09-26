"""Fonction de debbug suite message:
incorrect startxref pointer(1)
"""
from pathlib import Path
import pypdf

docs_path = Path("rag_documents/")

for pdf_file in docs_path.glob("*.pdf"):
    print(f"\n🔍 Test {pdf_file.name}...")
    try:
        with open(pdf_file, 'rb') as file:
            reader = pypdf.PdfReader(file)
            print(f"✅ Pages: {len(reader.pages)}")

            # Test extraction première page
            if reader.pages:
                text = reader.pages[0].extract_text()
                print(f"✅ Texte page 1: {len(text)} chars")

    except Exception as e:
        print(f"❌ CORROMPU: {pdf_file.name} → {e}")
