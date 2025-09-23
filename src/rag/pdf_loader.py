"""
Chargement des pdf
data/rag/documents
"""
import logging
from pathlib import Path
from typing import List
import pypdf

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrait le texte d'un PDF."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"❌ Erreur PDF {pdf_path}: {e}")
        return ""

def load_all_pdfs(docs_path: str) -> List[dict]:
    """Charge tous les PDFs d'un dossier."""
    docs_path = Path(docs_path)
    pdf_files = list(docs_path.glob("*.pdf"))

    documents = []
    for pdf_file in pdf_files:
        logger.info(f"📄 Chargement {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        if text:
            documents.append({
                "text": text,
                "source": pdf_file.name
            })

    logger.info(f"📚 {len(documents)} documents chargés")
    return documents
