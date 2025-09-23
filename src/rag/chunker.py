"""Définit le découpage en chunk pour la pipeline du RAG"""
import sys
from pathlib import Path
from typing import List

# Path setup pour imports Summora
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.config import RAGConfig, Chunk

def simple_chunking(text: str, source: str, config: RAGConfig = None) -> List[Chunk]:
    """Découpe le texte en chunks définit avec chunk_size et overlap"""
    if not config:
        config = RAGConfig()

    chunks = []
    words = text.split()

    chunk_id = 0
    for i in range(0, len(words), config.chunk_size - config.overlap):
        chunk_words = words[i:i + config.chunk_size]
        chunk_text = " ".join(chunk_words)

        if len(chunk_text.strip()) > 50:
            chunks.append(Chunk(
                text=chunk_text.strip()
                ,source=source
                ,chunk_id=chunk_id
            ))
            chunk_id += 1

    return chunks

def chunk_all_documents(documents: List[dict], config: RAGConfig = None) -> List[Chunk]:
    """Découpe en chunks tous les documents fournis et retourne une liste complète"""
    all_chunks = []

    for doc in documents:
        chunks = simple_chunking(doc["text"], doc["source"], config)
        all_chunks.extend(chunks)

    return all_chunks
