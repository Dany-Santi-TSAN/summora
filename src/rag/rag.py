"""Module principal d'un RAG léger : chargement, chunking, embeddings et recherche vectorielle avec FAISS"""

import logging
import sys
from typing import List, Dict
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import os
import time
from huggingface_hub import login
from dotenv import load_dotenv

# Path setup pour imports Summora
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import des modules Summora
from src.rag.config import RAGConfig, Chunk
from src.rag.pdf_loader import load_all_pdfs
from src.rag.chunker import chunk_all_documents
from src.rag.embedding import create_embeddings, create_faiss_index

logger = logging.getLogger(__name__)

# Charge du HF_TOKEN depuis depuis .env
load_dotenv()

class RAGlight:
    """Implémenter un RAG léger avec SentenceTransformer et FAISS"""

    def __init__(self, docs_path: str="data/rag/documents", hf_token: str = None):
        # Clé pour HF
        if hf_token is None:
            hf_token = os.getenv('HF_TOKEN')

        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        self.docs_path = docs_path
        self.config = RAGConfig()
        self.model = SentenceTransformer(self.config.model_name)

        self.chunks: List[Chunk] = []
        self.index = None

        logger.info("🤖 Le RAG léger est initialisé...")

    def build_rag_pipeline(self):
        """Consctruit la pipeline RAG complet : load -> chunk -> embeddings -> index"""
        logger.info("🤖 Construction du RAG léger...")
        start_time = time.time()

        # Chargement des documents
        documents = load_all_pdfs(self.docs_path)

        # Chunk les documents
        self.chunks = chunk_all_documents(documents, self.config)
        logger.info(f"✂️ {len(self.chunks)} chunks créés")

        # Embeddings + Indexation
        embeddings = create_embeddings(self.chunks, self.config.model_name)
        self.index = create_faiss_index(embeddings)

        logger.info("✅ Pipeline RAG construite!")
        duration = time.time() - start_time
        logger.info(f"✅ RAG construit en {duration:.2f}s")

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Fonction de requête dans le RAG et retourne une liste de dictionnaire avec les métadonnées"""
        if not self.index:
            logger.error("❌ RAG pas construit")
            return [] # retourne List[Dict] vide

        if not top_k:
            top_k = self.config.top_k

        # Encodage de la requête
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding) # normalisation classique pour le RAG suivant la doc

        # Recherche
        scores, indices = self.index.search(query_embedding, top_k)

        # Résultats
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "rank": i+1
                    ,"score": float(score)
                    ,"text": chunk.text
                    ,"source": chunk.source
                })

        return results

    def query_rag(self, question: str, top_k: int = None) -> Dict:
        """Interface principale. Interroge le RAG et retourne le contexte et la source"""
        logger.info(f"🔍 Query: {question[:50]}...")

        results = self.search(question, top_k)

        if not results:
            return {
                "question": question,
                "answer": "Aucun document pertinent.",
                "method": "rag_léger"
            }

        context = "\n\n".join([r["text"] for r in results])
        sources = list(set([r["source"] for r in results]))

        return {
            "question": question
            ,"context": context
            ,"sources": sources
            ,"method": "rag_léger"
        }


# === Factory function ===
def create_rag(docs_path: str = "data/rag/documents/") -> RAGlight:
    """Instancie et construit un RAG prêt à l'emploi"""
    rag = RAGlight(docs_path)
    rag.build_rag_pipeline()
    return rag

# === Convenience function (interface simplifié) ===
def query_documents(question: str, docs_path: str = "data/rag/documents/") -> Dict:
    """Interroge un dossier de documents grâce un interface de requêtes et retourne la réponse RAG"""
    rag = create_rag(docs_path)
    return rag.query_rag(question)
