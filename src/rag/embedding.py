"""Module de génération d'embeddings et indexation vectorielle avec FAISS"""

import logging
import sys
from pathlib import Path
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Path setup pour imports Summora
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import module Summora
from src.rag.config import RAGConfig, Chunk

logger = logging.getLogger(__name__)

def create_embeddings(chunks: List[Chunk], model_name: str = None) -> np.ndarray:
    """Fonction de création des embeddings à partir de la
    liste fourni par chunker.py avec SentenceTransformer"""
    if not model_name:
        model_name = RAGConfig().model_name

    logger.info("Création embeddings...")
    model = SentenceTransformer(model_name)

    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts)

    logger.info(f"✅ Embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    return embeddings

def create_faiss_index(embeddings : np.ndarray) -> faiss.IndexFlatIP:
    """Fonction de création de l'indexation FAISS (cosine similarity) à partir des embeddings"""
    logger.info(f"✅ Input embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Normalisation pour cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    logger.info(f"🎯 Index créé: {index.ntotal} vecteurs")
    return index
