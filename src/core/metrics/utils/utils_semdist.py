"""
SemDist - Fonction d'embeddings de phrases.
Basé sur la doc officielle Kim et al. (2021) et HATS.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_sentence_embeddings(sentences, model_name="default"):
    """
    Génère les embeddings de phrases selon l'approche SemDist officielle.

    Implémente la méthode décrite dans Kim et al. (2021) :
    "calcul d'une similarité cosinus entre la référence et l'hypothèse
    en utilisant des embeddings obtenus au niveau de la phrase"

    Args:
        sentences (list): Liste de phrases à encoder
        model_name (str): Nom du modèle d'embeddings

    Retourne:
        np.array: Matrice d'embeddings (n_sentences, embedding_dim)
    """
    try:
        # Import dynamique comme recommandé
        from sentence_transformers import SentenceTransformer

        # Modèle par défaut simple pour démarrer
        if model_name == "default":
            model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            model = SentenceTransformer(model_name)

        # Génération embeddings
        embeddings = model.encode(sentences)

        logger.debug(f"✅ Embeddings générés: {len(sentences)} phrases, dim={embeddings.shape[1]}")
        return embeddings

    except ImportError:
        raise ImportError("pip install sentence-transformers pour SemDist")
    except Exception as e:
        logger.error(f"❌ Erreur génération embeddings: {e}")
        raise
