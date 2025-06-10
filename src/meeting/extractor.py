"""
Extracteur de contenu spécialisé pour les meetings
Topics, actions, décisions, sentiment et insights business
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize

from ..core.utils import get_meeting_stopwords

logger = logging.getLogger(__name__)

# === NLP setup : modules, corpus et fallback ===
# v1.2 rajout prévu NLTK : opinion_lexicon, names et reuters

try:
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    import nltk

    try:
        nltk.data.find("corpora/wordnet")
        logger.info("✅ Wordnet déjà présent")
    except LookupError:
        logger.info("⬇️ Téléchargement Wordnet...")
        nltk.download('wordnet', quiet=True)

    try:
        nltk.data.find("corpora/omw-1.4")
        logger.info("✅ omw 1.4 déjà présent")
    except LookupError:
        logger.info("⬇️ Téléchargement omw-1.4...")
        nltk.download('omw-1.4', quiet=True)

    LEMMATIZATION_AVAILABLE = True
    french_stemmer = SnowballStemmer('french')

except ImportError:
    LEMMATIZATION_AVAILABLE = False
    french_stemmer = None
    logger.warning("Lemmatization non disponible - LDA sentiment moins précis")

# === Vérification LDA / sklearn.decomposition ===

try:
    from sklearn.decomposition import LatentDirichletAllocation
    LDA_AVAILABLE = True
    logger.info("✅ LDA importé avec succès")

except ImportError:
    LDA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LDA non disponible - utilisation YAKE + TF-IDF uniquement")

# === Configuration Extractor setup ===

@dataclass
class MeetingExtractionConfig:
    """Configuration pour l'extraction de contenu meeting."""
    # YAKE config pour meetings
    yake_language: str = "fr"
    yake_max_ngram: int = 3
    yake_deduplication_threshold: float = 0.7
    yake_top_keywords: int = 20

    # TF-IDF config pour meetings
    tfidf_max_features: int = 150
    tfidf_ngram_range: tuple = (1, 3)
    tfidf_min_df: int = 1
    tfidf_max_df: float = 0.85

    # LDA config pour meetings
    lda_max_iter: int = 15
    lda_learning_method: str = "batch"
    lda_random_state: int = 42
    lda_words_per_topic: int = 6

    # Pondération meeting-specific
    yake_weight: float = 1.2      # YAKE plus important pour meetings
    tfidf_weight: float = 1.5     # TF-IDF reste référence
    lda_weight: float = 1.0       # LDA moins critique pour meetings
    action_weight: float = 2.0    # Actions prioritaires

    # Méthodes actives
    enabled_methods: List[str] = None
    extract_actions: bool = True
    extract_decisions: bool = True
    extract_sentiment: bool = True
    use_lemmatization: bool = True

    def __post_init__(self):
        if self.enabled_methods is None:
            methods = ['yake', 'tfidf', 'actions', 'decisions']
            if LDA_AVAILABLE:
                methods.append('lda_sentiment')
            self.enabled_methods = methods
