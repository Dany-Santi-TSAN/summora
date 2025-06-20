"""
Summora - Module d'Extraction de Contenu
Extraction de topics, actions, décisions et insights depuis transcriptions
Usage: python main_extract.py transcription.txt --topics --actions --decisions
"""
import argparse
import json
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Imports spécialisés avec fallbacks

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords, opinion_lexicon, names, extended_omw, wordnet, reuters
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Import Summora utils

try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.core.utils import get_meeting_stopwords
    SUMMORA_UTILS_AVAILABLE = True
except ImportError:
    SUMMORA_UTILS_AVAILABLE = False

def setup_logging(verbose: bool = False, quiet: bool = False):
    """
    Configuration du logging
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level
        ,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ,datefmt='%H:%M:%S'
    )

# Configuration dataclass

@dataclass
class ExtractionConfig:
    """ Configuration pour l'extraction de contenu"""
    #YAKE
    yake_language: str="fr"
    yake_max_ngram: int=3
    yake_duplication_threshold: float=0.7
    yake_top_keyword: int=15

    # TF-IDF
    tfidf_max_features: int=100
    tfidf_ngram_range: tuple=(1,2)
    tfidf_min_df: int=1
    tfidf_max_df: float=0.85

    #Content analysis
    min_text_lenght: int=50
    action_keywords_weight: float=2.0
    decision_keywords_weight: float=2.0

@dataclass
class ExtractionResults:
    """Résultats d'extraction structurés."""

    #Metadata
    source_file: str
    extraction_timestamp: str
    config_used: dict

    #Topics
    yake_keywords: List[Tuple[str,float]] = None
    tfidf_topics: List[Tuple[str,float]] = None
    combined_topics: List[str] = None

    #Actions et décisions
    actions_detected: List[Dict] = None
    decisions_detected: List[Dict] = None

    #Métriqus business
    business_metrics: Dict = None

    #Insights
    meeting_insights: List[str] = None
    recommendations: List[str] = None

    #Status
    extraction_success: bool = False
    processing_time: float=0.0

class DependencyChecker:
    """Vérficateur de dépendances pour l'extraction."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _check_and_log_dependencies(self):
        """Vérifie et log le statut des dépendances."""
        missing_deps = []

        if not YAKE_AVAILABLE:
            self.logger.warning("⚠️ YAKE non disponible")
            self.logger.info("💡 Extraction topics limitée sans YAKE")
            missing_deps.append("yake")

        if not SKLEARN_AVAILABLE:
            self.logger.warning("⚠️ Scikit-learn non disponible")
            self.logger.info("💡 TF-IDF et LDA indisponibles sans scikit-learn")
            missing_deps.append("scikit-learn")

        if not NLTK_AVAILABLE:
            self.logger.warning("⚠️ NLTK non disponible")
            self.logger.info("💡 Tokenisation basique utilisée sans NLTK")
            missing_deps.append("nltk")

        if not SUMMORA_UTILS_AVAILABLE:
            self.logger.warning("⚠️ Summora utils indisponibles - utilisation fallback")

        # Log des fonctionnalités disponibles
        available_features = []
        if YAKE_AVAILABLE:
            available_features.append("YAKE")
        if SKLEARN_AVAILABLE:
            available_features.append("TF-IDF")
        if NLTK_AVAILABLE:
            available_features.append("NLTK")

        if available_features:
            self.logger.info(f"✅ Fonctionnalités disponibles: {', '.join(available_features)}")

        # Suggestion d'installation
        if missing_deps:
            deps_str = " ".join(missing_deps)
            self.logger.info(f"🔧 Installation suggérée: pip install {deps_str}")
            self.logger.info("📋 Ou utilisez: pip install -r requirements.txt")

        return missing_deps

    def get_stopwords(self) -> set:
        """Retourne les stopwords appropriés."""
        if SUMMORA_UTILS_AVAILABLE:
            return get_meeting_stopwords()
        else:
            return self._get_fallback_stopwords()

    def _get_fallback_stopwords(self) -> set:
        """Stopwords de fallback si NLTK/Summora indisponibles."""
        return {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
            'par', 'grand', 'donc', 'alors', 'bien', 'très', 'où', 'du', 'quand', 'mais'
        }

class TopicExtractor:
    """Extracteur de topics spécialisé pour les réunions"""

    def __init__(self, config: ExtractionConfig, stopwords: set):
        self.config = config
        self.stopwords = stopwords
        self.logger = logging.getLogger(__name__)

    def extract_yake_keywords(self, text:str) -> List[Tuple[str, float]]:
        """
        Extrait les mots clés avec YAKE

        Args:
            text: Transcription à analyser

        Retourne:
            Liste de (keyword, scoring) triée par pertinence
        """
        if not YAKE_AVAILABLE:
            self.logger.warning("YAKE non disponible - skip extraction")
            return []

        try:
            kw_extractor = yake.KeywordExtractor(
                lan=self.config.yake_language
                ,n=self.config.yake_max_ngram
                ,dedupLim=self.config.yake_duplication_threshold
                ,top=self.config.yake_top_keyword
            )

            keywords = kw_extractor.extract_keywords(text)

            #Filtrer les keywords en fonction de la taille (trop courts) et des stopwords
            filtered = []
            for kw, score in keywords:
                if len(kw) > 2 and kw.lower() not in self.stopwords:
                    filtered.append((kw, round(score, 4)))

            self.logger.info(f"YAKE: {len(filtered)} keywords extraits")
            return filtered

        except Exception as e:
            self.logger.error(f"Erreur YAKE: {e}")
            return []

    def extract_tfidf_topics(self, text:str) -> List[Tuple[str, float]]:
        """
        Extrait les topics avec TF-IDF.

        Args:
            text: Transcription à analyser

        Retourne:
            Liste de (terme, score) triée par importance
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn non disponible - skip TF-IDF")
            return []

        try:
            #Tokenisation en phrases pour TF-IDF
            if NLTK_AVAILABLE:
                sentences = sent_tokenize(text, language='french')
            else:
            #Fallback simple
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]

            if len(sentences) < 2:
                self.logger.warning("Pas assez de phrases pour TF-IDF")
                return []

            #Configuration TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                min_df=self.config.tfidf_min_df,
                max_df=self.config.tfidf_max_df,
                stop_words=list(self.stopwords)
            )

            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()

            #Calcul la moyenne des scores
            mean_scores = tfidf_matrix.mean(axis=0).A1

            #Tri par score décroissant
            sorted_indices = mean_scores.argsort()[::-1]

            topics = []
            for idx in sorted_indices[:15]:  # Top 15
                if mean_scores[idx] > 0:
                    topics.append((feature_names[idx], round(mean_scores[idx], 4)))

            self.logger.info(f"TF-IDF: {len(topics)} topics extraits")
            return topics

        except Exception as e:
            self.logger.error(f"Erreur TF-IDF: {e}")
            return []

    def combine_topics(self, yake_keywords: List[Tuple[str, float]],
                      tfidf_topics: List[Tuple[str, float]]) -> List[str]:

        """Combine les topics des différentes méthodes."""

        combined_topics = []
        seen_topics = set()

        for kw, score in (yake_keywords[:10] + tfidf_topics[:10]):
            kw_clean = kw.lower().strip()
            if kw_clean not in seen_topics and len(kw_clean) > 2:
                combined_topics.append(kw)
                seen_topics.add(kw_clean)

        return combined_topics

class ActionDecisionDetector:
    """Détecteur d'actions et décisions spécialisé"""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        #Keywords pour les réunions
        self.meeting_keywords = {
            'action': [
                'action', 'tâche', 'faire', 'réaliser', 'livrer', 'assigner',
                'responsable', 'charge de', 'doit', 'va faire', 'prendre en charge',
                'todo', 'à faire', 'next step', 'prochaine étape'
            ],
            'decision': [
                'décision', 'décider', 'valider', 'approuver', 'trancher',
                'choix', 'opter pour', 'retenir', 'adopter', 'accepter',
                'refuser', 'rejeter', 'arbitrage', 'conclusion'
            ],
            'planning': [
                'planning', 'délai', 'échéance', 'deadline', 'calendrier',
                'roadmap', 'timeline', 'avant', 'pour le', 'date limite',
                'livraison', 'fin', 'début', 'lancement'
            ],
            'question': [
                'question', 'souci', 'problème', 'blocage', 'difficulté',
                'bug', 'issue', 'point bloquant', 'interrogation', 'clarification'
            ],
            'agreement': [
                'd\'accord', 'ok', 'parfait', 'exactement', 'entendu',
                'validé', 'approuvé', 'c\'est bon', 'ça marche', 'deal'
            ]
        }

    def detect_actions_decisions(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Détecte les actions et décisions dans la transcription de la réunion.

        Args:
            text: Transcription à analyser

        Retourne:
            Tuple (actions, decisions)
        """
        actions = []
        decisions = []

        #Tokenisation en phrases
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text, language='french')
        else:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]

        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            #Détection d'actions
            action_score = 0
            action_matches = []
            for keyword in self.meeting_keywords['action']:
                if keyword in sentence_lower:
                    action_score += 1
                    action_matches.append(keyword)

            if action_score > 0:
                actions.append({
                    'sentence': sentence.strip(),
                    'sentence_index': i,
                    'keywords_matched': action_matches,
                    'confidence_score': action_score,
                    'type': 'action'
                })

            #Détection de décisions
            decision_score = 0
            decision_matches = []
            for keyword in self.meeting_keywords['decision']:
                if keyword in sentence_lower:
                    decision_score += 1
                    decision_matches.append(keyword)

            if decision_score > 0:
                decisions.append({
                    'sentence': sentence.strip(),
                    'sentence_index': i,
                    'keywords_matched': decision_matches,
                    'confidence_score': decision_score,
                    'type': 'decision'
                })

        self.logger.info(f"Détection: {len(actions)} actions, {len(decisions)} décisions")
        return actions, decisions

class BusinessMetricsCalculator:
    """Calculateur de métriques business spécialisé"""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_business_metrics(self, text: str, actions: List[Dict],
                                 decisions: List[Dict]) -> Dict:

        """
        Calcule des métriques business pour les réunions.

        Args:
            text: Transcrption complète
            actions: Liste des actions détectées (detect_actions_decisions)
            decisions: Liste des décisions détectées (detect_actions_decisions)

        Retourne:
            Dict avec métriques business
        """
        word_count = len(text.split())

        #Densité actionnable
        actionnable_items = len(actions) + len(decisions)
        actionnable_density = (actionnable_items / word_count * 100) if word_count > 0 else 0

        #Ration action/décision
        if len(decisions) > 0:
            action_decision_ratio = len(actions) / len(decisions)
        else:
            action_decision_ratio = len(actions) if len(actions) > 0 else 0

        #Scoring efficacité meeting
        efficiency_score = min(actionnable_density * 10 ,100)

        #Détection de structure avec booléen
        text_lower = text.lower()
        structure_indicators = [
            'ordre du jour', 'pour commencer','agenda', 'point suivant', 'première partie',
            'pour conclure', 'pour finir', 'synthèse', 'résumé', 'bilan', 'next steps'
        ]

        structure_detected = any(indicator in text_lower for indicator in structure_indicators)

        return {
            'word_count': word_count,
            'actionnable_items_count': actionnable_items,
            'actionnable_density_percent': round(actionnable_density, 2),
            'actions_count': len(actions),
            'decisions_count': len(decisions),
            'action_decision_ratio': round(action_decision_ratio, 2),
            'meeting_efficiency_score': round(efficiency_score, 1),
            'has_meeting_structure': structure_detected,
            'avg_action_confidence': round(sum(a['confidence_score'] for a in actions) / len(actions), 2) if actions else 0,
            'avg_decision_confidence': round(sum(d['confidence_score'] for d in decisions) / len(decisions), 2) if decisions else 0
        }
