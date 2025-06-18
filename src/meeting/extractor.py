"""
Extracteur de contenu spécialisé pour les meetings
Topics, actions, décisions, sentiment et insights business
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk

# from .sentiment_analyzer import MeetingSentimentAnalyzer
# from .action_detector import MeetingActionDetector
# from .decision_detector import MeetingDecisionDetector
# from .topic_extractor import MeetingTopicExtractor

from ..core.utils import get_meeting_stopwords

logger = logging.getLogger(__name__)

# === NLP setup : modules, corpus et fallback ===
# v1.2 rajout prévu NLTK : opinion_lexicon, names et reuters

try:
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

    # Corpus pour sentiment analysis (V1.2)
    try:
        nltk.data.find('corpora/opinion_lexicon')
        logger.info("✅ opinion_lexicon déjà présent")
    except LookupError:
        nltk.download('opinion_lexicon', quiet=True)

    # Corpus pour détection participants (V1.2)
    try:
        nltk.data.find('corpora/names')
        logger.info("✅ corpora names déjà présent")
    except LookupError:
        nltk.download('names', quiet=True)

    LEMMATIZATION_AVAILABLE = True
    SENTIMENT_CORPUS_AVAILABLE = True
    NAMES_CORPUS_AVAILABLE = True
    french_stemmer = SnowballStemmer('french')

except ImportError as e:
    LEMMATIZATION_AVAILABLE = False
    SENTIMENT_CORPUS_AVAILABLE = False
    NAMES_CORPUS_AVAILABLE = False
    french_stemmer = None
    logger = logging.getLogger(__name__)
    logger.warning(f"NLTK resources non disponibles: {e}")

# === Vérification LDA / sklearn.decomposition ===
# LDA pour analyse sentiment avancée

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

# === Orchestrateur principal (Simple et lisible) ===

class MeetingContentExtractor:
    """
    Orchestrateur principal pour l'extraction d'informations structurées à partir de contenus de réunion.

    Ce composant central coordonne plusieurs modules spécialisés afin d'analyser un meeting transcript
    et d'en extraire les éléments clés utiles à la prise de décision.

    Il délègue les tâches suivantes :
    - TopicExtractor : identification des mots-clés et des thématiques abordées
    - ActionDetector : détection des actions à mener, avec attribution aux responsables
    - DecisionDetector : identification des décisions prises et validations formelles
    - SentimentAnalyzer : analyse du ton et des émotions des participants

    Fonctionne comme un pipeline orchestré, à la manière d'un scheduler, chaque module étant responsable
    d'une étape spécifique de l'analyse.
    """

    def __init__(self, config: Optional[MeetingExtractionConfig] = None):
        """
        Initialise l'orchestration avec ses extracteurs spécialisés.

        Args:
            config: Configuration d'extraction (défauts intellignet si None)
        """
        self.config = config or MeetingExtractionConfig()
        self.meeting_stopwords = get_meeting_stopwords()

        # === Vérification des ressources NLP disponibles ===
        self.check_nlp_ressources()

        # === Initialisation des extracteurs spécialisés ===
        self.init_extractors()

        logger.info(f"🎯 Extracteur meeting initialisé")
        logger.info(f"   • Méthodes: {self.config.topic_methods}")
        logger.info(f"   • Lemmatization: {'✅' if LEMMATIZATION_AVAILABLE else '❌'}")
        logger.info(f"   • Sentiment corpus: {'✅' if SENTIMENT_CORPUS_AVAILABLE else '❌'}")
        logger.info(f"   • Names corpus: {'✅' if NAMES_CORPUS_AVAILABLE else '❌'}")

    def _check_nlp_ressources(self) -> None:
        """
        Vérifie et adapte la config selon les ressources NLP disponibles.
        """

        if not LDA_AVAILABLE and self.config.use_lemmatization:
            logger.warning("⚠️ Lemmatization demandée mais NLTK indisponible")
            self.config.use_lemmatization = False

        if not SENTIMENT_CORPUS_AVAILABLE and self.config.extract_sentiment:
            logger.warning("⚠️ Sentiment analysis limitée sans opinion_lexicon")

        if not LDA_AVAILABLE and self.config.topic_methods:
            logger.warning("⚠️ LDA sentiment indisponible, passage en mode YAKE+TF-IDF")
            self.topic_methods = [m for m in self.config.topic_methods if m != 'lda_sentiment']

    def _init_extractors(self) -> None:
        """
        Initialise les extracteurs spécialisés selon la configuration et ressources disponibles.
        """

        self.topic_extractor = MeetingContentExtractor(
            methods=self.config.topic_methods
            ,stopwords=self.meeting_stopwords
            ,weights={
                'yake': self.config.yake_weight
                ,'tfidf': self.config.tfidf_weight
            }
            ,lemmatization_available=LEMMATIZATION_AVAILABLE
            ,french_stemmer=french_stemmer
        )

        # Action detector (si activé)
        if self.config.extract_actions:
            self.action_detector = MeetingActionDetector(
                stopwords=self.meeting_stopwords
                ,weight=self.config.action_weight
            )

        # Décision detector (si activé)
        if self.config.extract_decisions:
            self.decision_detector = MeetingDecisionDetector(
                stopwords=self.meeting_stopwords
            )

        # Sentiment analyzer (si activé et ressources disponibles)
        if self.config.extract_sentiment:
            self.sentiment_analyzer = MeetingSentimentAnalyzer(
                use_lemmatization = self.config.use_lemmatization and LEMMATIZATION_AVAILABLE
                ,sentiment_corpus_available = SENTIMENT_CORPUS_AVAILABLE
                ,lda_available = LDA_AVAILABLE
                ,stopwords=self.meeting_stopwords
                ,french_stemmer=french_stemmer
            )

# === Méthode principale (Simple orchestration) ===

def extract_meeting_content(self, text: str,
                               methods: Optional[List[str]] = None) -> Dict:
    """
    Point d'entrée principal pour l'extraction de contenu meeting.

    Args:
        text: Texte transcrit du meeting
        methods: Méthodes spécifiques à utiliser (override config)

    Retourne:
        Dict: Contenu extrait complet avec tous les insights
    """
    if not text.strip():
        return {"error": "Aucun texte de meeting fourni"}

    methods = methods or self._get_active_methods()

    logger.info(f"🎯 Extraction contenu meeting")
    logger.info(f"📄 Texte: {len(text)} caractères")
    logger.info(f"🔧 Méthodes: {', '.join(methods)}")
    logger.info("-" * 50)

    results = {}

    # === Extraction par composant spécialisé ===

    # 1. Topics et mots-clés
    if any(method in methods for method in ['yake', 'tfidf', 'topics']):
        logger.info("📌 Extraction topics...")
        results['topics'] = self.topic_extractor.extract_topics(text)

    # 2. Actions avec responsables
    if 'actions' in methods and hasattr(self, 'action_detector'):
        logger.info("📌 Détection actions...")
        results['actions'] = self.action_detector.detect_actions(text)

    # 3. Décisions et validations
    if 'decisions' in methods and hasattr(self, 'decision_detector'):
        logger.info("📌 Détection décisions...")
        results['decisions'] = self.decision_detector.detect_decisions(text)

    # 4. Analyse sentiment
    if 'sentiment' in methods and hasattr(self, 'sentiment_analyzer'):
        logger.info("📌 Analyse sentiment...")
        results['sentiment'] = self.sentiment_analyzer.analyze_sentiment(text)

    # === Synthèse finale ===

    if self._has_successful_extractions(results):
        logger.info("🏆 Synthèse insights meeting...")
        results['meeting_summary'] = self._create_meeting_summary(results)
        results['meeting_insights'] = self._generate_insights(results, text)
    else:
        results['meeting_summary'] = {"error": "Aucune extraction réussie"}

    return results

# === Méthodes utilitaires ===

def _get_active_methods(self) -> List[str]:
    """
    Retourne les méthodes actives selon la configuration.
    """

    methods = []

    if self.config.extract_topics:
        methods.extend(self.config.topic_methods)

    if self.config.extract_actions:
        methods.extend('actions')

    if self.config.extract_decisions:
        methods.extend('decisions')

    if self.config.extract_sentiment:
        methods.extend('sentiment')

    return methods

def _has_successful_extractions(self, results:Dict) -> bool:
    """
    Vérifie si au moins une extraction a réussi.
    """
    if not results:
        self.logger.warning("⚠️ Aucun résultat d'extraction fourni")
        return False

    successful = 0
    total_methods = len(results)

    try:
        for method, result in results.items():
            try:
                # Validation stricte
                is_valid = (
                    isinstance(result, dict) and
                    'error' not in result and
                    bool(result) # Dictionnaire non vide
                )

                if is_valid:
                    #Valdidation spécifique par type d'extraction
                    if method == "topics" and "topics" in result:
                        topic_count = len(result['topics'])
                        if topic_count > 0:
                            successful += 1
                            self.logger.debug(f"✅ {method}: {topic_count} topics extraits")
                        else:
                            self.logger.warning(f"⚠️ {method}: aucun topic extrait")

                    elif method == 'actions' and 'actions' in result:
                        action_count = len(result['actions'])
                        if action_count > 0:
                            successful += 1
                            self.logger.debug(f"✅ {method}: {action_count} actions détectées")
                        else:
                            self.logger.warning(f"⚠️ {method}: aucune action détectée")

                else:
                    # Log détaillé des échecs
                    if not isinstance(result, dict):
                        self.logger.warning(f"⚠️ {method}: résultat non-dict ({type(result)})")
                    elif "error" in result:
                        error_msg = result.get("error", "Erreur inconnue")
                        self.logger.warning(f"⚠️ {method}: {error_msg}")
                    elif not result:
                        self.logger.warning(f"⚠️ {method}: résultat vide")

            except Exception as e:
                self.logger.warning(f"⚠️ Erreur validation {method}: {e}")

        # Bilan final avec les métriques
        success_rate = (successful / total_methods) * 100 if total_methods > 0 else 0

        if successful > 0:
            self.logger.info(f"📊 Bilan extractions: {successful}/{total_methods} réussies ({success_rate:.2f}%)")
        else:
            self.logger.error(f"❌ Aucune extraction réussie sur {total_methods} tentatives")

    except Exception as e:
        self.logger.error(f"❌ Erreur critique validation extractions: {e}")
        self.logger.error(f"   Type results: {type(results)}")
        self.logger.error(f"   Contenu: {str(results)[:200]}...")

        return False

def _create_meeting_summary(self, results:Dict) -> Dict:
    """
    Crée un résumé combiné intelligent.
    Délègue la logique complexe aux extracteurs spécialisés.
    """
    summary = {
        "method": "Combinaison multi-extracteurs spécifique aux réunions",
        "extractors_used": list(results.keys()),
        "timestamp": datetime.now().isoformat()
    }

    # Combine les topics si disponibles
    if 'topics' in results:
        summary.update(self.topic_extractor.get_top_topics(results['topics']))

    # Ajoute les insights d'actions si disponibles
    if 'actions' in results and hasattr(self, 'action_detector'):
            summary.update(self.action_detector.get_action_insights(results['actions']))

    return summary

def _generate_insights(self, results : Dict, original_text: str) -> Dict:
    """
    Génère des insights business à partir de tous les résultats (Dict).
    Coordonne les analyseurs spécialisés.
    """

    insights = {
        'analysis_timestamp': datetime.now().isoformat()
        ,'text_length': len(original_text)
        ,'extractors_summary': {}
    }

    # Insighs par extractor
    for extractor_name, extractor_results in results.items():
        if extractor_name in ['topics', 'actions', 'decisions', 'sentiment']:
            extractor = getattr(self, f"{extractor_name.rstrip('s')}_'extractor' if extractor_name=='topics' else 'detector' if extractor_name in ['actions','decisions'] else 'analyzer'")

            if hasattr(extractor,'get_insights'):
                insights["extractors_summary"][extractor_name] = extractor.get_insights(extractor_results)

    # Insights globaux (délégués aux extracteurs)
    if hasattr(self, 'sentiment_analyzer') and 'sentiment' in results:
        insights["overall_sentiment"] = self.sentiment_analyzer.get_overall_sentiment(results["sentiment"])

    return insights


# === Factory functions (Usage simplifié) ===


def create_meeting_extractor(extract_topics: bool = True,
                           extract_actions: bool = True,
                           extract_decisions: bool = True,
                           extract_sentiment: bool = True) -> MeetingContentExtractor:
    """
    Factory pour créer un extracteur avec configuration simple.

    Args:
        extract_topics: Extraire les topics et mots-clés
        extract_actions: Détecter les actions et responsables
        extract_decisions: Détecter les décisions
        extract_sentiment: Analyser le sentiment

    Returns:
        MeetingContentExtractor: Instance configurée
    """
    config = MeetingExtractionConfig(
        extract_topics=extract_topics,
        extract_actions=extract_actions,
        extract_decisions=extract_decisions,
        extract_sentiment=extract_sentiment
    )

    return MeetingContentExtractor(config)

def extract_meeting_content(text: str, **config_kwargs) -> Dict:
    """
    Helper function pour extraction rapide de contenu meeting.

    Args:
        text: Texte transcrit du meeting
        **config_kwargs: Paramètres de configuration

    Returns:
        Dict: Contenu meeting extrait
    """
    config = MeetingExtractionConfig(**config_kwargs) if config_kwargs else None
    extractor = MeetingContentExtractor(config)
    return extractor.extract_meeting_content(text)
