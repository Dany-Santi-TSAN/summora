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


#### Import Summora utils - Version mise à jour ####

try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.core.utils import get_meeting_stopwords

    # ✅ NOUVEAU: Import fallback depuis module spécialisé
    try:
        from src.core.utils_stopwords_meeting_fr import get_fallback_stopwords_fr
        FALLBACK_STOPWORDS_AVAILABLE = True
    except ImportError:
        FALLBACK_STOPWORDS_AVAILABLE = False

    SUMMORA_UTILS_AVAILABLE = True
except ImportError:
    SUMMORA_UTILS_AVAILABLE = False
    FALLBACK_STOPWORDS_AVAILABLE = False


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
    yake_top_keywords: int=15

    # TF-IDF
    tfidf_max_features: int=100
    tfidf_ngram_range: tuple=(1,2)
    tfidf_min_df: int=1
    tfidf_max_df: float=0.85

    #Content analysis
    min_text_length: int=50
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
            if FALLBACK_STOPWORDS_AVAILABLE:
                self.logger.info("✅ Fallback stopwords disponible")
            else:
                self.logger.warning("⚠️ Fallback stopwords aussi indisponible")

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


        return missing_deps

    def get_stopwords(self) -> set:
        """Retourne les stopwords appropriés avec fallaback intelligent"""

        # Priorité 1: Summora utils complet (NLTK + meetings)
        if SUMMORA_UTILS_AVAILABLE:
           try:
               stopwords = get_meeting_stopwords()
               self.logger.info(f"✅ Stopwords Summora: {len(stopwords)} mots")
               return stopwords
           except Exception as e:
               self.logger.warning(f"Erreur stopwords Summora: {e}")

        # Priorité 2: Module spécialisé fallback
        if FALLBACK_STOPWORDS_AVAILABLE:
            try:
                from src.core.utils_stopwords_meeting_fr import get_fallback_stopwords_fr
                stopwords = get_fallback_stopwords_fr()
                self.logger.info(f"✅ Stopwords fallback spécialisés: {len(stopwords)} mots")
                return stopwords
            except Exception as e:
                self.logger.warning(f"Erreur stopwords fallback: {e}")

        # Priorité 3: NLTK seul
        if NLTK_AVAILABLE:
            try:
                from nltk.corpus import stopwords
                french_stopwords = set(stopwords.words('french'))
                self.logger.info(f"✅ Stopwords NLTK français: {len(french_stopwords)} mots")
                return french_stopwords
            except Exception as e:
                self.logger.warning(f"Erreur stopwords NLTK: {e}")

        # Priorité 4: (dernier recours) Fallback minimal
        self.logger.warning("⚠️ Utilisation fallback minimal - qualité topics dégradée")
        return self._get_minimal_fallback_stopwords()

    def _get_minimal_fallback_stopwords(self) -> set:
        """Fallback minimal en dernier recours."""

        return {
            # Essentiels français
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout',

            # Contractions critiques
            "c'est", "qu'il", "j'ai", "qu'on", "bon", "oui", "non", "bien", "fin",

            # Connecteurs meetings
            "donc", "alors", "voici", "voilà", "enfin", "bref"

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
                ,top=self.config.yake_top_keywords
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

        # Import vocabulaire business centralisé
        try:
            import sys
            from pathlib import Path
            if str(Path(__file__).parent.parent.parent) not in sys.path:
                sys.path.append(str(Path(__file__).parent.parent.parent))

            from src.core.business_vocabulary import BUSINESS_KEYWORDS
            self.meeting_keywords = BUSINESS_KEYWORDS
            self.logger.info(f"📋 Vocabulaire business: {len(BUSINESS_KEYWORDS)} catégories")

        except ImportError as e:
            self.logger.warning(f"⚠️ Import vocabulaire centralisé échoué: {e}")
            self.meeting_keywords = self._get_fallback_keywords()

    def _get_fallback_keywords(self) -> dict:
        """Keywords de secours si module centralisé indisponible."""
        return {
            'actions': [
                'action', 'tâche', 'faire', 'réaliser', 'livrer', 'assigner',
                'responsable', 'charge de', 'doit', 'va faire', 'prendre en charge',
                'todo', 'à faire', 'next step', 'prochaine étape'
            ],
            'decisions': [
                'décision', 'décider', 'valider', 'approuver', 'trancher',
                'choix', 'opter pour', 'retenir', 'adopter', 'accepter',
                'refuser', 'rejeter', 'arbitrage', 'conclusion'
            ]
            # ... autres catégories
        }

    def detect_actions_decisions(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Détecte les actions et décisions dans la transcription.

        Args:
            text: Transcription à analyser

        Retourne:
            Tuple (actions, decisions)
        """
        actions = []
        decisions = []

        # Tokenisation en phrases
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text, language='french')
        else:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]

        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            # Détection d'actions
            action_score = 0
            action_matches = []
            action_keywords = self.meeting_keywords.get('actions', [])

            for keyword in action_keywords:
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

            # Détection de décisions
            decision_score = 0
            decision_matches = []
            decision_keywords = self.meeting_keywords.get('decisions', [])

            for keyword in decision_keywords:
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

    def get_available_categories(self) -> List[str]:
        """
        Retourne les catégories disponibles dans le vocabulaire.

        Retourne:
            List[str]: Liste des catégories business
        """
        return list(self.meeting_keywords.keys())

    def analyze_keyword_distribution(self, text: str) -> Dict[str, int]:
        """
        Analyse la répartition des mots-clés par catégorie.

        Args:
            text: Texte à analyser

        Retourne:
            Dict: Comptage par catégorie
        """
        text_lower = text.lower()
        distribution = {}

        for category, keywords in self.meeting_keywords.items():
            count = 0
            for keyword in keywords:
                count += text_lower.count(keyword)
            distribution[category] = count

        return distribution

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

class MeetingInsightsGenerator:
    """Générateur d'insights et recommandations spécialisé."""
    # V1 pour une génération de recommandation rules based simple
    # V2 expérimentation pour un LLM contextuel 🤖

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_insights(self, metrics: Dict, actions: List[Dict],
                         decisions: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Génère des insights et recommandations.

        Args:
            metrics: Métriques business (calculate_business_metrics)
            actions: Actions détectées (detect_actions_decisions)
            decisions: Décisions détectées (detect_actions_decisions)

        Retourne:
            Tuple (insights, recommendations)
        """
        insights = []
        recommendations = []

        #Insights sur l'efficacité
        efficiency = metrics['meeting_efficiency_score']
        if efficiency >= 80:
            insights.append(f"Meeting très efficace (score: {efficiency}%)")
        elif efficiency >= 60:
            insights.append(f"Meeting d'efficacité correcte (score: {efficiency}%)")
        else:
            insights.append(f"Meeting peu efficace (score: {efficiency}%)")
            recommendations.append("Améliorer la structure et définir des objectifs clairs")

        #Insights sur les actions
        if metrics['actions_count'] == 0:
            insights.append("Aucune action concrète identifiée")
            recommendations.append("Définir des actions claires avec responsables et échéances")
        elif metrics['actions_count'] > 10:
            insights.append(f"Beaucoup d'actions identifiées ({metrics['actions_count']})")
            recommendations.append("Prioriser les actions les plus importantes")
        else:
            insights.append(f"{metrics['actions_count']} action(s) identifiée(s)")

        #Insights sur les décisions
        if metrics['decisions_count'] == 0:
            insights.append("Aucune décision claire prise")
            recommendations.append("Clarifier les décisions prises et les communiquer")
        else:
            insights.append(f"{metrics['decisions_count']} décision(s) prise(s)")

        #Insights sur la structure
        if not metrics['has_meeting_structure']:
            insights.append("Pas de structure meeting détectée")
            recommendations.append("Utiliser un agenda structuré pour les prochains meetings")
        else:
            insights.append("Meeting structuré avec agenda")

        #Insights sur l'équilibre action/décision
        ratio = metrics['action_decision_ratio']
        if ratio > 5:
            insights.append("Beaucoup d'actions par rapport aux décisions")
            recommendations.append("S'assurer que les décisions précèdent les actions")
        elif ratio < 0.5 and metrics['decisions_count'] > 0:
            insights.append("Peu d'actions concrètes suite aux décisions")
            recommendations.append("Traduire les décisions en actions concrètes")

        return insights, recommendations

# ==============================================================================================
#                                 === Orchestrateur ===
# ==============================================================================================

class MeetingContentExtractor:
    """Orchestrateur principal pour l'extraction de contenu d'une transcription de réunion"""

    def __init__(self, config: Optional[ExtractionConfig]=None):
        """
        Initialise l'extracteur avec ses composants spécialisés.

        Args:
            config: Configuration d'extraction
        """
        self.config = config or ExtractionConfig()
        self.logger = logging.getLogger(__name__)

        #Vérification des dépendances
        self.dependency_checker = DependencyChecker()
        self.dependency_checker._check_and_log_dependencies()

        #Initialisation des composants spécialisés
        stopwords = self.dependency_checker.get_stopwords()
        self.topic_extractor = TopicExtractor(self.config, stopwords)
        self.actionnable_detector = ActionDecisionDetector(self.config)
        self.metrics_calculator = BusinessMetricsCalculator(self.config)
        self.insights_generator = MeetingInsightsGenerator(self.config) # V1 reco : rules based simple

    def extract_content(self, text: str) -> ExtractionResults:
        """
        Extrait toute l'analyse du contenu à partir de la transcription de réunion via orchestration.

        Args:
            text: Texte de transcription

        Retourne:
            ExtractionResults avec tous les éléments extraits
        """
        start_time = datetime.now()

        if len(text) < self.config.min_text_length:
            self.logger.error(f"Texte trop court ({len(text)} caractères)")
            return ExtractionResults(
                source_file="",
                extraction_timestamp=datetime.now().isoformat(),
                config_used=asdict(self.config),
                extraction_success=False
            )

        try:
            self.logger.info("⏱️ Démarrage extraction de contenu...")

            # 1. Extraction topics via TopicExtractor
            self.logger.info("📊 Extraction topics...")
            yake_keywords = self.topic_extractor.extract_yake_keywords(text)
            tfidf_topics = self.topic_extractor.extract_tfidf_topics(text)
            combined_topics = self.topic_extractor.combine_topics(yake_keywords, tfidf_topics)

            # 2. Détection actions/décisions via ActionDecisionDetector
            self.logger.info("🔥 Détection actions et décisions...")
            actions, decisions = self.actionnable_detector.detect_actions_decisions(text)

            # 3. Métriques business via BusinessMetricsCalculator
            self.logger.info("📊 Calcul métriques business...")
            business_metrics = self.metrics_calculator.calculate_business_metrics(text, actions, decisions)

            # 4. Insights via MeetingInsightsGenerator
            self.logger.info("💡 Génération insights...") # v1
            insights, recommendations = self.insights_generator.generate_insights(business_metrics, actions, decisions)

            # Temps de traitement
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Log intelligent selon la durée
            if processing_time >= 60:
                minutes = int(processing_time // 60)
                seconds = int(processing_time % 60)
                time_str = f"{minutes}m {seconds}s" if seconds > 0 else f"{minutes}m"
            else:
                time_str = f"{processing_time:.2f}s"

            self.logger.info(f"⚡ Extraction terminée avec succès en {time_str}")

            return ExtractionResults(
                source_file=""
                ,extraction_timestamp=datetime.now().isoformat()
                ,config_used=asdict(self.config)
                ,yake_keywords=yake_keywords
                ,tfidf_topics=tfidf_topics
                ,combined_topics=combined_topics
                ,actions_detected=actions
                ,decisions_detected=decisions
                ,business_metrics=business_metrics
                ,meeting_insights=insights
                ,recommendations=recommendations
                ,extraction_success=True
                ,processing_time=processing_time
            )

        except Exception as e:
            self.logger.error(f"❌ Erreur extraction: {e}")
            import traceback
            traceback.print_exc()

            return ExtractionResults(
                source_file=""
                ,extraction_timestamp=datetime.now().isoformat()
                ,config_used=asdict(self.config)
                ,extraction_success=False
            )

def load_transcription_file(file_path: Path) -> Tuple[str, Dict]:
    """
    Charge un fichier de transcription (txt ou json).

    Args:
        file_path: Chemin vers le fichier

    Retourne:
        Tuple (text, metadata)
    """
    try:
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'content' in data and 'text' in data['content']:
                    return data['content']['text'], data
                else:
                    return data.get('text', ''), data
        else:
            #Fichier .txt - extraction du texte après les métadonnées
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            #Chercher la section transcription
            if "TRANSCRIPTION COMPLÈTE" in content:
                text_start = content.find("TRANSCRIPTION COMPLÈTE")
                text_start = content.find("\n", text_start) + 1
                text = content[text_start:].strip()

                #Supprimer le footer si présent
                if "Généré par Summora" in text:
                    text = text.split("Généré par Summora")[0].strip()

                return text, {"source": "txt_file"}
            else:
                #Fichier texte simple
                return content.strip(), {"source": "simple_txt"}

    except Exception as e:
        raise Exception(f"Erreur lecture fichier {file_path}: {e}")

def save_extraction_results(results: ExtractionResults, output_path: Optional[Path] = None) -> Path:
    """
    Sauvegarde les résultats d'extraction.

    Args:
        results: Résultats à sauvegarder
        output_path: Chemin de sortie (optionnel)

    Retourne:
        Path: Chemin du fichier sauvegardé
    """
    if output_path:
        save_path = output_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path("output/extractions") / f"extraction_results_{timestamp}.json"

    #Créer le dossier si nécessaire
    save_path.parent.mkdir(parents=True, exist_ok=True)

    #Conversion en dict pour JSON
    results_dict = asdict(results)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    return save_path

def print_extraction_summary(results: ExtractionResults, source_file: str):
    """Affiche un résumé des résultats d'extraction."""
    print("\n" + "="*60)
    print("🔍 SUMMORA - RÉSULTATS EXTRACTION")
    print("="*60)

    if not results.extraction_success:
        print("❌ Extraction échouée")
        return

    #Info générale
    print(f"📁 Fichier source: {source_file}")
    print(f"⏱️  Temps traitement: {results.processing_time:.2f}s")

    #Topics
    if results.combined_topics:
        print(f"\n🎯 TOPICS PRINCIPAUX ({len(results.combined_topics)})")
        for i, topic in enumerate(results.combined_topics[:10], 1):
            print(f"   {i:2d}. {topic}")

    #Actions
    if results.actions_detected:
        print(f"\n🎯 ACTIONS DÉTECTÉES ({len(results.actions_detected)})")
        for i, action in enumerate(results.actions_detected[:5], 1):
            conf = action['confidence_score']
            sentence = action['sentence'][:80] + "..." if len(action['sentence']) > 80 else action['sentence']
            print(f"   {i:2d}. [{conf}] {sentence}")

        if len(results.actions_detected) > 5:
            print(f"   ... et {len(results.actions_detected) - 5} autres actions")

    #Décisions
    if results.decisions_detected:
        print(f"\n🚀 DÉCISIONS DÉTECTÉES ({len(results.decisions_detected)})")
        for i, decision in enumerate(results.decisions_detected[:5], 1):
            conf = decision['confidence_score']
            sentence = decision['sentence'][:80] + "..." if len(decision['sentence']) > 80 else decision['sentence']
            print(f"   {i:2d}. [{conf}] {sentence}")

    #Métriques business
    if results.business_metrics:
        metrics = results.business_metrics
        print(f"\n📊 MÉTRIQUES BUSINESS")
        print(f"   • Efficacité meeting: {metrics['meeting_efficiency_score']}%")
        print(f"   • Densité actionnable: {metrics['actionnable_density_percent']}%")
        print(f"   • Structure détectée: {'Oui' if metrics['has_meeting_structure'] else 'Non'}")
        print(f"   • Actions/Décisions: {metrics['actions_count']}/{metrics['decisions_count']}")

    #Insights
    if results.meeting_insights:
        print(f"\n💡 INSIGHTS")
        for i, insight in enumerate(results.meeting_insights, 1):
            print(f"   {i:2d}. {insight}")

    #Recommandations
    if results.recommendations:
        print(f"\n👀 RECOMMANDATIONS")
        for i, rec in enumerate(results.recommendations, 1):
            print(f"   {i:2d}. {rec}")

    print("="*60)

def main():
    """Point d'entrée principal du module d'extraction."""

    parser = argparse.ArgumentParser(
        description="Summora - Module d'Extraction de Contenu"
        ,formatter_class=argparse.RawDescriptionHelpFormatter
        ,epilog=

        """
    Exemples d'usage:
    python main_extract.py transcription.txt                      # Extraction complète
    python main_extract.py results.json --output extract.json     # Depuis JSON
    python main_extract.py transcription.txt --no-actions         # Sans actions
    python main_extract.py transcription.txt --yake-only          # YAKE uniquement
        """
    )

    # Arguments obligatoires
    parser.add_argument(
        "input_file"
        ,type=str
        ,help="Fichier de transcription (.txt ou .json)"
    )

    # Options d'extraction
    parser.add_argument(
        "--topics"
        ,action="store_true"
        ,help="Extraire seulement les topics"
    )

    parser.add_argument(
        "--actions"
        ,action="store_true"
        ,help="Extraire seulement les actions"
    )

    parser.add_argument(
        "--decisions"
        ,action="store_true"
        ,help="Extraire seulement les décisions"
    )

    parser.add_argument(
        "--yake-only"
        ,action="store_true"
        ,help="Utiliser seulement YAKE (pas TF-IDF)"
    )

    parser.add_argument(
        "--no-actions"
        ,action="store_true"
        ,help="Skip l'extraction d'actions"
    )

    parser.add_argument(
        "--no-decisions"
        ,action="store_true"
        ,help="Skip l'extraction de décisions"
    )

    # Options de configuration
    parser.add_argument(
        "--max-keywords"
        ,type=int
        ,default=15
        ,help="Nombre max de keywords (défaut: 15)"
    )

    parser.add_argument(
        "--ngram-max"
        ,type=int
        ,default=3
        ,help="N-gram maximum pour YAKE (défaut: 3)"
    )

    # Options de sortie
    parser.add_argument(
        "--output", "-o"
        ,type=str
        ,help="Fichier de sortie JSON"
    )

    parser.add_argument(
        "--no-save"
        ,action="store_true"
        ,help="Ne pas sauvegarder les résultats"
    )

    # Options système
    parser.add_argument(
        "--verbose", "-v"
        ,action="store_true"
        ,help="Mode verbeux"
    )

    parser.add_argument(
        "--quiet", "-q"
        ,action="store_true"
        ,help="Mode silencieux"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)

    try:
        # Validation du fichier d'entrée
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"❌ Fichier non trouvé: {input_path}")
            return 1

        if not args.quiet:
            print("🔍 SUMMORA - MODULE EXTRACTION")
            print(f"📁 Fichier: {input_path.name}")
            print("-" * 50)

        # Configuration d'extraction
        config = ExtractionConfig(
            yake_top_keywords=args.max_keywords,
            yake_max_ngram=args.ngram_max
        )

        # Chargement du fichier
        if not args.quiet:
            logger.info(f"📖 Chargement fichier: {input_path.name}")

        text, metadata = load_transcription_file(input_path)

        if len(text.strip()) < 10:
            print(f"❌ Fichier vide ou texte insuffisant")
            return 1

        if not args.quiet:
            word_count = len(text.split())
            logger.info(f"📝 Texte chargé: {word_count} mots")

        # Création de l'extracteur (orchestrateur)
        extractor = MeetingContentExtractor(config)

        # Extraction du contenu
        if not args.quiet:
            print("🔄 Extraction en cours...")

        results = extractor.extract_content(text)
        results.source_file = str(input_path)

        if not results.extraction_success:
            print("❌ Échec de l'extraction")
            return 1

        # Filtrage selon les options
        if args.topics:
            # Garder seulement les topics
            results.actions_detected = []
            results.decisions_detected = []
            results.business_metrics = None
            results.meeting_insights = []
            results.recommendations = []

        if args.actions:
            # Garder seulement les actions
            results.yake_keywords = []
            results.tfidf_topics = []
            results.combined_topics = []
            results.decisions_detected = []

        if args.decisions:
            # Garder seulement les décisions
            results.yake_keywords = []
            results.tfidf_topics = []
            results.combined_topics = []
            results.actions_detected = []

        if args.no_actions:
            results.actions_detected = []

        if args.no_decisions:
            results.decisions_detected = []

        if args.yake_only:
            results.tfidf_topics = []
            # Reconstruire combined_topics avec seulement YAKE
            results.combined_topics = [kw for kw, score in results.yake_keywords[:10]]

        # Sauvegarde des résultats
        if not args.no_save:
            output_path = Path(args.output) if args.output else None
            saved_path = save_extraction_results(results, output_path)
            if not args.quiet:
                logger.info(f"💾 Résultats sauvegardés: {saved_path}")

        # Affichage des résultats
        if not args.quiet:
            print_extraction_summary(results, input_path.name)
        elif args.topics and results.combined_topics:
            # Mode quiet + topics only
            for topic in results.combined_topics[:10]:
                print(topic)
        elif args.actions and results.actions_detected:
            # Mode quiet + actions only
            for action in results.actions_detected:
                print(action['sentence'])
        elif args.decisions and results.decisions_detected:
            # Mode quiet + decisions only
            for decision in results.decisions_detected:
                print(decision['sentence'])

        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Extraction interrompue par l'utilisateur")
        return 1
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
