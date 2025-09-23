"""
Extracteur de contenu spécialisé pour les meetings
insights business : Topics, actions, décisions et type de réunion
Version finalisée avec vocabulaires spécialisés
"""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

import yake

# Imports des vocabulaires spécialisés
from ..core.business_vocabulary import (
    BUSINESS_KEYWORDS
    ,get_keyword_category
    ,is_business_keyword
)

from ..core.utils_stopwords_meeting_fr import get_all_meeting_stopwords_fr

logger = logging.getLogger(__name__)

# === Configuration améliorée ===

@dataclass
class MeetingExtractionConfig:
    """Configuration pour l'extraction de contenu meeting."""
    # YAKE config pour meetings
    yake_language: str = "fr"
    yake_max_ngram: int = 3
    yake_deduplication_threshold: float = 0.7
    yake_top_keywords: int = 15

    # Méthodes actives
    enabled_methods: List[str] = None
    extract_topics: bool = True
    extract_actions: bool = True
    extract_decisions: bool = True
    extract_meeting_type: bool= True
    use_business_vocabulary: bool = True
    use_enhanced_stopwords: bool = True

    def __post_init__(self):
        if self.enabled_methods is None:
            self.enabled_methods = ['yake', 'actions', 'decisions', 'meeting_type']

# === Extracteur Topics amélioré ===

class MeetingTopicExtractor:
    """Extracteur de topics avec YAKE + vocabulaire business."""

    def __init__(self, config: MeetingExtractionConfig):
        self.config = config

        # Stopwords améliorés (base NLTK + spécialisés meetings)
        if config.use_enhanced_stopwords:
            self.stopwords = get_all_meeting_stopwords_fr()
            logger.info(f"✅ Stopwords meetings: {len(self.stopwords)} mots")
        else:
            # Fallback avec stopwords français essentiels
            basic_stopwords = {
                'le', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir',
                'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec',
                'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'donc',
                'alors', 'bien', 'très', 'où', 'du', 'quand', 'mais',
                'sans', 'sous', 'entre', 'après', 'avant', 'pendant',
            }
            self.stopwords = basic_stopwords
            logger.info(f"✅ Stopwords basiques français: {len(self.stopwords)} mots")

    def extract_topics(self, text: str) -> Dict:
        """Extraction topics avec YAKE amélioré."""
        try:
            # Configuration YAKE
            kw_extractor = yake.KeywordExtractor(
                lan=self.config.yake_language
                ,n=self.config.yake_max_ngram
                ,dedupLim=self.config.yake_deduplication_threshold
                ,top=self.config.yake_top_keywords
            )

            # Extraction
            keywords = kw_extractor.extract_keywords(text)

            # Format topics avec enrichissement business
            topics = []
            for result in keywords:
                # YAKE retourne (keyword, score)
                if len(result) == 2:
                    score_val, keyword_val = result

                    # Détection automatique du format
                    if isinstance(score_val, str):
                        keyword = score_val
                        score = keyword_val
                    else:
                        score = score_val
                        keyword = keyword_val
                else:
                    continue

                # Conversion sécurisée du score
                try:
                    numeric_score = float(score)
                    inverted_score = round(1 - numeric_score, 3)
                except (ValueError, TypeError):
                    inverted_score = 0.0

                # Validation keyword
                if not isinstance(keyword, str):
                    continue

                topic_info = {
                    "keyword": keyword
                    ,"score": inverted_score
                    ,"method": "yake_enhanced"
                }

                # Enrichissement avec vocabulaire business
                if self.config.use_business_vocabulary:
                    topic_info["business_category"] = get_keyword_category(keyword)
                    topic_info["is_business"] = is_business_keyword(keyword)

                topics.append(topic_info)

            # Tri par pertinence business + score
            if self.config.use_business_vocabulary:
                topics.sort(key=lambda x: (x["is_business"], x["score"]), reverse=True)

            return {
                "topics": topics
                ,"method": "yake_enhanced"
                ,"total_found": len(topics)
                ,"business_topics": len([t for t in topics if t.get("is_business", False)])
            }

        except Exception as e:
            logger.error(f"❌ Erreur extraction YAKE: {e}")
            return {"topics": [], "error": str(e)}

# === Détecteur Actions amélioré ===

class MeetingActionDetector:
    """Détecteur d'actions avec vocabulaire business."""

    def __init__(self, config: MeetingExtractionConfig):
        self.config = config

        # Mots-clés actions du vocabulaire business
        if config.use_business_vocabulary:
            self.action_keywords = BUSINESS_KEYWORDS['actions']
            logger.info(f"✅ Vocabulaire d'actions : {len(self.action_keywords)} mots-clés")
        else:
            # Fallback patterns basiques
            self.action_keywords = [
                'action', 'à faire', 'tâche', 'réaliser', 'prochaine étape',
                'todo', 'livrer', 'livrable', 'responsable de'
            ]

    def detect_actions(self, text: str) -> Dict:
        """Détection d'actions avec vocabulaire spécialisé."""
        try:
            actions = []
            sentences = text.split('.')

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Ignore phrases trop courtes
                    continue

                # Vérification présence mots-clés actions
                sentence_lower = sentence.lower()
                action_score = 0
                matched_keywords = []

                for keyword in self.action_keywords:
                    if keyword.lower() in sentence_lower:
                        action_score += 1
                        matched_keywords.append(keyword)

                # Si au moins un mot-clé action trouvé
                if action_score > 0:
                    actions.append({
                        "action": sentence
                        ,"confidence": min(action_score / 3, 1.0)
                        ,"keywords_matched": matched_keywords
                        ,"method": "business_vocabulary"
                    })

            # Tri par confiance
            actions.sort(key=lambda x: x["confidence"], reverse=True)

            return {
                "actions": actions[:10]
                ,"method": "business_vocabulary_enhanced"
                ,"total_found": len(actions)
            }

        except Exception as e:
            logger.error(f"❌ Erreur détection actions: {e}")
            return {"actions": [], "error": str(e)}

# === Détecteur Décisions amélioré ===

class MeetingDecisionDetector:
    """Détecteur de décisions avec vocabulaire business."""

    def __init__(self, config: MeetingExtractionConfig):
        self.config = config

        # Mots-clés décisions du vocabulaire business
        if config.use_business_vocabulary:
            self.decision_keywords = BUSINESS_KEYWORDS['decisions']
            logger.info(f"✅ Vocabulaire décisions: {len(self.decision_keywords)} mots-clés")
        else:
            # Fallback patterns basiques
            self.decision_keywords = [
                'décision', 'décider', 'validé', 'approuvé',
            ]

    def detect_decisions(self, text: str) -> Dict:
        """Détection de décisions avec vocabulaire spécialisé."""
        try:
            decisions = []
            sentences = text.split('.')

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Ignore phrases trop courtes
                    continue

                # Vérification présence mots-clés décisions
                sentence_lower = sentence.lower()
                decision_score = 0
                matched_keywords = []

                for keyword in self.decision_keywords:
                    if keyword.lower() in sentence_lower:
                        decision_score += 1
                        matched_keywords.append(keyword)

                # Si au moins un mot-clé décision trouvé
                if decision_score > 0:
                    decisions.append({
                        "decision": sentence
                        ,"confidence": min(decision_score / 2, 1.0)
                        ,"keywords_matched": matched_keywords
                        ,"method": "business_vocabulary"
                    })

            # Tri par confiance
            decisions.sort(key=lambda x: x["confidence"], reverse=True)

            return {
                "decisions": decisions[:10]
                ,"method": "business_vocabulary_enhanced"
                ,"total_found": len(decisions)
            }

        except Exception as e:
            logger.error(f"❌ Erreur détection décisions: {e}")
            return {"decisions": [], "error": str(e)}

# === Détecteur type de meeting ===

class MeetingTypeDetector:
    """Détecteur de type de réunions pour enrichir la recommandation."""

    def __init__(self, config: MeetingExtractionConfig):
        self.config = config

        # Mots-clés actions du vocabulaire business
        if config.use_business_vocabulary:
            self.meeting_type_keywords = BUSINESS_KEYWORDS['meeting_type']
            logger.info(f"✅ Vocabulaire contextuel du type de réunion : {len(self.meeting_type_keywords)} mots-clés")
        else:
            # Fallback patterns basiques
            self.meeting_type_keywords = {
            "brainstorming": ["brainstorm", "idées", "créativité"],
            "copil": ["copil", "pilotage", "stratégie"],
            "rétrospective": ["rétro", "amélioration", "sprint"],
            "client": ["client", "démonstration", "livrable"],
            "conflit": ["conflit", "désaccord", "tension"],
            "décisionnelle": ["décision", "choix", "arbitrage"]
        }

    def detect_meeting_type(self, text: str) -> str:
        """Détecte le type de réunion avec des mots-clés."""

        text_lower = text.lower()

        for meeting_type, keywords in self.meeting_type_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return meeting_type.capitalize()
        return "Général"  # Fallback par défaut

# === Orchestrateur principal ===


class MeetingContentExtractor:
    """
    Orchestrateur principal pour l'extraction d'informations structurées à partir de contenus de réunion.

    Ce composant central coordonne plusieurs modules spécialisés afin d'analyser un meeting transcript
    et d'en extraire les éléments clés utiles à la prise de décision.

    Il délègue les tâches suivantes :
    - TopicExtractor : identification des mots-clés et des thématiques abordées
    - ActionDetector : détection des actions à mener, avec attribution aux responsables
    - DecisionDetector : identification des décisions prises et validations formelles

    Fonctionne comme un pipeline orchestré, à la manière d'un scheduler, chaque module étant responsable
    d'une étape spécifique de l'analyse.
    """

    def __init__(self, config: Optional[MeetingExtractionConfig] = None):
        self.config = config or MeetingExtractionConfig()

        # Initialisation des extracteurs améliorés
        self.topic_extractor = MeetingTopicExtractor(self.config)

        if self.config.extract_actions:
            self.action_detector = MeetingActionDetector(self.config)

        if self.config.extract_decisions:
            self.decision_detector = MeetingDecisionDetector(self.config)

        if self.config.extract_meeting_type:
            self.meeting_type_detector = MeetingTypeDetector(self.config)

        logger.info("🎯 Extracteur meeting amélioré initialisé")
        logger.info(f"   • Méthodes: {self.config.enabled_methods}")
        logger.info(f"   • Vocabulaire business: {'✅' if self.config.use_business_vocabulary else '❌'}")
        logger.info(f"   • Stopwords améliorés: {'✅' if self.config.use_enhanced_stopwords else '❌'}")

    def _get_active_methods(self) -> List[str]:
        """Retourne les méthodes actives selon la configuration."""
        methods = []

        if 'yake' in self.config.enabled_methods or 'topics' in self.config.enabled_methods:
            methods.append('topics')

        if self.config.extract_actions:
            methods.append('actions')

        if self.config.extract_decisions:
            methods.append('decisions')

        if self.config.extract_meeting_type:
            methods.append('meeting_type')

        return methods

    def _has_successful_extractions(self, results: Dict) -> bool:
        """Vérifie si au moins une extraction a réussi."""
        if not results:
            logger.warning("⚠️ Aucun résultat d'extraction fourni")
            return False

        successful = 0
        total_methods = len(results)

        for method, result in results.items():
            if method == 'metadata':  # Skip metadata
                total_methods -= 1
                continue

            if isinstance(result, dict) and 'error' not in result:
                if method == 'topics' and result.get('topics'):
                    successful += 1
                elif method == 'actions' and result.get('actions'):
                    successful += 1
                elif method == 'decisions' and result.get('decisions'):
                    successful += 1
                elif method == 'meeting_type' and result.get('meeting_type'):
                    successful += 1

        success_rate = (successful / total_methods * 100) if total_methods > 0 else 0
        logger.info(f"📊 Bilan extractions: {successful}/{total_methods} réussies ({success_rate:.1f}%)")

        return successful > 0

    def _create_meeting_summary(self, results: Dict) -> Dict:
        """Crée un résumé combiné intelligent."""
        summary = {
            "method": "Combinaison extracteurs spécialisés meetings"
            ,"extractors_used": [k for k in results.keys() if k != 'metadata']
            ,"timestamp": datetime.now().isoformat()
        }

        # Combine les meilleurs topics
        if 'topics' in results and results['topics'].get('topics'):
            top_topics = results['topics']['topics'][:5]
            summary['top_topics'] = [t['keyword'] for t in top_topics]

        # Compte actions et décisions
        if 'actions' in results:
            summary['actions_count'] = len(results['actions'].get('actions', []))

        if 'decisions' in results:
            summary['decisions_count'] = len(results['decisions'].get('decisions', []))

        if 'meeting_type' in results:
            meeting_type_data = results['meeting_type']
        if isinstance(meeting_type_data, dict):
            summary['meeting_type'] = meeting_type_data.get('meeting_type', 'Général')
        else:
            summary['meeting_type'] = meeting_type_data

        return summary

    def extract_meeting_content(self, text: str, methods: Optional[List[str]] = None) -> Dict:
        """Point d'entrée principal pour l'extraction améliorée."""

        if not text.strip():
            return {"error": "Aucun texte de meeting fourni"}

        methods = methods or self.config.enabled_methods
        results = {}

        logger.info(f"🎯 Extraction contenu meeting améliorée")
        logger.info(f"📄 Texte: {len(text)} caractères")
        logger.info(f"🔧 Méthodes: {', '.join(methods)}")

        # 1. Topics (YAKE amélioré)
        if any(method in methods for method in ['yake', 'topics']):
            logger.info("📌 Extraction topics...")
            results['topics'] = self.topic_extractor.extract_topics(text)

        # 2. Actions (vocabulaire business)
        if 'actions' in methods and hasattr(self, 'action_detector'):
            logger.info("📌 Détection actions...")
            results['actions'] = self.action_detector.detect_actions(text)

        # 3. Décisions (vocabulaire business)
        if 'decisions' in methods and hasattr(self, 'decision_detector'):
            logger.info("📌 Détection décisions...")
            results['decisions'] = self.decision_detector.detect_decisions(text)

        # 4. Type de meeting
        if 'meeting_type' in methods and hasattr(self, 'meeting_type_detector'):
            logger.info("📌 Détection type de meeting...")
            results['meeting_type'] = {
                'meeting_type': self.meeting_type_detector.detect_meeting_type(text),
                'method': 'business_vocabulary_enhanced'
                }

        # Synthèse finale
        if self._has_successful_extractions(results):
            logger.info("🏆 Synthèse insights meeting...")
            results['meeting_summary'] = self._create_meeting_summary(results)

        # Métadonnées améliorées
        results['metadata'] = {
            "text_length": len(text)
            ,"methods_used": list(k for k in results.keys() if k != 'metadata')
            ,"config": {
                "business_vocabulary": self.config.use_business_vocabulary
                ,"enhanced_stopwords": self.config.use_enhanced_stopwords
            }
            ,"timestamp": datetime.now().isoformat()
        }

        return results

# === Factory functions améliorées ===

def create_meeting_extractor(extract_topics: bool = True
                           ,extract_actions: bool = True
                           ,extract_decisions: bool = True
                           ,extract_meeting_type: bool = True
                           ,use_business_vocabulary: bool = True
                           ,use_enhanced_stopwords: bool = True) -> MeetingContentExtractor:
    """Factory pour créer un extracteur amélioré."""

    config = MeetingExtractionConfig(
        extract_topics=extract_topics
        ,extract_actions=extract_actions
        ,extract_decisions=extract_decisions
        ,extract_meeting_type=extract_meeting_type
        ,use_business_vocabulary=use_business_vocabulary
        ,use_enhanced_stopwords=use_enhanced_stopwords
    )

    return MeetingContentExtractor(config)

def extract_meeting_content(text: str, **config_kwargs) -> Dict:
    """Helper function pour extraction rapide améliorée."""

    config = MeetingExtractionConfig(**config_kwargs) if config_kwargs else None
    extractor = MeetingContentExtractor(config)
    return extractor.extract_meeting_content(text)
