"""
Extracteur LLM Qwen enrichi par l'intelligence YAKE
Utilise les stopwords et vocabulaire business de YAKE pour guider Qwen
"""
import json
import os
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Path setup pour imports Summora
sys.path.append(str(Path(__file__).parent.parent.parent))

# Imports Summora pour enrichissement YAKE
from src.meeting.extractor import MeetingContentExtractor, MeetingExtractionConfig
from src.core.business_vocabulary import BUSINESS_KEYWORDS

load_dotenv()
logger = logging.getLogger(__name__)

class QwenEnhancedExtractor:
    """
    Extracteur Qwen enrichi par l'intelligence YAKE.

    Pipeline hybride :
    1. YAKE extrait les mots-clés business + contexte
    2. Qwen utilise ces indices pour une extraction plus précise
    """

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv('OPENROUTER_API_KEY')

        if not api_key:
            raise ValueError("Clé API OpenRouter requise")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        # Modèles LLM
        self.extractor_model = "qwen/qwen3-235b-a22b:free"
        self.judge_model = "tngtech/deepseek-r1t-chimera:free"

        # Extracteur YAKE pour preprocessing
        yake_config = MeetingExtractionConfig(
            use_business_vocabulary=True,
            use_enhanced_stopwords=True,
            extract_actions=True,
            extract_decisions=True
        )
        self.yake_extractor = MeetingContentExtractor(yake_config)

        logger.info("🧠 Extracteur Qwen+YAKE initialisé")

    def _extract_yake_context(self, transcription: str) -> Dict:
        """
        Extrait le contexte business via YAKE pour guider Qwen.
        """
        logger.info("📊 Extraction contexte YAKE...")

        try:
            yake_results = self.yake_extractor.extract_meeting_content(transcription)

            # Extraction des éléments pertinents
            context = {
                "business_topics": [],
                "action_indicators": [],
                "decision_indicators": [],
                "business_density": 0
            }

            # Topics business détectés par YAKE
            if 'topics' in yake_results and yake_results['topics'].get('topics'):
                business_topics = [
                    topic['keyword'] for topic in yake_results['topics']['topics']
                    if topic.get('is_business', False)
                ]
                context["business_topics"] = business_topics[:10]  # Top 10

            # Actions détectées
            if 'actions' in yake_results and yake_results['actions'].get('actions'):
                context["action_indicators"] = [
                    action['action'][:100] for action in yake_results['actions']['actions'][:3]
                ]

            # Décisions détectées
            if 'decisions' in yake_results and yake_results['decisions'].get('decisions'):
                context["decision_indicators"] = [
                    decision['decision'][:100] for decision in yake_results['decisions']['decisions'][:3]
                ]

            # Densité business
            if 'topics' in yake_results:
                total_topics = len(yake_results['topics'].get('topics', []))
                business_topics_count = yake_results['topics'].get('business_topics', 0)
                context["business_density"] = (business_topics_count / total_topics * 100) if total_topics > 0 else 0

            logger.info(f"✅ Contexte YAKE: {len(context['business_topics'])} topics business")
            return context

        except Exception as e:
            logger.warning(f"⚠️ Erreur extraction YAKE: {e}")
            return {"business_topics": [], "action_indicators": [], "decision_indicators": [], "business_density": 0}

    def _build_enhanced_prompt(self, transcription: str, yake_context: Dict) -> str:

        business_categories = {
            'actions': BUSINESS_KEYWORDS.get('actions', [])[:8],
            'decisions': BUSINESS_KEYWORDS.get('decisions', [])[:8],
            'planning': BUSINESS_KEYWORDS.get('planning', [])[:5]
        }

        return f"""Tu es expert en analyse de réunions d'entreprise.

CONTEXTE DETECTÉ :
- Topics business : {', '.join(yake_context['business_topics'][:5])}
- Actions identifiées : {len(yake_context['action_indicators'])}

Analyse cette transcription :
{transcription}

Réponds EXACTEMENT dans ce format :

TOPICS PRINCIPAUX:
1. [topic business 1]
2. [topic business 2]
3. [topic business 3]
4. [topic business 4]
5. [topic business 5]

POINTS CLÉS:
- [point actionnable 1]
- [point actionnable 2]
- [point actionnable 3]
- [point actionnable 4]
- [point actionnable 5]
- [point actionnable 6]
- [point actionnable 7]
- [point actionnable 8]
- [point actionnable 9]
- [point actionnable 10]

RÉSUMÉ:
[Synthèse en 2-3 phrases de cette réunion]"""

    def extract_meeting_insights_enhanced(self, transcription: str) -> Dict:
        """
        Extraction enrichie par l'intelligence YAKE + Qwen.
        """
        logger.info(f"🧠 Extraction Qwen+YAKE - Input: {len(transcription)} chars")
        start_time = time.time()

        # 1. Preprocessing YAKE
        yake_context = self._extract_yake_context(transcription)

        # 2. Prompt enrichi
        enhanced_prompt = self._build_enhanced_prompt(transcription, yake_context)

        # 3. Extraction LLM
        try:
            response = self.client.chat.completions.create(
                model=self.extractor_model,
                messages=[{"role": "user", "content": enhanced_prompt}],
                max_tokens=50000,
                temperature=0.1
            )

            duration = time.time() - start_time
            result = response.choices[0].message.content

            logger.info(f"✅ Extraction Qwen+YAKE terminée en {duration:.2f}s")

            # Parse et validation
            try:
                parsed_result = json.loads(result)
                return {
                    "success": True,
                    "data": parsed_result,
                    "yake_context": yake_context,
                    "enhancement_used": True,
                    "metrics": {
                        "duration": duration,
                        "model": self.extractor_model,
                        "yake_topics_used": len(yake_context['business_topics']),
                        "business_density": yake_context['business_density']
                    }
                }
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur parsing JSON: {e}")
                return {
                    "success": False,
                    "error": "json_parsing_failed",
                    "raw_content": result,
                    "yake_context": yake_context
                }

        except Exception as e:
            logger.error(f"❌ Erreur extraction Qwen+YAKE: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "yake_context": yake_context,
                "metrics": {"duration": time.time() - start_time}
            }

    def judge_enhanced_extraction(self, transcription: str, extraction_data: Dict, yake_context: Dict) -> Dict:
        """
        Évaluation enrichie tenant compte du contexte YAKE.
        """
        extraction_insights = extraction_data.get("insights_business", {})

        prompt = f"""Tu es expert en évaluation d'extraction de réunions business.

CONTEXTE YAKE ORIGINAL :
- Topics business : {yake_context['business_topics'][:3]}
- Densité business : {yake_context['business_density']:.1f}%

EXTRACTION À ÉVALUER :
Topics: {extraction_data.get('topics_principaux', [])}
Points clés: {extraction_data.get('points_a_retenir', [])[:5]}
Actions clés: {extraction_insights.get('actions_cles', [])}
Décisions: {extraction_insights.get('decisions_prises', [])}

TRANSCRIPTION (extrait) :
{transcription[:2000]}...

Évalue selon ces critères (score 0-100) :
- **Pertinence business** : Cohérence avec contexte YAKE
- **Complétude** : Couverture des éléments importants
- **Actionnabilité** : Qualité des actions/décisions extraites

JSON uniquement :
{{
    "pertinence_business": 85,
    "completude": 90,
    "actionnabilite": 88,
    "score_global": 88,
    "coherence_yake": 92,
    "justification": "Excellente extraction enrichie qui exploite bien le contexte YAKE..."
}}"""

        logger.info("⚖️ Évaluation enrichie par Judge...")
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )

            duration = time.time() - start_time
            result = response.choices[0].message.content

            logger.info(f"📊 Judge response ({len(result)} chars): {result[:100]}...")

            if not result.strip():
                logger.error("❌ Judge a retourné une réponse vide")
                return {
                    "success": False,
                    "error": "empty_response",
                    "metrics": {"duration": duration}
                }

            try:
                # Nettoyage de la réponse JSON (au cas où il y aurait du texte après)
                result_cleaned = result.strip()

                # Tentative d'extraction du premier bloc JSON valide
                if result_cleaned.startswith('{'):
                    # Trouve la fin du premier objet JSON
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(result_cleaned):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break

                    if json_end > 0:
                        result_cleaned = result_cleaned[:json_end]

                judge_scores = json.loads(result_cleaned)
                logger.info(f"✅ Judge enrichi terminé en {duration:.2f}s - Score global: {judge_scores.get('score_global', 'N/A')}")
                return {
                    "success": True,
                    "scores": judge_scores,
                    "enhancement_evaluated": True,
                    "metrics": {"duration": duration, "model": self.judge_model}
                }
            except json.JSONDecodeError as e:
                logger.error(f"❌ Judge JSON parsing failed: {e}")
                logger.error(f"📋 Raw response: {result}")
                return {
                    "success": False,
                    "error": "json_parsing_failed",
                    "raw_content": result,
                    "metrics": {"duration": duration}
                }

        except Exception as e:
            logger.error(f"❌ Erreur judge enrichi: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"duration": time.time() - start_time}
            }

# Fonction utilitaire
def extract_with_qwen_enhanced(transcription: str) -> Dict:
    """
    Extraction enrichie YAKE+Qwen pour usage direct.
    """
    extractor = QwenEnhancedExtractor()

    # Extraction enrichie
    extraction_result = extractor.extract_meeting_insights_enhanced(transcription)

    if not extraction_result["success"]:
        return {
            "method": "qwen_enhanced_by_yake",
            "success": False,
            "error": extraction_result["error"],
            "yake_context": extraction_result.get("yake_context", {})
        }

    # Évaluation enrichie
    judge_result = extractor.judge_enhanced_extraction(
        transcription,
        extraction_result["data"],
        extraction_result["yake_context"]
    )

    return {
        "method": "qwen_enhanced_by_yake",
        "success": True,
        "extraction": extraction_result["data"],
        "yake_context": extraction_result["yake_context"],
        "quality_scores": judge_result.get("scores", {}),
        "metrics": extraction_result["metrics"],
        "enhancement_used": True
    }

# Test
if __name__ == "__main__":
    test_text = """
    Réunion budget 2025. Présents : Marie, Jean, Paul.
    Décision d'augmenter le budget marketing de 20%.
    Action pour Jean : préparer le plan détaillé d'ici vendredi.
    Objectif : lancement campagne en mars 2025.
    Paul s'occupe de la validation des KPIs.
    """

    result = extract_with_qwen_enhanced(test_text)
    print("🧪 Test Qwen Enhanced:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
