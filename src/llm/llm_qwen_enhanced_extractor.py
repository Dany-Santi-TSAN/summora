"""
Extracteur LLM Qwen enrichi par l'intelligence YAKE
Utilise les stopwords et vocabulaire business de YAKE pour guider Qwen
REFACTO : centralisation avec llm_config.py
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
from src.config.llm_config import MODELS, PROMPTS, config

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
            api_key = config.openrouter_api_key

        if not api_key:
            raise ValueError("Clé API OpenRouter requise")

        self.client = OpenAI(
            api_key=api_key
            ,base_url=config.base_url
        )

        # Modèles LLM
        self.extractor_model = MODELS['extraction_enhanced']
        self.judge_model = MODELS['judge_primary']

        # Extracteur YAKE pour preprocessing
        yake_config = MeetingExtractionConfig(
            use_business_vocabulary=True
            ,use_enhanced_stopwords=True
            ,extract_actions=True
            ,extract_decisions=True
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
                "business_topics": []
                ,"action_indicators": []
                ,"decision_indicators": []
                ,"business_density": 0
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

    def extract_meeting_insights_enhanced(self, transcription: str) -> Dict:
        """
        Extraction enrichie par l'intelligence YAKE + Qwen.
        """
        logger.info(f"🧠 Extraction Qwen+YAKE - Input: {len(transcription)} chars")
        start_time = time.time()

        # 1. Preprocessing YAKE
        yake_context = self._extract_yake_context(transcription)

        # 2. Prompt enrichi
        enhanced_prompt = PROMPTS['extraction_hybrid'](transcription, yake_context)

        # 3. Extraction LLM
        try:
            response = self.client.chat.completions.create(
                model=self.extractor_model
                ,messages=[{"role": "user", "content": enhanced_prompt}]
                ,max_tokens=config.max_tokens
                ,temperature=config.temperature
            )

            duration = time.time() - start_time
            result = response.choices[0].message.content
            logger.info(f"RAW RESPONSE: '{result[:200]}'")

            logger.info(f"✅ Extraction Qwen+YAKE terminée en {duration:.2f}s")

            # Parse et validation
            try:
                parsed_result = json.loads(result)
                return {
                    "success": True
                    ,"data": parsed_result
                    ,"yake_context": yake_context
                    ,"enhancement_used": True
                    ,"metrics": {
                        "duration": duration
                        ,"model": self.extractor_model
                        ,"yake_topics_used": len(yake_context['business_topics'])
                        ,"business_density": yake_context['business_density']
                    }
                }
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur parsing JSON: {e}")
                return {
                    "success": False
                    ,"error": "json_parsing_failed"
                    ,"raw_content": result
                    ,"yake_context": yake_context
                }

        except Exception as e:
            logger.error(f"❌ Erreur extraction Qwen+YAKE: {str(e)}")
            return {
                "success": False
                ,"error": str(e)
                ,"yake_context": yake_context
                ,"metrics": {"duration": time.time() - start_time}
            }

    def judge_enhanced_extraction(self, transcription: str, extraction_data: Dict, yake_context: Dict) -> Dict:
        """
        Évaluation enrichie tenant compte du contexte YAKE.
        """
        extraction_insights = extraction_data.get("insights_business", {})

        prompt = prompt = f"""Tu es un expert en évaluation d'extraction de réunions.
**INSTRUCTIONS STRICTES :**
- Réponds **UNIQUEMENT** avec un JSON valide, sans balises Markdown, sans commentaires, sans introduction.
- **Personnalise les scores et justifications** en fonction du contenu réel de la transcription et de l'extraction.
- Ne répète **JAMAIS** les mêmes scores ou justifications pour des extractions différentes.

**EXEMPLES DE RÉPONSES (few-shot) :**
1. **Exemple 1 (Extraction de haute qualité) :**
   {{
       "pertinence_business": 95,
       "completude": 98,
       "actionnabilite": 90,
       "score_global": 94,
       "coherence_yake": 95,
       "justification": "L'extraction couvre tous les points clés de la réunion, y compris les décisions actionnables et les détails contextuels. Les topics YAKE sont parfaitement alignés avec le contenu."
   }}

2. **Exemple 2 (Extraction moyenne) :**
   {{
       "pertinence_business": 70,
       "completude": 65,
       "actionnabilite": 75,
       "score_global": 70,
       "coherence_yake": 80,
       "justification": "L'extraction manque certains détails importants sur les prochaines étapes, mais les topics principaux sont correctement identifiés. La cohérence avec YAKE est bonne, mais incomplète."
   }}

3. **Exemple 3 (Extraction de faible qualité) :**
   {{
       "pertinence_business": 50,
       "completude": 40,
       "actionnabilite": 30,
       "score_global": 40,
       "coherence_yake": 60,
       "justification": "L'extraction est superficielle et ne reflète pas les discussions clés. Les points actionnables sont absents et les topics YAKE ne sont pas bien exploités."
   }}

**CRITÈRES D'ÉVALUATION (score 0-100) :**
- **Pertinence business** : Les topics reflètent-ils les enjeux business de la réunion ?
- **Complétude** : Tous les points importants sont-ils couverts ?
- **Actionnabilité** : Les points extraits sont-ils exploitables pour des actions concrètes ?
- **Cohérence YAKE** : Les topics YAKE sont-ils bien intégrés et pertinents ?
- **Score global** : Moyenne pondérée des critères ci-dessus.

**CONTEXTE YAKE ORIGINAL :**
- Topics business : {yake_context.get('business_topics', [])[:3]}
- Densité business : {yake_context.get('business_density', 0):.1f}%
- Actions identifiées : {len(yake_context.get('action_indicators', []))}
- Décisions identifiées : {len(yake_context.get('decision_indicators', []))}

**DONNÉES À ÉVALUER :**
- TRANSCRIPTION ORIGINALE (extrait) :
{transcription[:2000]}...

- EXTRACTION À ÉVALUER :
Topics: {extraction_data.get('topics_principaux', [])}
Points clés: {extraction_data.get('points_a_retenir', [])[:5]}...
Résumé: {extraction_data.get('resume_abstractif', '')[:200]}...

- INSIGHTS BUSINESS :
Actions clés: {extraction_insights.get('actions_cles', [])}
Décisions prises: {extraction_insights.get('decisions_prises', [])}

**FORMAT DE RÉPONSE OBLIGATOIRE :**
{{
    "pertinence_business": [score],
    "completude": [score],
    "actionnabilite": [score],
    "score_global": [score],
    "coherence_yake": [score],
    "justification": "[Justification DÉTAILLÉE analysant la cohérence entre YAKE, extraction et insights business]"
}}

**RÈGLE ABSOLUE :**
- Commence ta réponse **immédiatement** par le JSON, sans espace ni saut de ligne avant.
- **Adapte les scores et justifications** en fonction du contenu réel. Ne copie jamais les exemples.
"""


        logger.info("⚖️ Évaluation enrichie par Judge...")
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model
                ,messages=[{"role": "user", "content": prompt}]
                ,max_tokens=config.judge_max_tokens
                ,temperature=config.temperature
            )

            duration = time.time() - start_time
            result = response.choices[0].message.content
            logger.info(f"JUDGE RAW RESPONSE: '{result[:200]}'")

            logger.info(f"📊 Judge response ({len(result)} chars): {result[:100]}...")

            if not result.strip():
                logger.error("❌ Judge a retourné une réponse vide")
                return {
                    "success": False
                    ,"error": "empty_response"
                    ,"metrics": {"duration": duration}
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
                    "success": True
                    ,"scores": judge_scores
                    ,"enhancement_evaluated": True
                    ,"metrics": {"duration": duration, "model": self.judge_model}
                }
            except json.JSONDecodeError as e:
                logger.error(f"❌ Judge JSON parsing failed: {e}")
                logger.error(f"📋 Raw response: {result}")
                return {
                    "success": False
                    ,"error": "json_parsing_failed"
                    ,"raw_content": result
                    ,"metrics": {"duration": duration}
                }

        except Exception as e:
            logger.error(f"❌ Erreur judge enrichi: {str(e)}")
            return {
                "success": False
                ,"error": str(e)
                ,"metrics": {"duration": time.time() - start_time}
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
            "method": "qwen_enhanced_by_yake"
            ,"success": False
            ,"error": extraction_result["error"]
            ,"yake_context": extraction_result.get("yake_context", {})
        }

    # Évaluation enrichie
    judge_result = extractor.judge_enhanced_extraction(
        transcription
        ,extraction_result["data"]
        ,extraction_result["yake_context"]
    )

    return {
        "method": "qwen_enhanced_by_yake"
        ,"success": True
        ,"extraction": extraction_result["data"]
        ,"yake_context": extraction_result["yake_context"]
        ,"quality_scores": judge_result.get("scores", {})
        ,"metrics": extraction_result["metrics"]
        ,"enhancement_used": True
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
