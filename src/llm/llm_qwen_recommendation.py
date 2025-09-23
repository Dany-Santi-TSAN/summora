"""
LLM Qwen Recommendation pour amélioration des meetings
Module avec LLM Juge + SpotChecker + Eval fallback
REFACTO : centralisé avec llm_config.py
"""
import json
import time
import logging
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import sys

# Path setup pour imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Imports SUMMORA
from src.config.llm_config import MODELS, PROMPTS, config

# Charge les variables depuis .env
load_dotenv()

logger = logging.getLogger(__name__)

class QwenRecommendator:
    """
    Générateur de recommandations meeting avec Qwen + évaluation qualité intégrée.

    Architecture cohérente avec les autres modules LLM :
    - Génération avec Qwen
    - Évaluation avec Judge DeepSeek R1
    - API unifiée recommend_and_evaluate()
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le recommandateur Qwen avec juge intégré.

        Args:
            api_key: Clé API OpenRouter (ou depuis .env)
        """
        if api_key is None:
            api_key = config.openrouter_api_key

        if not api_key:
            raise ValueError("Clé API OpenRouter requise (OPENROUTER_API_KEY en env ou paramètre)")

        self.client = OpenAI(
            api_key=api_key
            ,base_url=config.base_url
        )

        # Modèles LLM
        self.recommender_model = MODELS['recommendation']
        self.judge_model = MODELS['judge_primary']
        self.fallback_judge = MODELS['judge_fallback']

        logger.info("💡 Recommandeur Qwen+Judge initialisé")

    def generate_meeting_recommendations(self, transcription: str, enhanced_data: Dict = None,
                                  meeting_context: str = "", specialized_context: str = "",
                                  meeting_type: str = "Général") -> Dict:
        """
        Génère des recommandations d'amélioration meeting avec contexte RAG enrichi.

        Utilise les contextes enrichis pour adapter les recommandations :
        - RAG : Documents leadership et best practices
        - Meeting context : Type + topics + actions détectées
        - Specialized context : Few-shot adaptatif selon le type

        Args:
            transcription: Transcription de la réunion
            enhanced_data: Données enrichies RAG + extraction
            meeting_context: Contexte meeting enrichi (TYPE: X | TOPICS: Y | ACTIONS: Z)
            specialized_context: Few-shot spécialisé selon type meeting
            meeting_type: Type de meeting détecté

        Returns:
            Dict: Recommandations structurées avec métriques enrichissement
        """
        logger.info(f"💡 Génération recommandations RAG-Enhanced - Input: {len(transcription)} chars")
        start_time = time.time()

        # Construction du prompt avec tous les contextes
        prompt = PROMPTS['recommendation'](transcription,
                                            enhanced_data,
                                            meeting_context,
                                            specialized_context,
                                            meeting_type
                                            )

        try:
            response = self.client.chat.completions.create(
            model=self.recommender_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens_reco,
            temperature=config.temperature_reco
            )

            duration = time.time() - start_time
            result = response.choices[0].message.content
            logger.info(f"RAW RESPONSE '{result[:200]}'")
            logger.info(f"Len de RAW RESPONSE: '{len(result)}'")

            logger.info(f"✅ Recommandations générées en {duration:.2f}s - Output: {len(result)} chars")

            # Clean parsing json
            cleaned_response = result.strip()

            # Retire les balises markdown
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Retire '```json'
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]   # Retire '```'
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Retire '```' final

            cleaned_response = cleaned_response.strip()

            # DEBUG
            logger.info(f"DEBUG - Après nettoyage: {cleaned_response[:200]}...")


            # Parse et validation JSON
            try:
                parsed_result = json.loads(cleaned_response)
                logger.info(f"✅ JSON parsing réussi")
                logger.info(f"Clés JSON: {list(parsed_result.keys())}")
                if 'recommandations_principales' in parsed_result:
                    logger.info(f"Nb recommandations: {len(parsed_result['recommandations_principales'])}")

                # Enrichissement des métriques avec contextes utilisés
                context_metrics = {
                    "rag_context_used": bool(enhanced_data and 'rag_context' in enhanced_data),
                    "meeting_context_used": bool(meeting_context),
                    "specialized_context_used": bool(specialized_context),
                    "meeting_type_detected": meeting_type
                    }

                return {
                    "success": True,
                    "data": parsed_result,
                    "raw_content": result,
                    "metrics": {
                        "duration": duration,
                        "model": self.recommender_model,
                        "input_chars": len(transcription),
                        "output_chars": len(result),
                        "context_enriched": bool(enhanced_data),
                        "enhancement_level": self._determine_enhancement_level(context_metrics),
                        **context_metrics
                    }
            }
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur parsing JSON recommandations: {e}")
                return {
                    "success": False,
                    "error": "json_parsing_failed",
                    "raw_content": result,
                    "meeting_type_detected": meeting_type,
                    "metrics": {"duration": duration}
           }

        except Exception as e:
            logger.error(f"❌ Erreur génération recommandations: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "meeting_type_detected": meeting_type,
                "metrics": {"duration": time.time() - start_time}
                }

    def _determine_enhancement_level(self, context_metrics: Dict) -> str:
        """Détermine le niveau d'enrichissement utilisé."""
        if context_metrics["rag_context_used"] and context_metrics["meeting_context_used"] and context_metrics["specialized_context_used"]:
            return "rag_meeting_specialized_full"
        elif context_metrics["rag_context_used"] and context_metrics["meeting_context_used"]:
            return "rag_meeting_enhanced"
        elif context_metrics["rag_context_used"]:
            return "rag_basic"
        elif context_metrics["meeting_context_used"]:
            return "meeting_context_only"
        else:
            return "basic"

    def judge_recommendations_quality(self, transcription: str, recommendations_data: Dict) -> Dict:
        """
        Évaluation de la qualité des recommandations par LLM Judge.
        Pattern identique aux autres modules pour cohérence.

        Args:
            transcription: Transcription originale
            recommendations_data: Recommandations générées

        Returns:
            Dict: Scores de qualité des recommandations
        """
        prompt = PROMPTS['judge_recommendation'](transcription, recommendations_data)

        logger.info("⚖️ Évaluation recommandations par Judge...")
        start_time = time.time()

        # Tentative avec judge principal
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model
                ,messages=[{"role": "user", "content": prompt}]
                ,max_tokens=config.judge_max_tokens
                ,temperature=config.temperature
            )

            duration = time.time() - start_time
            result = response.choices[0].message.content.strip()

            logger.info(f"✅ Judge recommandations terminé en {duration:.2f}s")

            if not result.strip():
                logger.error("❌ Judge a retourné une réponse vide")
                return {
                    "success": False,
                    "error": "empty_response",
                    "fallback_score": 50.0,
                    "metrics": {"duration": duration}
                }

            try:
                # Extraction JSON robuste (même logique que corrector)
                result_cleaned = result.strip()
                json_start = -1
                json_end = -1

                # Trouve le premier '{'
                for i, char in enumerate(result_cleaned):
                    if char == '{':
                        json_start = i
                        break

                if json_start >= 0:
                    # Compte les accolades pour trouver la fin du JSON
                    brace_count = 0
                    for i in range(json_start, len(result_cleaned)):
                        char = result_cleaned[i]
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break

                    if json_end > 0:
                        json_content = result_cleaned[json_start:json_end]
                        judge_scores = json.loads(json_content)
                    else:
                        raise json.JSONDecodeError("Fin de JSON non trouvée", result_cleaned, 0)
                else:
                    raise json.JSONDecodeError("Début de JSON non trouvé", result_cleaned, 0)

                score_global = judge_scores.get('score_global', 0)

                logger.info(f"✅ Judge recommandations - Score global: {score_global}")

                return {
                    "success": True,
                    "scores": judge_scores,
                    "score_global": score_global,
                    "qualite_conseil": judge_scores.get('qualite_conseil', 'medium'),
                    "metrics": {
                        "duration": duration,
                        "model": self.judge_model,
                        "fallback_used": False
                    }
                }

            except json.JSONDecodeError as e:
                logger.error(f"❌ Judge JSON parsing failed: {e}")
                logger.error(f"📋 Raw response: {result}")
                return {
                    "success": False,
                    "error": "judge_json_parsing_failed",
                    "raw_content": result,
                    "fallback_score": 50.0,
                    "metrics": {"duration": duration}
                }

        except Exception as e:
            logger.warning(f"Judge principal échoué: {str(e)}, tentative fallback...")

            # Fallback judge (même logique que extractor)
            try:
                response = self.client.chat.completions.create(
                    model=self.fallback_judge
                    ,messages=[{"role": "user", "content": prompt}]
                    ,max_tokens=config.judge_max_tokens
                    ,temperature=config.temperature
                )

                duration = time.time() - start_time
                result = response.choices[0].message.content

                logger.info(f"✅ Fallback Judge recommandations terminé en {duration:.2f}s")

                try:
                    # Extraction JSON simple pour fallback
                    if result.strip().startswith('{'):
                        judge_scores = json.loads(result.strip())
                    else:
                        # Cherche le JSON dans la réponse
                        lines = result.split('\n')
                        json_line = None
                        for line in lines:
                            if line.strip().startswith('{'):
                                json_line = line.strip()
                                break

                        if json_line:
                            judge_scores = json.loads(json_line)
                        else:
                            raise json.JSONDecodeError("JSON non trouvé", result, 0)

                    return {
                        "success": True,
                        "scores": judge_scores,
                        "score_global": judge_scores.get('score_global', 50),
                        "qualite_conseil": judge_scores.get('qualite_conseil', 'medium'),
                        "metrics": {
                            "duration": duration,
                            "model": self.fallback_judge,
                            "fallback_used": True
                        }
                    }
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "fallback_judge_json_parsing_failed",
                        "raw_content": result,
                        "fallback_score": 40.0,
                        "metrics": {"duration": duration}
                    }

            except Exception as e2:
                logger.error(f"❌ Tous les judges ont échoué: {str(e2)}")
                return {
                    "success": False,
                    "error": f"all_judges_failed: {str(e2)}",
                    "fallback_score": 30.0,
                    "metrics": {"duration": time.time() - start_time}
                }

    def recommend_and_evaluate(self, transcription: str, enhanced_data: Dict = None) -> Dict:
        """
        Pipeline complet : Génération recommandations + Évaluation qualité avec contexte RAG enrichi.

        Traite les données enrichies incluant :
        - Contexte RAG (documents leadership)
        - Contexte meeting (type, topics, actions/décisions)
        - Contexte spécialisé (few-shot adaptatif)

        Args:
            transcription: Transcription de la réunion
            enhanced_data: Données enrichies incluant RAG + contexte meeting
                - rag_context: Documents leadership pertinents
                - meeting_context: Type + topics + actions détectées
                - specialized_context: Few-shot adaptatif selon type
                - meeting_type: Type détecté (brainstorming, décisionnelle, etc.)

        Retourne:
            Dict: Recommandations + évaluation avec métriques enrichissement
        """
        logger.info(f"Pipeline recommandations RAG-Enhanced - Input: {len(transcription)} chars")
        start_time = time.time()

        # Extraction des contextes enrichis
        meeting_context = enhanced_data.get('meeting_context', '') if enhanced_data else ''
        meeting_type = enhanced_data.get('meeting_type', 'Général') if enhanced_data else 'Général'
        specialized_context = enhanced_data.get('specialized_context', '') if enhanced_data else ''
        rag_used = 'rag_context' in (enhanced_data or {})

        # 1. Génération recommandations avec contexte enrichi
        recommendation_result = self.generate_meeting_recommendations(
                        transcription,
                        enhanced_data,
                        meeting_context=meeting_context,
                        specialized_context=specialized_context,
                        meeting_type=meeting_type
                        )

        if not recommendation_result["success"]:
            return {
                "method": "rag_enhanced_qwen_recommender",
                "success": False,
                "error": "recommendation_generation_failed",
                "recommendation_details": recommendation_result,
                "meeting_type_detected": meeting_type,
                "rag_used": rag_used,
                "metrics": {"duration": time.time() - start_time}
            }

        recommendations_data = recommendation_result["data"]

        # 2. Évaluation qualité avec contexte
        judge_result = self.judge_recommendations_quality(transcription, recommendations_data)

        total_duration = time.time() - start_time

        return {
            "method": "rag_enhanced_qwen_recommender",
            "success": True,
            "recommendations": {
                "recommandations_principales": recommendations_data.get("recommandations_principales", []),
                "resume_conseil": recommendations_data.get("resume_conseil", ""),
                "score_amelioration_potentiel": recommendations_data.get("score_amelioration_potentiel", 50),
                "nb_recommandations": len(recommendations_data.get("recommandations_principales", []))
            },
            "quality_evaluation": judge_result,
            "ready_for_implementation": recommendation_result["success"] and judge_result.get("success", False),
            "conseil_quality": judge_result.get("qualite_conseil", "medium"),
            "meeting_type_detected": meeting_type,
            "rag_used": rag_used,
            "contexts_used": {
                "rag_context": bool(enhanced_data and 'rag_context' in enhanced_data),
                "meeting_context": bool(meeting_context),
                "specialized_context": bool(specialized_context)
            },
            "metrics": {
                "total_duration": total_duration,
                "generation_duration": recommendation_result["metrics"]["duration"],
                "judge_duration": judge_result.get("metrics", {}).get("duration", 0),
                "recommender_model": self.recommender_model,
                "judge_model": self.judge_model,
                "context_enriched": recommendation_result["metrics"]["context_enriched"],
                "enhancement_level": "rag_meeting_specialized" if rag_used and meeting_context else "basic"
            }
        }

# Factory et fonctions utilitaires (rétrocompatibilité)
def create_qwen_recommender() -> QwenRecommendator:
    """Factory pour créer le recommandateur Qwen."""
    return QwenRecommendator()

def generate_meeting_recommendations_simple(transcription: str) -> Dict:
    """
    Fonction utilitaire pour génération rapide de recommandations.
    Usage basique sans contexte d'extraction.
    """
    recommender = create_qwen_recommender()

    # Génération simple sans évaluation
    result = recommender.generate_meeting_recommendations(transcription)

    if result["success"]:
        return {
            "method": "qwen_llm_recommender_simple",
            "success": True,
            "recommendations": result["data"],
            "metrics": result["metrics"]
        }
    else:
        return {
            "method": "qwen_llm_recommender_simple",
            "success": False,
            "error": result["error"],
            "metrics": result["metrics"]
        }

# Fonction pipeline complet (nouvelle API recommandée)
def recommend_and_evaluate_meeting(transcription: str, extraction_data: Dict = None) -> Dict:
    """
    Pipeline complet recommandations+évaluation pour usage direct.
    Nouvelle API recommandée - compatible avec main_reco.py.

    Args:
        transcription: Transcription de la réunion
        extraction_data: Résultats d'extraction (optionnel, pour enrichir contexte)

    Returns:
        Dict: Recommandations complètes avec évaluation
    """
    recommender = create_qwen_recommender()
    return recommender.recommend_and_evaluate(transcription, extraction_data)

if __name__ == "__main__":
    # Test simple si exécuté directement
    test_transcription = "Réunion test sur la productivité. Beaucoup de discussions mais peu de décisions claires."
    result = generate_meeting_recommendations_simple(test_transcription)
    print(f"Test result: {result.get('success', False)}")
