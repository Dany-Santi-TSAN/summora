"""
LLM Qwen Recommendation pour amélioration des meetings
Module avec LLM Juge + SpotChecker + Eval fallback
"""
import json
import os
import time
import logging
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Charge les variables depuis .env
load_dotenv()

logger = logging.getLogger(__name__)

class QwenRecommendator:
    """
    Générateur de recommandations meeting avec Qwen + évaluation qualité intégrée.

    Architecture cohérente avec les autres modules LLM :
    - Génération avec Qwen
    - Évaluation avec Judge DeepSeek
    - API unifiée recommend_and_evaluate()
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le recommandateur Qwen avec juge intégré.

        Args:
            api_key: Clé API OpenRouter (ou depuis .env)
        """
        if api_key is None:
            api_key = os.getenv('OPENROUTER_API_KEY')

        if not api_key:
            raise ValueError("Clé API OpenRouter requise (OPENROUTER_API_KEY en env ou paramètre)")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        # Modèles LLM (pattern identique aux autres modules)
        self.recommender_model = "qwen/qwen3-235b-a22b:free"
        self.judge_model = "tngtech/deepseek-r1t-chimera:free"
        self.fallback_judge = "nousresearch/deephermes-3-llama-3-8b-preview:free"

        logger.info("💡 Recommandeur Qwen+Judge initialisé")

    def generate_meeting_recommendations(self, transcription: str, extraction_data: Dict = None) -> Dict:
        """
        Génère des recommandations d'amélioration meeting ultra simples et académiques.

        Args:
            transcription: Transcription de la réunion
            extraction_data: Données d'extraction (optionnel, pour enrichir contexte)

        Returns:
            Dict: Recommandations structurées
        """
        # Contexte enrichi si extraction disponible
        context_info = ""
        if extraction_data and extraction_data.get('success'):
            extraction = extraction_data.get('extraction', {})
            topics = extraction.get('topics_principaux', [])
            points = extraction.get('points_a_retenir', [])

            context_info = f"""
CONTEXTE EXTRACTION DISPONIBLE :
Topics identifiés : {', '.join(topics[:3])}
Points clés : {len(points)} identifiés
            """

        prompt = f"""Tu es consultant expert en optimisation de meetings d'entreprise.

Analyse cette transcription de réunion et génère des recommandations CONCRÈTES et ACTIONNABLES pour améliorer les futurs meetings.

{context_info}

TRANSCRIPTION À ANALYSER :
{transcription[:3000]}

Génère 5-8 recommandations dans ces catégories :
- **Structure** : Améliorer l'organisation
- **Animation** : Optimiser la conduite
- **Participation** : Favoriser l'engagement
- **Efficacité** : Maximiser les résultats
- **Communication** : Clarifier les échanges

Format JSON uniquement :
{{
    "recommandations_principales": [
        {{
            "categorie": "Structure",
            "titre": "Définir un ordre du jour précis",
            "description": "Préparer et partager l'agenda 24h avant le meeting avec objectifs clairs",
            "impact": "high",
            "facilite_implementation": "easy"
        }},
        {{
            "categorie": "Animation",
            "titre": "Améliorer la gestion du temps",
            "description": "Utiliser un timer et allouer des créneaux précis pour chaque sujet",
            "impact": "medium",
            "facilite_implementation": "medium"
        }}
    ],
    "resume_conseil": "Synthèse des 2-3 améliorations prioritaires pour transformer ce type de réunion",
    "score_amelioration_potentiel": 75
}}"""

        logger.info(f"💡 Génération recommandations - Input: {len(transcription)} chars")
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.recommender_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.2  # Légère créativité pour recommandations variées
            )

            duration = time.time() - start_time
            result = response.choices[0].message.content

            logger.info(f"✅ Recommandations générées en {duration:.2f}s - Output: {len(result)} chars")

            # Parse et validation JSON
            try:
                parsed_result = json.loads(result)
                return {
                    "success": True,
                    "data": parsed_result,
                    "raw_content": result,
                    "metrics": {
                        "duration": duration,
                        "model": self.recommender_model,
                        "input_chars": len(transcription),
                        "output_chars": len(result),
                        "context_enriched": bool(extraction_data)
                    }
                }
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur parsing JSON recommandations: {e}")
                return {
                    "success": False,
                    "error": "json_parsing_failed",
                    "raw_content": result,
                    "metrics": {"duration": duration}
                }

        except Exception as e:
            logger.error(f"❌ Erreur génération recommandations: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"duration": time.time() - start_time}
            }

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
        recommandations = recommendations_data.get("recommandations_principales", [])
        resume_conseil = recommendations_data.get("resume_conseil", "")

        prompt = f"""Tu es expert en évaluation de qualité de recommandations business.

Évalue ces recommandations d'amélioration meeting selon ces critères (score 0-100) :
- **Pertinence** : Les recommandations sont-elles adaptées aux problèmes détectés ?
- **Actionnabilité** : Peut-on facilement les mettre en œuvre ?
- **Impact** : Vont-elles vraiment améliorer les futures réunions ?
- **Spécificité** : Sont-elles concrètes et précises ?

TRANSCRIPTION ORIGINALE (extrait) :
{transcription[:2000]}...

RECOMMANDATIONS À ÉVALUER :
{json.dumps(recommandations[:3], indent=2, ensure_ascii=False)}

Résumé conseil : {resume_conseil}

Réponds en JSON uniquement :
{{
    "pertinence": 85,
    "actionnabilite": 90,
    "impact_potentiel": 80,
    "specificite": 88,
    "score_global": 86,
    "justification": "Bonnes recommandations concrètes et applicables...",
    "qualite_conseil": "high"
}}"""

        logger.info("⚖️ Évaluation recommandations par Judge...")
        start_time = time.time()

        # Tentative avec judge principal
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
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
                    model=self.fallback_judge,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
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

    def recommend_and_evaluate(self, transcription: str, extraction_data: Dict = None) -> Dict:
        """
        Pipeline complet : Génération recommandations + Évaluation qualité.
        API unifiée sur modèle des autres modules LLM.

        Args:
            transcription: Transcription de la réunion
            extraction_data: Données d'extraction (optionnel)

        Returns:
            Dict: Recommandations + évaluation complète
        """
        logger.info(f"💡 Pipeline recommandations+évaluation - Input: {len(transcription)} chars")
        start_time = time.time()

        # 1. Génération recommandations
        recommendation_result = self.generate_meeting_recommendations(transcription, extraction_data)

        if not recommendation_result["success"]:
            return {
                "method": "qwen_recommender_with_judge",
                "success": False,
                "error": "recommendation_generation_failed",
                "recommendation_details": recommendation_result,
                "metrics": {"duration": time.time() - start_time}
            }

        recommendations_data = recommendation_result["data"]

        # 2. Évaluation qualité
        judge_result = self.judge_recommendations_quality(transcription, recommendations_data)

        total_duration = time.time() - start_time

        return {
            "method": "qwen_recommender_with_judge",
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
            "metrics": {
                "total_duration": total_duration,
                "generation_duration": recommendation_result["metrics"]["duration"],
                "judge_duration": judge_result.get("metrics", {}).get("duration", 0),
                "recommender_model": self.recommender_model,
                "judge_model": self.judge_model,
                "context_enriched": recommendation_result["metrics"]["context_enriched"]
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
