"""
Extracteur LLM Qwen pour analyse de réunions
Module spécialisé pour l'extraction d'insights business
REFACTO : utilisation de la config centralisée llm_config.py
"""
import json
import os
import re
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


# Charge les variables d'environnement depuis .env
load_dotenv()

logger = logging.getLogger(__name__)

class FallBackExtractor:
    """Extracteur d'insights business avec un LLM Fallback gratuit de OpenRouter."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise l'extracteur Fallback gratuit.

        Args:
            api_key: Clé API OpenRouter (ou depuis .env)
        """
        if api_key is None:
            api_key = config.openrouter_api_key

        if not api_key:
            raise ValueError("Clé API OpenRouter requise (OPENROUTER_API_KEY en env)")

        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url
        )

        # Modèles pour extraction
        self.extractor_model = MODELS['extraction_fallback']
        self.extractor_model_name = self._extract_model_short_name(self.extractor_model)
        self.judge_model = MODELS['judge_primary']
        self.fallback_judge = MODELS['judge_fallback']

    def _extract_model_short_name(self, full_model_name: str) -> str:
        """Extrait un nom court et lisible depuis le nom de modèle OpenRouter."""
        # Récupère la partie après le /
        model_core = full_model_name.split("/")[-1]
        # Supprime le suffixe après : (ex. :free, :pro)
        model_core = model_core.split(":")[0]
        # Optionnel : ne garde que les éléments alpha-numériques et les points/traits
        match = re.search(r"(llama-[\d\.]+-[\w\d]+)", model_core)
        return match.group(1) if match else model_core

    def extract_meeting_insights(self, transcription: str) -> Dict:
        """
        Extraction business intelligence avec prompt validé.
        """
        prompt = PROMPTS['extraction_hybrid'](transcription)

        logger.info(f"🧠 Extraction Fallback - Input: {len(transcription)} chars")
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.extractor_model
                ,messages=[{"role": "user", "content": prompt}]
                ,max_tokens=config.max_tokens
                ,temperature=config.temperature
            )

            duration = time.time() - start_time
            result = response.choices[0].message.content

            logger.info(f"✅ Extraction LLM {self.extractor_model_name} terminée en {duration:.2f}s - Output: {len(result)} chars")

            # Parse et validation JSON
            try:
                parsed_result = json.loads(result)
                return {
                    "success": True,
                    "data": parsed_result,
                    "raw_content": result,
                    "metrics": {
                        "duration": duration,
                        "model": self.extractor_model,
                        "input_chars": len(transcription),
                        "output_chars": len(result)
                    }
                }
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur parsing JSON: {e}")
                return {
                    "success": False,
                    "error": "json_parsing_failed",
                    "raw_content": result,
                    "metrics": {"duration": duration}
                }

        except Exception as e:
            logger.error(f"❌ Erreur extraction Qwen: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"duration": time.time() - start_time}
            }

    def judge_extraction_quality(self, transcription: str, extraction_data: Dict) -> Dict:
        """
        Évaluation de la qualité de l'extraction par LLM Judge.
        """

        prompt = PROMPTS['judge_extraction'](transcription, extraction_data)

        logger.info("⚖️ Évaluation par Judge...")
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
            result = response.choices[0].message.content

            logger.info(f"✅ Judge terminé en {duration:.2f}s")

            try:
                judge_scores = json.loads(result)
                return {
                    "success": True,
                    "scores": judge_scores,
                    "metrics": {
                        "duration": duration,
                        "model": self.judge_model,
                        "fallback_used": False
                    }
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "judge_json_parsing_failed",
                    "raw_content": result
                }

        except Exception as e:
            logger.warning(f"Judge principal échoué: {str(e)}, tentative fallback...")

            # Fallback judge
            try:
                response = self.client.chat.completions.create(
                    model=self.fallback_judge,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.judge_max_tokens,
                    temperature=config.temperature
                )

                duration = time.time() - start_time
                result = response.choices[0].message.content

                logger.info(f"✅ Fallback Judge terminé en {duration:.2f}s")

                try:
                    judge_scores = json.loads(result)
                    return {
                        "success": True,
                        "scores": judge_scores,
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
                        "raw_content": result
                    }

            except Exception as e2:
                logger.error(f"❌ Tous les judges ont échoué: {str(e2)}")
                return {
                    "success": False,
                    "error": f"all_judges_failed: {str(e2)}",
                    "metrics": {"duration": time.time() - start_time}
                }

# Factory et fonctions utilitaires
def create_fallback_extractor() -> FallBackExtractor:
    """Factory pour créer LLM Fallback gratuit de OpenRouter."""
    return FallBackExtractor()

def extract_with_fallback_llm(transcription: str) -> Dict:
    """
    Fonction utilitaire pour extraction rapide avec un fallback LLM.
    Compatible avec l'architecture Summora existante.
    """
    extractor = create_fallback_extractor()

    # 1. Extraction
    extraction_result = extractor.extract_meeting_insights(transcription)

    if not extraction_result["success"]:
        return {
            "method": "llm_fallback_extractor",
            "success": False,
            "error": extraction_result["error"],
            "metrics": extraction_result["metrics"]
        }

    # 2. Évaluation qualité
    judge_result = extractor.judge_extraction_quality(
        transcription,
        extraction_result["data"]
    )

    return {
        "method": "llm_fallback_extractor",
        "success": True,
        "extraction": extraction_result["data"],
        "quality_scores": judge_result.get("scores", {}),
        "metrics": {
            "extraction": extraction_result["metrics"],
            "judge": judge_result.get("metrics", {}),
            "total_duration": (
                extraction_result["metrics"]["duration"] +
                judge_result.get("metrics", {}).get("duration", 0)
            )
        }
    }

# Test simple
if __name__ == "__main__":
    pass
