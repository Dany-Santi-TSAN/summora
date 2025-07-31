"""
Correcteur LLM Qwen pour transcriptions de réunions avec évaluation intégrée
Module activé uniquement sur demande (téléchargement transcription propre)
Refacto sur modèle llm_qwen_enhanced_extractor.py pour cohérence
"""
import json
import os
import time
import logging
from typing import Dict, Optional, List
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

# Charge les variables depuis .env
load_dotenv()

logger = logging.getLogger(__name__)

class QwenCorrector:
    """
    Correcteur de transcription avec Qwen + évaluation qualité intégrée.

    Même architecture que QwenEnhancedExtractor pour cohérence :
    - Correction avec Qwen
    - Évaluation avec Judge DeepSeek
    - API unifiée correct_and_evaluate()
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le correcteur Qwen avec juge intégré.

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

        # Modèles LLM (même pattern que enhanced_extractor)
        self.corrector_model = "qwen/qwen3-235b-a22b:free"
        self.judge_model = "tngtech/deepseek-r1t-chimera:free"

        # Gestion des tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_per_chunk = 45000

        logger.info("🔧 Correcteur Qwen+Judge initialisé")

    def count_tokens(self, text: str) -> int:
        """Compte les tokens d'un texte."""
        return len(self.encoding.encode(text))

    def chunk_transcription(self, transcription: str) -> List[str]:
        """
        Découpe intelligente de la transcription en chunks si nécessaire.
        Préserve la cohérence sémantique.
        """
        token_count = self.count_tokens(transcription)

        if token_count <= self.max_tokens_per_chunk:
            return [transcription]

        logger.info(f"📝 Transcription longue ({token_count} tokens), découpage en chunks...")

        # Découpage par paragraphes d'abord
        paragraphs = transcription.split('\n\n')
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

            if self.count_tokens(test_chunk) > self.max_tokens_per_chunk:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Si un paragraphe seul est trop long, découpe par phrases
                if self.count_tokens(paragraph) > self.max_tokens_per_chunk:
                    sentences = paragraph.split('. ')
                    sentence_chunk = ""

                    for sentence in sentences:
                        test_sentence = sentence_chunk + sentence + ". "
                        if self.count_tokens(test_sentence) > self.max_tokens_per_chunk:
                            if sentence_chunk:
                                chunks.append(sentence_chunk.strip())
                            sentence_chunk = sentence + ". "
                        else:
                            sentence_chunk = test_sentence

                    if sentence_chunk:
                        current_chunk = sentence_chunk.strip()
                    else:
                        current_chunk = ""
                else:
                    current_chunk = paragraph
            else:
                current_chunk = test_chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"📋 Découpé en {len(chunks)} chunks")
        return chunks

    def correct_transcription_chunk(self, chunk: str, chunk_index: int = 0) -> Dict:
        """
        Corrige un chunk de transcription avec prompt Ground Truth optimisé.
        """
        prompt = f"""Corrige COMPLÈTEMENT cette transcription. IMPORTANT: traite TOUT le texte fourni.

CONSIGNES SPÉCIALES :
- Correction pure sans reformulation
- ATTENTION : Préserve les noms propres, prénoms, villes, marques (ne les modifie pas)
- Structure en paragraphes lisibles
- Largeur ~80 caractères par ligne

{chunk}

Retourne le texte corrigé et formaté."""

        logger.info(f"✏️ Correction chunk {chunk_index + 1} - {len(chunk)} chars")
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.corrector_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50000,
                temperature=0.0  # Déterminisme strict Ground Truth
            )

            duration = time.time() - start_time
            corrected_text = response.choices[0].message.content

            logger.info(f"✅ Correction chunk {chunk_index + 1} terminée en {duration:.2f}s")

            return {
                "success": True,
                "corrected_text": corrected_text,
                "metrics": {
                    "duration": duration,
                    "input_chars": len(chunk),
                    "output_chars": len(corrected_text),
                    "model": self.corrector_model,
                    "chunk_index": chunk_index
                }
            }

        except Exception as e:
            logger.error(f"❌ Erreur correction chunk {chunk_index + 1}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_text": chunk,  # Fallback vers texte original
                "metrics": {
                    "duration": time.time() - start_time,
                    "chunk_index": chunk_index
                }
            }

    def correct_full_transcription(self, transcription: str) -> Dict:
        """
        Corrige une transcription complète, avec gestion des chunks si nécessaire.
        """
        logger.info(f"📝 Début correction transcription - {len(transcription)} chars")
        start_time = time.time()

        # 1. Découpage si nécessaire
        chunks = self.chunk_transcription(transcription)

        # 2. Correction de chaque chunk
        corrected_chunks = []
        total_errors = []
        chunk_metrics = []

        for i, chunk in enumerate(chunks):
            result = self.correct_transcription_chunk(chunk, i)

            if result["success"]:
                corrected_chunks.append(result["corrected_text"])
            else:
                # Fallback vers texte original en cas d'erreur
                corrected_chunks.append(result["original_text"])
                total_errors.append(f"Chunk {i+1}: {result['error']}")

            chunk_metrics.append(result["metrics"])

        # 3. Assemblage final
        corrected_transcription = "\n\n".join(corrected_chunks)
        total_duration = time.time() - start_time

        logger.info(f"✅ Correction complète terminée en {total_duration:.2f}s")

        if total_errors:
            logger.warning(f"⚠️ {len(total_errors)} erreurs lors de la correction")

        return {
            "success": len(total_errors) == 0,
            "corrected_transcription": corrected_transcription,
            "original_transcription": transcription,
            "errors": total_errors if total_errors else None,
            "metrics": {
                "total_duration": total_duration,
                "chunks_processed": len(chunks),
                "chunks_success": len(chunks) - len(total_errors),
                "original_chars": len(transcription),
                "corrected_chars": len(corrected_transcription),
                "improvement_ratio": len(corrected_transcription) / len(transcription) if len(transcription) > 0 else 1.0,
                "chunk_details": chunk_metrics
            }
        }

    def judge_correction_quality(self, original: str, corrected: str) -> Dict:
        """
        Évaluation qualitative par Juge DeepSeek R1T - Version correction.
        Même pattern que llm_qwen_enhanced_extractor.py pour cohérence.

        Args:
            original: Transcription brute Whisper
            corrected: Transcription corrigée Qwen

        Returns:
            Dict: Scores + justification + insights correction
        """
        prompt = f"""Tu es expert en évaluation de correction de transcriptions multilingues.

Tu dois évaluer la QUALITÉ DE CORRECTION de cette transcription selon critères professionnels.

TRANSCRIPTION ORIGINALE (Whisper Medium):
{original[:2000]}

TRANSCRIPTION CORRIGÉE (LLM Ground Truth):
{corrected[:2000]}

Évalue la correction selon ces critères (score 0-100) :
- **Fidélité** : Préservation du contenu sans reformulation excessive
- **Grammaire** : Correction orthographique et grammaticale
- **Ponctuation** : Amélioration structure et lisibilité
- **Préservation** : Maintien du sens et termes techniques

IMPORTANT: Retourne UNIQUEMENT le JSON, pas d'explication avant.

{{
    "fidelite_contenu": 85,
    "correction_grammaire": 90,
    "amelioration_ponctuation": 88,
    "preservation_sens": 92,
    "score_global": 88,
    "justification": "Excellente correction qui préserve le sens tout en améliorant la lisibilité...",
    "note_sur_10": 8.8,
    "qualite_ground_truth": "high"
}}"""

        logger.info("⚖️ Évaluation correction par Judge...")
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.0  # Déterminisme strict
            )

            duration = time.time() - start_time
            result = response.choices[0].message.content.strip()

            logger.info(f"📊 Judge response ({len(result)} chars): {result[:100]}...")

            if not result.strip():
                logger.error("❌ Judge a retourné une réponse vide")
                return {
                    "success": False,
                    "error": "empty_response",
                    "fallback_score": 50.0,
                    "metrics": {"duration": duration}
                }

            try:
                # Extraction JSON robuste (le juge retourne texte + JSON)
                result_cleaned = result.strip()

                # Cherche le bloc JSON dans la réponse
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
                        logger.debug(f"📋 JSON extrait: {json_content[:100]}...")
                        judge_scores = json.loads(json_content)
                    else:
                        raise json.JSONDecodeError("Fin de JSON non trouvée", result_cleaned, 0)
                else:
                    raise json.JSONDecodeError("Début de JSON non trouvé", result_cleaned, 0)
                score_global = judge_scores.get('score_global', 0)
                note_sur_10 = judge_scores.get('note_sur_10', score_global / 10)

                logger.info(f"✅ Judge correction terminé en {duration:.2f}s - Score global: {score_global}")

                return {
                    "success": True,
                    "scores": judge_scores,
                    "score_global": score_global,
                    "note_sur_10": note_sur_10,
                    "qualite_ground_truth": judge_scores.get('qualite_ground_truth', 'medium'),
                    "metrics": {"duration": duration, "model": self.judge_model}
                }

            except json.JSONDecodeError as e:
                logger.error(f"❌ Judge JSON parsing failed: {e}")
                logger.error(f"📋 Raw response: {result}")
                return {
                    "success": False,
                    "error": "json_parsing_failed",
                    "raw_content": result,
                    "fallback_score": 50.0,
                    "metrics": {"duration": duration}
                }

        except Exception as e:
            logger.error(f"❌ Erreur judge correction: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fallback_score": 0.0,
                "metrics": {"duration": time.time() - start_time}
            }

    def correct_and_evaluate(self, transcription: str) -> Dict:
        """
        Pipeline complet : Correction + Évaluation qualité.
        API unifiée sur modèle enhanced_extractor.

        Args:
            transcription: Transcription brute à corriger

        Returns:
            Dict: Correction + évaluation complète
        """
        logger.info(f"🔧 Pipeline correction+évaluation - Input: {len(transcription)} chars")
        start_time = time.time()

        # 1. Correction
        correction_result = self.correct_full_transcription(transcription)

        if not correction_result["success"]:
            return {
                "method": "qwen_corrector_with_judge",
                "success": False,
                "error": "correction_failed",
                "correction_details": correction_result,
                "metrics": {"duration": time.time() - start_time}
            }

        original_text = correction_result["original_transcription"]
        corrected_text = correction_result["corrected_transcription"]

        # 2. Évaluation qualité
        judge_result = self.judge_correction_quality(original_text, corrected_text)

        total_duration = time.time() - start_time

        return {
            "method": "qwen_corrector_with_judge",
            "success": True,
            "correction": {
                "original_text": original_text,
                "corrected_text": corrected_text,
                "improvement_ratio": correction_result["metrics"]["improvement_ratio"],
                "chunks_processed": correction_result["metrics"]["chunks_processed"]
            },
            "quality_evaluation": judge_result,
            "ready_for_download": correction_result["success"] and judge_result.get("success", False),
            "ground_truth_quality": judge_result.get("qualite_ground_truth", "medium"),
            "metrics": {
                "total_duration": total_duration,
                "correction_duration": correction_result["metrics"]["total_duration"],
                "judge_duration": judge_result.get("metrics", {}).get("duration", 0),
                "corrector_model": self.corrector_model,
                "judge_model": self.judge_model
            }
        }

# Factory et fonctions utilitaires (rétrocompatibilité)
def create_qwen_corrector() -> QwenCorrector:
    """Factory pour créer le correcteur Qwen."""
    return QwenCorrector()

def correct_transcription_for_download(transcription: str) -> Dict:
    """
    Fonction utilitaire pour correction de transcription avant téléchargement.
    Usage : quand le manager clique sur "Télécharger transcription propre".

    MISE À JOUR : Utilise maintenant le pipeline complet avec évaluation.
    """
    corrector = create_qwen_corrector()

    # Utilise la nouvelle API unifiée
    result = corrector.correct_and_evaluate(transcription)

    # Format rétrocompatible
    if result["success"]:
        return {
            "method": "qwen_llm_corrector_with_judge",
            "success": True,
            "corrected_text": result["correction"]["corrected_text"],
            "original_text": result["correction"]["original_text"],
            "quality_scores": result["quality_evaluation"].get("scores", {}),
            "ground_truth_quality": result["ground_truth_quality"],
            "metrics": result["metrics"],
            "ready_for_download": result["ready_for_download"]
        }
    else:
        return {
            "method": "qwen_llm_corrector_with_judge",
            "success": False,
            "error": result["error"],
            "metrics": result["metrics"],
            "ready_for_download": False
        }

# Fonction pipeline complet (nouvelle API recommandée)
def correct_and_evaluate_transcription(transcription: str) -> Dict:
    """
    Pipeline complet correction+évaluation pour usage direct.
    Nouvelle API recommandée post-refacto.
    """
    corrector = create_qwen_corrector()
    return corrector.correct_and_evaluate(transcription)

if __name__ == "__main__":
    pass
