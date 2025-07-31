#!/usr/bin/env python3
"""
Main Corrector - Interface correction transcription Summora V3
Ground Truth orchestrée par LLM - Correction pure sans reformulation
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup path pour imports Summora
sys.path.append(str(Path(__file__).parent.parent))

# Import des modules Summora
from src.llm.llm_qwen_corrector import correct_and_evaluate_transcription
from src.core.metrics.evaluator import create_summora_evaluator, EvaluationReport
from src.qa.spot_checker import SpotChecker

from dotenv import load_dotenv

# Variables d'environnement
load_dotenv()

class TranscriptionCorrectionPipeline:
    """
    Pipeline de correction avec évaluation complète.

    Workflow : Transcription brute → Correction LLM → Évaluation (Juge + Métriques)
    """

    def __init__(self):
        """Initialise le pipeline de correction simplifié."""
        self.evaluator = create_summora_evaluator()

        logger.info("🔧 Pipeline correction initialisé (Qwen+Judge+Evaluator)")

    def load_transcription_file(self, transcription_path: str) -> Dict:
        """
        Charge un fichier transcription depuis output/transcriptions/.

        Args:
            transcription_path: Chemin vers le fichier transcription

        Returns:
            Dict: Contenu transcription + métadonnées
        """
        file_path = Path(transcription_path)

        if not file_path.exists():
            logger.error(f"❌ Fichier transcription non trouvé: {file_path}")
            return {"error": "file_not_found", "path": str(file_path)}

        try:
            # Lecture du fichier TXT
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read().strip()

            logger.info(f"✅ Transcription chargée: {len(raw_text)} caractères")

            return {
                "success": True,
                "raw_transcription": raw_text,
                "source_file": str(file_path),
                "filename": file_path.stem,
                "char_count": len(raw_text),
                "word_count": len(raw_text.split())
            }

        except Exception as e:
            logger.error(f"❌ Erreur lecture transcription: {e}")
            return {"error": str(e), "path": str(file_path)}

    def evaluate_correction_metrics(self, original: str, corrected: str) -> Dict:
        """
        Évaluation quantitative avec evaluator.py (Primary + Experimental).
        Utilise votre EvaluationReport dataclass existante.

        Args:
            original: Transcription brute (référence)
            corrected: Transcription corrigée (hypothèse)

        Returns:
            Dict: Métriques complètes CER/WER/BERT + PER/SemDist
        """
        try:
            logger.info("📊 Évaluation métriques quantitatives (Primary + Experimental)...")

            # Évaluation complète (Primary + Experimental avec PER/SemDist)
            evaluation_report: EvaluationReport = self.evaluator.evaluate_complete(
                reference=original,  # Brute = référence
                hypothesis=corrected,  # Corrigée = hypothèse
                include_experimental=False  # Skip PER/SemDist pour économiser RAM
            )

            # Conversion EvaluationReport → Dict pour sérialisation JSON
            metrics_result = {
                "success": True,
                "primary_metrics": {
                    "cer_score": evaluation_report.cer_result.score if evaluation_report.cer_result else None,
                    "cer_grade": evaluation_report.cer_result.get_grade() if evaluation_report.cer_result else None,
                    "wer_score": evaluation_report.wer_result.score if evaluation_report.wer_result else None,
                    "wer_grade": evaluation_report.wer_result.get_grade() if evaluation_report.wer_result else None,
                    "bert_score": evaluation_report.bert_result.score if evaluation_report.bert_result else None,
                    "bert_grade": evaluation_report.bert_result.get_grade() if evaluation_report.bert_result else None,
                    "composite_score": evaluation_report.primary_composite_score
                },
                "experimental_metrics": {
                    "per_score": evaluation_report.per_result.score if evaluation_report.per_result else None,
                    "per_grade": evaluation_report.per_result.get_grade() if evaluation_report.per_result else None,
                    "semdist_score": evaluation_report.semdist_result.score if evaluation_report.semdist_result else None,
                    "semdist_grade": evaluation_report.semdist_result.get_grade() if evaluation_report.semdist_result else None,
                    "composite_score": evaluation_report.experimental_composite_score
                },
                "overall_assessment": {
                    "grade": evaluation_report.overall_grade,
                    "recommendations": evaluation_report.recommendations,
                    "metrics_used": evaluation_report.metrics_used
                },
                "processing_time": evaluation_report.processing_time_total
            }

            # Log détaillé des résultats
            logger.info(f"✅ Métriques calculées - Grade global: {evaluation_report.overall_grade}")
            logger.info(f"📊 Primary composite: {evaluation_report.primary_composite_score:.3f}")
            logger.info(f"🧪 Experimental composite: {evaluation_report.experimental_composite_score:.3f}")

            return metrics_result

        except Exception as e:
            logger.error(f"❌ Erreur évaluation métriques: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback": True
            }

    def save_corrected_transcription(self, corrected_text: str, original_filename: str) -> str:
        """
        Sauvegarde la transcription corrigée.

        Args:
            corrected_text: Texte corrigé
            original_filename: Nom du fichier original

        Returns:
            str: Chemin du fichier sauvegardé
        """
        # Répertoire de sortie
        output_dir = Path("output/corrector_transcription")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Nom du fichier avec suffixe
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{original_filename}_corrected_{timestamp}.txt"
        output_path = output_dir / output_filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(corrected_text)

            logger.info(f"💾 Transcription corrigée sauvegardée: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            return ""

    def save_evaluation_report(self, evaluation_data: Dict, original_filename: str) -> str:
        """
        Sauvegarde le rapport d'évaluation interne.

        Args:
            evaluation_data: Données d'évaluation complètes
            original_filename: Nom du fichier original

        Returns:
            str: Chemin du rapport sauvegardé
        """
        # Répertoire de sortie
        output_dir = Path("output/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Nom du rapport
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"evaluate_correction_transcription_{original_filename}_{timestamp}.json"
        report_path = output_dir / report_filename

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, ensure_ascii=False, indent=2)

            logger.info(f"📋 Rapport d'évaluation sauvegardé: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde rapport: {e}")
            return ""

    def process_transcription_correction(self, transcription_path: str
                                         ,enable_spot_check = False
                                         ,spot_sample: int=3) -> Dict:
        """
        Pipeline complet de correction avec évaluation.

        Args:
            transcription_path: Chemin vers fichier transcription brute

        Returns:
            Dict: Résultats complets de correction + évaluation
        """
        logger.info(f"🔄 Démarrage pipeline correction: {transcription_path}")
        start_time = datetime.now()

        # 1. Chargement transcription
        load_result = self.load_transcription_file(transcription_path)
        if "error" in load_result:
            return load_result

        original_text = load_result["raw_transcription"]
        filename = load_result["filename"]

        # 2. Correction + Évaluation intégrée (Qwen + Juge)
        logger.info("🤖 Correction+Évaluation Qwen+Judge en cours...")
        correction_result = correct_and_evaluate_transcription(original_text)

        if not correction_result["success"]:
            logger.error("❌ Échec correction+évaluation")
            return {"error": "correction_evaluation_failed", "details": correction_result}

        corrected_text = correction_result["correction"]["corrected_text"]
        judge_evaluation = correction_result["quality_evaluation"]

        # 3. Métriques quantitatives
        logger.info("📊 Calcul métriques quantitatives...")
        metrics_result = self.evaluate_correction_metrics(original_text, corrected_text)

        # 4. Spotcheck QA (optionnel)
        spot_check_file = ""
        if enable_spot_check:
            logger.info("🎯 Génération spot-check QA...")
            spot_checker = SpotChecker(sample_size=spot_sample)
            samples = spot_checker.random_sample(corrected_text)

            logger.info(f"🔍 DEBUG: type(samples)={type(samples)}, samples={samples}")

            if samples:
                spot_check_file = spot_checker.save_samples_for_annotation(samples)
                logger.info(f"📋 {len(samples)} échantillons prêts pour annotation: {spot_check_file}")


        # 5. Sauvegarde transcription corrigée
        corrected_file_path = self.save_corrected_transcription(corrected_text, filename)

        # 6. Assemblage rapport final
        final_report = {
            # Métadonnées
            "processing_timestamp": start_time.isoformat(),
            "processing_duration": (datetime.now() - start_time).total_seconds(),
            "pipeline_version": "summora_v3_ground_truth",

            # Fichiers
            "source_transcription": transcription_path,
            "corrected_transcription_file": corrected_file_path,
            "original_filename": filename,

            # Textes
            "original_transcription": original_text,
            "corrected_transcription": corrected_text,

            # Statistiques de base
            "stats": {
                "original_chars": len(original_text),
                "corrected_chars": len(corrected_text),
                "original_words": len(original_text.split()),
                "corrected_words": len(corrected_text.split()),
                "length_ratio": len(corrected_text) / len(original_text) if original_text else 0
            },

            # Correction LLM (Qwen + Juge intégré)
            "correction_pipeline": {
                "method": correction_result["method"],
                "success": correction_result["success"],
                "ready_for_download": correction_result["ready_for_download"],
                "ground_truth_quality": correction_result["ground_truth_quality"],
                "metrics": correction_result["metrics"]
            },

            # Évaluation qualitative (Juge DeepSeek intégré)
            "judge_evaluation": judge_evaluation,

            # Métriques quantitatives (Primary + Experimental)
            "quantitative_metrics": metrics_result,

            # QA Spot-checker
            "spot_check_file": spot_check_file,
            "spot_check_enabled": bool(spot_check_file),

            # Statut global
            "overall_success": correction_result["success"] and judge_evaluation.get("success", False),
            "ground_truth_quality": correction_result["ground_truth_quality"]
        }

        # 6. Sauvegarde rapport d'évaluation
        report_path = self.save_evaluation_report(final_report, filename)
        final_report["evaluation_report_file"] = report_path

        # 7. Log final
        processing_time = final_report["processing_duration"]
        judge_success = judge_evaluation.get("success", False)
        judge_score = judge_evaluation.get("score_global", 0) if judge_success else judge_evaluation.get("fallback_score", 0)
        overall_grade = metrics_result.get("overall_assessment", {}).get("grade", "D")

        logger.info(f"✅ Pipeline terminé en {processing_time:.2f}s")
        logger.info(f"🎯 Juge: {judge_score}/100 | Métriques: {overall_grade}")
        logger.info(f"📁 Transcription corrigée: {corrected_file_path}")

        return final_report

def main():
    """Interface CLI pour correction de transcription."""

    parser = argparse.ArgumentParser(description="Summora V3 - Correction Ground Truth")
    parser.add_argument("transcription_file",
                       help="Chemin vers fichier transcription (output/transcriptions/...)")
    parser.add_argument("--enable-spot-check", action="store_true",
                        help="Active le spot-check QA pour annotation humaine")
    parser.add_argument("--spot-sample", type=int, default=3,
                        help="Nombre d'échantillons pour spot-check (à défaut: 3)")
    parser.add_argument("--correction-only", action="store_true",
                       help="Mode correction seule (sans évaluation ni métriques)")
    parser.add_argument("--light-eval", action="store_true",
                       help="Mode allégé (skip experimental metrics)")
    parser.add_argument("--verbose", action="store_true", help="Mode verbose")

    args = parser.parse_args()

    # Configuration logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Vérification fichier
    if not Path(args.transcription_file).exists():
        logger.error(f"❌ Fichier transcription non trouvé: {args.transcription_file}")
        sys.exit(1)

    try:
        # Pipeline de correction
        pipeline = TranscriptionCorrectionPipeline()

        # Mode correction seule (rapide)
        if args.correction_only:
            # Chargement transcription
            load_result = pipeline.load_transcription_file(args.transcription_file)
            if "error" in load_result:
                logger.error(f"❌ Échec chargement: {load_result['error']}")
                sys.exit(1)

            original_text = load_result["raw_transcription"]
            filename = load_result["filename"]

            # Correction simple
            from src.llm.llm_qwen_corrector import correct_and_evaluate_transcription
            start_time = datetime.now()

            logger.info("🤖 Correction seule en cours...")
            correction_result = correct_and_evaluate_transcription(original_text)

            if not correction_result["success"]:
                logger.error(f"❌ Échec correction: {correction_result['error']}")
                sys.exit(1)

            corrected_text = correction_result["correction"]["corrected_text"]
            processing_time = (datetime.now() - start_time).total_seconds()

            # Sauvegarde rapide
            corrected_file = pipeline.save_corrected_transcription(corrected_text, filename)

            # Affichage simplifié
            print("\n" + "="*50)
            print("🔧 CORRECTION SEULE - RÉSULTATS")
            print("="*50)

            print(f"\n📁 Source: {args.transcription_file}")
            print(f"📝 Corrigé: {corrected_file}")
            print(f"⏱️  Temps: {processing_time:.2f}s")

            print(f"\n📊 STATISTIQUES:")
            original_words = len(original_text.split())
            corrected_words = len(corrected_text.split())
            print(f"   • Mots: {original_words} → {corrected_words}")
            print(f"   • Ratio: {corrected_words/original_words:.2f}")

            print(f"\n📋 APERÇU CORRECTION:")
            print(f"AVANT: {original_text[:150]}...")
            print(f"APRÈS: {corrected_text[:150]}...")

            print(f"\n✅ Correction terminée ! Fichier: {corrected_file}")
            return

        # Pipeline complet (existant)
        results = pipeline.process_transcription_correction(
            args.transcription_file
            ,enable_spot_check=getattr(args, 'enable_spot_check', False)
            ,spot_sample=getattr(args, 'spot_sample', 3)
            )

        if "error" in results:
            logger.error(f"❌ Échec pipeline: {results['error']}")
            sys.exit(1)

        # Affichage résultats
        print("\n" + "="*70)
        print("🎯 RÉSULTATS CORRECTION GROUND TRUTH")
        print("="*70)

        print(f"\n📁 Source: {results['source_transcription']}")
        print(f"📝 Corrigé: {results['corrected_transcription_file']}")
        print(f"📋 Rapport: {results['evaluation_report_file']}")

        print(f"\n📊 STATISTIQUES:")
        stats = results['stats']
        print(f"   • Mots: {stats['original_words']} → {stats['corrected_words']}")
        print(f"   • Ratio longueur: {stats['length_ratio']:.2f}")

        print(f"\n🤖 CORRECTION + JUGE INTÉGRÉ:")
        correction = results['correction_pipeline']
        print(f"   • Méthode: {correction['method']}")
        print(f"   • Qualité Ground Truth: {correction['ground_truth_quality'].upper()}")
        print(f"   • Prêt téléchargement: {'Oui' if correction['ready_for_download'] else 'Non'}")

        print(f"\n🎯 ÉVALUATION JUGE:")
        judge = results['judge_evaluation']
        if judge.get('success'):
            scores = judge.get('scores', {})
            print(f"   • Score global: {scores.get('score_global', 'N/A')}/100")
            print(f"   • Note sur 10: {scores.get('note_sur_10', 'N/A')}/10")
            print(f"   • Qualité GT: {scores.get('qualite_ground_truth', 'N/A')}")
        else:
            print(f"   • Erreur: {judge.get('error', 'Inconnue')}")
            if judge.get('fallback_score'):
                print(f"   • Score fallback: {judge['fallback_score']}")

        print(f"\n📊 MÉTRIQUES QUANTITATIVES:")
        metrics = results['quantitative_metrics']
        if metrics.get('success'):
            assessment = metrics['overall_assessment']
            primary = metrics['primary_metrics']
            experimental = metrics['experimental_metrics']
            print(f"   • Grade global: {assessment['grade']}")
            print(f"   • Primary composite: {primary['composite_score']:.3f}")
            print(f"   • Experimental composite: {experimental['composite_score']:.3f}")
            print(f"   • PER: {experimental.get('per_score', 'N/A')} | SemDist: {experimental.get('semdist_score', 'N/A')}")
        else:
            print(f"   • Erreur: {metrics.get('error', 'Inconnue')}")

        print(f"\n✅ Qualité Ground Truth: {results['ground_truth_quality'].upper()}")
        print(f"⏱️  Temps traitement: {results['processing_duration']:.2f}s")

    except KeyboardInterrupt:
        logger.info("🛑 Traitement interrompu")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
