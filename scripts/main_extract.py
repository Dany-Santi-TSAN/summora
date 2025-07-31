#!/usr/bin/env python3
"""
Summora V3 - Module d'Extraction Intelligent - Architecture Allégée
Pipeline cascade : Qwen Enhanced → LLM fallback gratuit → YAKE + BERTScore + SpotChecker
Usage: python scripts/main_extract.py transcription.txt --with-eval --enable-spot-check
"""
import argparse
import logging
import re
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Path setup pour imports
sys.path.append(str(Path(__file__).parent.parent))

# Imports Summora
from src.llm.llm_qwen_enhanced_extractor import extract_with_qwen_enhanced
from src.llm.llm_fallback_extractor import extract_with_fallback_llm, FallBackExtractor
from src.meeting.extractor import extract_meeting_content
from src.config.meeting_config import MeetingQualityThresholds
from src.qa.spot_checker import SpotChecker
from src.core.metrics.bertscore import BERTScoreCalculator

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# === CLASSE 1: ExtractionCascade (Pipeline extraction uniquement) ===
class ExtractionCascade:
    """Pipeline d'extraction en cascade avec fallbacks métiers."""

    def __init__(self):

        self.llm_fallback_extractor = FallBackExtractor()
        self.extraction_methods = [
            {
                'name': 'qwen_enhanced'
                ,'function': extract_with_qwen_enhanced
                ,'description': 'Qwen Enhanced (YAKE+LLM)'
            },
            {
                'name': 'llm_fallback_extractor'
                ,'function': extract_with_fallback_llm
                ,'description': f'LLM : {self.llm_fallback_extractor.extractor_model_name} - version gratuite (fallback)' # j'essaie de changer ici
            },
            {
                'name': 'yake_fallback'
                ,'function': self._extract_with_yake_adapted
                ,'description': 'YAKE (fallback)'
            }
        ]
        logger.info("🎯 Pipeline extraction cascade initialisé")

    def _extract_with_yake_adapted(self, transcription: str) -> Dict:
        """Adapte la sortie YAKE au format attendu par le pipeline."""
        try:
            result = extract_meeting_content(transcription)
            return {
                'method': 'yake_fallback',
                'success': 'error' not in result,
                'extraction': {
                    'topics_principaux': [t['keyword'] for t in result.get('topics', {}).get('topics', [])[:5]],
                    'points_a_retenir': [
                        action['action'] for action in result.get('actions', {}).get('actions', [])[:5]
                    ] + [
                        decision['decision'] for decision in result.get('decisions', {}).get('decisions', [])[:5]
                    ],
                    'resume_abstractif': f"Meeting analysé avec {len(result.get('topics', {}).get('topics', []))} topics détectés par YAKE."
                },
                'quality_scores': {'score_global': 65},
                'metrics': {'method': 'yake_extraction_only'}
            }
        except Exception as e:
            return {'method': 'yake_fallback', 'success': False, 'error': str(e)}

    def extract(self, transcription: str) -> Dict:
        """Exécute le pipeline d'extraction en cascade."""
        logger.info("🎯 Démarrage pipeline extraction cascade")
        extraction_attempts = []

        for i, method in enumerate(self.extraction_methods, 1):
            method_name = method['name']
            method_func = method['function']
            method_desc = method['description']

            logger.info(f"🧠 Tentative {i}: {method_desc}")

            try:
                result = method_func(transcription)
                extraction_attempts.append({
                    'method': method_name,
                    'success': result.get('success', False),
                    'error': result.get('error') if not result.get('success') else None
                })

                if result.get('success'):
                    logger.info(f"✅ {method_desc} réussi")
                    result['cascade_method'] = method_name
                    result['attempts'] = extraction_attempts
                    return result

            except Exception as e:
                logger.warning(f"⚠️ {method_desc} échoué: {str(e)}")
                extraction_attempts.append({
                    'method': method_name,
                    'success': False,
                    'error': str(e)
                })

        # Échec total
        logger.error("❌ Tous les extracteurs ont échoué")
        return {
            'method': 'extraction_failed',
            'success': False,
            'extraction': {
                'topics_principaux': [],
                'points_a_retenir': ['Extraction automatique impossible'],
                'resume_abstractif': 'Meeting nécessitant une analyse manuelle.'
            },
            'cascade_method': 'failed',
            'attempts': extraction_attempts,
            'error': 'all_extractors_failed'
        }


# === CLASSE 2: EvaluationExtraction (BERTScore + SpotChecker strategic) ===
class EvaluationExtraction:
    """Évaluation des extractions avec BERTScore et SpotChecker strategic."""

    def __init__(self):
        self.bertscore_calculator = BERTScoreCalculator()
        logger.info("📊 Évaluateur extraction initialisé (BERTScore + SpotChecker)")

    def evaluate_with_bertscore(self, transcription_brute: str, extraction_result: Dict) -> Dict:
        """
        Évaluation transcription vs extraction avec LLM Juge (backbone) + BERTScore (fallback).
        Architecture simplifiée : Juge business intelligent → BERTScore sémantique
        """
        try:
            logger.info("📊 Évaluation: LLM Juge (backbone) + BERTScore (fallback)")

            # Extraction du texte depuis résultat
            extracted_text = self._extract_text_from_result(extraction_result)

            if not extracted_text or not transcription_brute.strip():
                logger.warning("⚠️ Textes insuffisants pour évaluation")
                return {
                    "success": False,
                    "error": "insufficient_text_for_evaluation",
                    "bertscore": 0.0
                }

            # 1. LLM Juge (si disponible dans extraction_result)
            llm_judge_available = extraction_result.get('quality_evaluation', {}).get('success', False)

            # 2. BERTScore (fallback sémantique)
            logger.info("🧠 Calcul BERTScore (évaluation sémantique)...")
            bertscore_result = self.bertscore_calculator.calculate(transcription_brute, extracted_text)
            bertscore_score = bertscore_result.score
            bertscore_grade = bertscore_result.get_grade()

            # Recommandations selon hiérarchie: Juge > BERTScore
            if llm_judge_available:
                recommendations = ["🤖 LLM Juge disponible - évaluation business optimale"]
                primary_score = extraction_result['quality_evaluation'].get('score_global', 0) / 100
                primary_method = "llm_judge"
            elif bertscore_score > 0.5:
                recommendations = ["🧠 BERTScore élevé - bonne préservation sémantique"]
                primary_score = bertscore_score
                primary_method = "bertscore"
            else:
                recommendations = ["⚠️ Score sémantique faible - vérifier qualité extraction"]
                primary_score = bertscore_score
                primary_method = "bertscore_low"

            logger.info(f"✅ Évaluation: {primary_method} = {primary_score:.3f}")

            return {
                "success": True,
                "primary_method": primary_method,
                "primary_score": primary_score,

                # Détails par métrique
                "llm_judge": {
                    "available": llm_judge_available,
                    "score": extraction_result.get('quality_evaluation', {}).get('score_global', 0) if llm_judge_available else None
                },
                "bertscore": {
                    "score": bertscore_score,
                    "grade": bertscore_grade,
                    "precision": bertscore_result.details.get("precision", 0),
                    "recall": bertscore_result.details.get("recall", 0),
                    "model": bertscore_result.details.get("model", "xlm-roberta-base")
                },

                "recommendations": recommendations,
                "evaluation_hierarchy": "LLM Judge > BERTScore",
                "processing_time": bertscore_result.processing_time
            }

        except Exception as e:
            logger.error(f"❌ Erreur évaluation: {e}")
            return {"success": False, "error": str(e), "bertscore": 0.0}

    def generate_strategic_spotcheck(self, extraction_result: Dict, transcription: str,
                                   spot_sample_size: int = 3) -> Dict:
        """Génère un spot-check strategic sur les mots-clés business extraits."""
        try:
            logger.info("🎯 Génération spot-check strategic...")

            # Extraction des mots-clés business
            business_keywords = self._extract_business_keywords(extraction_result)

            # Fallback vers mots-clés standards si peu trouvés
            if len(business_keywords) < 3:
                business_keywords.extend(['action', 'décision', 'objectif', 'planning', 'réunion'])

            # Déduplication et limitation
            unique_keywords = list(dict.fromkeys([k.lower() for k in business_keywords if len(k) > 3]))[:8]
            logger.info(f"🔍 Mots-clés strategic: {unique_keywords[:5]}")

            # Génération échantillons strategic
            spot_checker = SpotChecker(sample_size=spot_sample_size)
            strategic_samples = spot_checker.strategic_sample(
                transcription,
                unique_keywords,
                context_window=600
            )

            if not strategic_samples:
                logger.warning("⚠️ Aucun échantillon strategic généré")
                return {
                    "success": False,
                    "error": "no_strategic_samples",
                    "keywords_used": unique_keywords
                }

            # Sauvegarde échantillons
            spot_check_file = spot_checker.save_samples_for_annotation(strategic_samples)

            logger.info(f"📋 {len(strategic_samples)} échantillons strategic sauvés: {spot_check_file}")

            return {
                "success": True,
                "samples_count": len(strategic_samples),
                "keywords_used": unique_keywords,
                "spot_check_file": spot_check_file,
                "sample_method": "strategic",
                "context_window": 600
            }

        except Exception as e:
            logger.error(f"❌ Erreur spot-check strategic: {e}")
            return {"success": False, "error": str(e), "keywords_used": []}

    def _extract_text_from_result(self, result: Dict) -> str:
        """Extrait le texte d'un résultat d'extraction pour comparaison."""
        if not result.get('success') or not result.get('extraction'):
            return ""

        extraction = result['extraction']
        return f"{' '.join(extraction.get('topics_principaux', []))} " \
               f"{' '.join(extraction.get('points_a_retenir', []))} " \
               f"{extraction.get('resume_abstractif', '')}"

    def _extract_business_keywords(self, extraction_result: Dict) -> List[str]:
        """Extrait les mots-clés business depuis les résultats d'extraction."""
        business_keywords = []

        if extraction_result.get('success') and extraction_result.get('extraction'):
            extraction_data = extraction_result['extraction']

            # Topics comme mots-clés
            topics = extraction_data.get('topics_principaux', [])
            business_keywords.extend([t.split()[0] for t in topics if len(t.split()) > 0])

            # Mots-clés depuis insights business
            insights = extraction_data.get('insights_business', {})
            for key in ['actions_cles', 'decisions_prises', 'next_steps']:
                items = insights.get(key, [])
                for item in items:
                    words = item.split()[:2]  # Premiers mots significatifs
                    business_keywords.extend(words)

        return business_keywords


# === CLASSE 3: ExtractionSaver (Sauvegarde résultats + évaluations) ===
class ExtractionSaver:
    """Gestionnaire de sauvegarde pour les résultats d'extraction."""

    def __init__(self, output_dir: str = "output/extractions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"💾 Sauvegarde configurée: {self.output_dir}")

    def save_results(self, results: Dict) -> str:
        """Sauvegarde les résultats dans output/extractions avec timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcription_name = Path(results["transcription_file"]).stem

        # Nom selon méthode cascade utilisée
        method = results["cascade_info"]["method_used"]
        filename = f"extraction_{method}_{transcription_name}_{timestamp}.json"

        output_path = self.output_dir / filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Extraction sauvée: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            return ""


# === TranscriptionQualityAnalyzer (gardé mais simplifié) ===
class TranscriptionQualityAnalyzer:
    """Analyseur de qualité transcription pour métriques extraction."""

    def __init__(self):
        self.thresholds = MeetingQualityThresholds()
        self.transcription_thresholds = {
            'min_length': 100,
            'min_words': 20,
            'min_sentences': 3,
            'structure_indicators': ['décision', 'action', 'objectif', 'planning', 'réunion']
        }

    def analyze_transcription_quality(self, transcription: str) -> Dict:
        """Analyse la qualité de la transcription pour métriques extraction."""
        quality_scores = {}
        issues = []

        # Métriques de base
        char_count = len(transcription)
        word_count = len(transcription.split())
        sentence_count = len([s for s in transcription.split('.') if s.strip()])

        quality_scores['char_count'] = char_count
        quality_scores['word_count'] = word_count
        quality_scores['sentence_count'] = sentence_count

        # Vérifications qualité
        if char_count < self.transcription_thresholds['min_length']:
            issues.append(f"Transcription très courte ({char_count} caractères)")

        if word_count < self.transcription_thresholds['min_words']:
            issues.append(f"Peu de contenu ({word_count} mots)")

        if sentence_count < self.transcription_thresholds['min_sentences']:
            issues.append(f"Structure fragmentée ({sentence_count} phrases)")

        # Densité business
        text_lower = transcription.lower()
        business_terms = sum(1 for term in self.transcription_thresholds['structure_indicators']
                           if term in text_lower)
        business_density = (business_terms / word_count * 100) if word_count > 0 else 0

        quality_scores['business_density'] = business_density
        quality_scores['business_terms_found'] = business_terms

        if business_density < self.thresholds.min_actionable_density:
            issues.append(f"Faible densité business ({business_density:.1f}%)")

        # Score global
        length_score = min(char_count / 1000, 1.0) * 30
        structure_score = min(sentence_count / 10, 1.0) * 30
        business_score = min(business_density / 5, 1.0) * 40

        global_score = int(length_score + structure_score + business_score)

        if global_score >= 85:
            grade = "A"
            quality_level = "excellent"
        elif global_score >= 75:
            grade = "B"
            quality_level = "bon"
        elif global_score >= 65:
            grade = "C"
            quality_level = "acceptable"
        else:
            grade = "D"
            quality_level = "faible"

        return {
            'global_score': global_score,
            'grade': grade,
            'quality_level': quality_level,
            'detailed_scores': quality_scores,
            'issues_detected': issues,
            'needs_improvement': len(issues) > 0 or global_score < 75
        }


# === FONCTIONS PRINCIPALES ===
def extract_from_transcription(transcription_file: str,
                              enable_recommendations: bool = False,
                              enable_spot_check: bool = False,
                              enable_bertscore_eval: bool = False,
                              spot_sample_size: int = 3) -> Dict:
    """
    Extraction complète depuis un fichier de transcription avec cascade et évaluations.
    """
    transcription_file = Path(transcription_file)

    if not transcription_file.exists():
        return {"error": "file_not_found", "path": str(transcription_file)}

    logger.info(f"🎬 Extraction depuis transcription: {transcription_file.name}")
    start_time = datetime.now()

    # 1. Lecture de la transcription
    logger.info("📖 Lecture transcription...")
    try:
        if transcription_file.suffix.lower() == '.json':
            with open(transcription_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                transcription = data.get('content', {}).get('text', '')
                metadata = data.get('transcription', {})
        else:
            with open(transcription_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "TRANSCRIPTION COMPLÈTE" in content:
                    transcription = content.split("TRANSCRIPTION COMPLÈTE")[1].split("="*70)[0].strip()
                else:
                    transcription = content
                metadata = {}
    except Exception as e:
        logger.error(f"❌ Erreur lecture fichier: {e}")
        return {"error": "file_read_failed", "details": str(e)}

    if not transcription.strip():
        logger.error("❌ Transcription vide")
        return {"error": "empty_transcription"}

    logger.info(f"📝 Transcription: {len(transcription)} chars, {len(transcription.split())} mots")

    # 2. Analyse qualité transcription
    logger.info("📊 Analyse qualité transcription...")
    quality_analyzer = TranscriptionQualityAnalyzer()
    transcription_quality = quality_analyzer.analyze_transcription_quality(transcription)

    # 3. Extraction cascade
    logger.info("🎯 Extraction cascade...")
    cascade = ExtractionCascade()
    extraction_result = cascade.extract(transcription)

    # 4. Évaluations (si activées)
    evaluator = EvaluationExtraction()

    # BERTScore evaluation (transcription brute vs extraction)
    bertscore_evaluation = {}
    if enable_bertscore_eval:
        bertscore_evaluation = evaluator.evaluate_with_bertscore(transcription, extraction_result)

    # Spot-check strategic
    strategic_spotcheck = {}
    if enable_spot_check:
        strategic_spotcheck = evaluator.generate_strategic_spotcheck(
            extraction_result, transcription, spot_sample_size
        )

    # 5. Recommandations via main_reco.py (si activées)
    recommendations = []
    if enable_recommendations:
        logger.info("💡 Appel main_reco.py pour recommandations...")
        try:
            # Import dynamique pour éviter dépendance circulaire
            sys.path.append(str(Path(__file__).parent))
            from main_reco import analyze_and_recommend

            # Préparation des données au format attendu par main_reco.py
            extraction_data_for_reco = {
                'transcription': {
                    'text': transcription,
                    'quality_analysis': transcription_quality
                },
                'extraction': extraction_result,
                'cascade_info': {
                    'method_used': extraction_result.get('cascade_method', 'unknown'),
                    'success': extraction_result.get('success', False)
                }
            }

            # Sauvegarde temporaire pour main_reco.py
            temp_file = Path("temp_extraction_for_reco.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_data_for_reco, f, indent=2, ensure_ascii=False)

            # Appel main_reco.py
            reco_results = analyze_and_recommend(str(temp_file))

            # Nettoyage
            temp_file.unlink()

            if reco_results.get('recommendations', {}).get('success'):
                recommendations = reco_results['recommendations'].get('recommendations', [])
                logger.info(f"✅ {len(recommendations)} recommandations générées")
            else:
                logger.warning("⚠️ Échec génération recommandations")

        except Exception as e:
            logger.error(f"❌ Erreur appel main_reco.py: {e}")

    total_duration = (datetime.now() - start_time).total_seconds()

    # Résultats consolidés
    results = {
        'transcription_file': str(transcription_file),
        'analysis_timestamp': datetime.now().isoformat(),
        'total_duration': total_duration,

        # Transcription
        'transcription': {
            'text': transcription,
            'metadata': metadata,
            'quality_analysis': transcription_quality
        },

        # Extraction cascade
        'extraction': extraction_result,
        'cascade_info': {
            'method_used': extraction_result.get('cascade_method', 'unknown'),
            'attempts': extraction_result.get('attempts', []),
            'success': extraction_result.get('success', False)
        },

        # Recommandations (si demandées)
        'recommendations': recommendations
    }

    # Ajout des évaluations si disponibles
    if bertscore_evaluation:
        results['bertscore_evaluation'] = bertscore_evaluation

    if strategic_spotcheck:
        results['strategic_spotcheck'] = strategic_spotcheck

    return results


def print_results_summary(results: Dict):
    """Affiche un résumé des résultats avec évaluations."""
    print("\n" + "="*70)
    print("🎯 SUMMORA V3 - EXTRACTION INTELLIGENTE")
    print("="*70)

    # Infos générales
    transcription_file = Path(results["transcription_file"]).name
    transcription_quality = results["transcription"]["quality_analysis"]
    extraction = results["extraction"]

    print(f"\n📁 Source: {transcription_file}")
    print(f"⏱️ Temps traitement: {results['total_duration']:.2f}s")

    # Qualité transcription
    trans_score = transcription_quality["global_score"]
    trans_grade = transcription_quality["grade"]
    print(f"\n📊 QUALITÉ TRANSCRIPTION: {trans_score}/100 (Grade {trans_grade})")

    # Pipeline cascade
    cascade_info = results["cascade_info"]
    method_used = cascade_info["method_used"]
    print(f"\n🎯 PIPELINE CASCADE:")
    print(f"   • Méthode utilisée: {method_used}")
    print(f"   • Succès: {'Oui' if cascade_info['success'] else 'Non'}")

    if cascade_info["success"] and extraction.get("extraction"):
        topics = extraction["extraction"].get("topics_principaux", [])
        points = extraction["extraction"].get("points_a_retenir", [])
        resume = extraction["extraction"].get("resume_abstractif", "")

        print(f"\n📋 CONTENU EXTRAIT:")
        print(f"   • Topics: {len(topics)}")
        for i, topic in enumerate(topics[:3], 1):
            print(f"     {i}. {topic}")
        if len(topics) > 3:
            print(f"     ... et {len(topics) - 3} autres")

        print(f"   • Points clés: {len(points)}")
        for i, point in enumerate(points[:3], 1):
            print(f"     {i}. {point}")
        if len(points) > 3:
            print(f"     ... et {len(points) - 3} autres")

        if resume:
            print(f"   • Résumé: {resume[:100]}{'...' if len(resume) > 100 else ''}")

    # Évaluation BERTScore
    if 'bertscore_evaluation' in results:
        eval_data = results['bertscore_evaluation']
        print(f"\n📊 ÉVALUATION BERTSCORE:")
        if eval_data.get('success'):
            primary_method = eval_data['primary_method']
            primary_score = eval_data['primary_score']
            print(f"   • Méthode principale: {primary_method}")
            print(f"   • Score principal: {primary_score:.3f}")

            bertscore = eval_data['bertscore']
            print(f"   • 🧠 BERTScore: {bertscore['score']:.3f} (Grade {bertscore['grade']})")

            recommendations = eval_data.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    print(f"     • {rec}")
        else:
            print(f"   • Erreur: {eval_data.get('error', 'Inconnue')}")

    # SpotCheck strategic
    if 'strategic_spotcheck' in results:
        spot = results['strategic_spotcheck']
        print(f"\n🎯 SPOTCHECK STRATEGIC:")
        if spot.get('success'):
            print(f"   • Échantillons générés: {spot['samples_count']}")
            print(f"   • Mots-clés ciblés: {len(spot.get('keywords_used', []))}")
            print(f"   • Fichier QA: {Path(spot['spot_check_file']).name}")
        else:
            print(f"   • Erreur: {spot.get('error', 'Aucun échantillon généré')}")

    # Recommandations meeting (si demandées)
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\n💡 RECOMMANDATIONS MEETING:")
        for i, rec in enumerate(recommendations[:3], 1):
            if isinstance(rec, dict):
                title = rec.get('titre', str(rec))[:60]
                category = rec.get('categorie', 'Conseil')
                print(f"   {i}. [{category}] {title}")
            else:
                print(f"   {i}. {str(rec)[:60]}")
        if len(recommendations) > 3:
            print(f"   ... et {len(recommendations) - 3} autres")

    print(f"\n✅ Pipeline terminé - Méthode: {method_used.upper()}")
    print("="*70)


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Summora V3 - Extraction Intelligente (Allégée) avec Cascade + BERTScore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
python scripts/main_extract.py transcription.txt                     # Extraction cascade basique
python scripts/main_extract.py transcription.txt --extraction-only  # Test rapide
python scripts/main_extract.py transcription.txt --with-reco        # Avec recommandations (via main_reco.py)
python scripts/main_extract.py transcription.txt --with-eval        # Avec évaluation BERTScore
python scripts/main_extract.py transcription.txt --enable-spot-check  # Avec SpotCheck strategic
python scripts/main_extract.py transcription.txt --with-eval --enable-spot-check --with-reco  # Complet

Formats supportés: .txt (brut) | .json (avec métadonnées)
        """
    )

    parser.add_argument("transcription_file", help="Fichier de transcription à analyser (.txt ou .json)")
    parser.add_argument("--extraction-only", action="store_true",
                       help="Mode extraction seule (sans évaluations ni recommandations)")
    parser.add_argument("--with-recommendations", "--with-reco", action="store_true",
                       help="Génère des recommandations d'amélioration (via main_reco.py)")
    parser.add_argument("--with-eval", action="store_true",
                       help="Active évaluation BERTScore")
    parser.add_argument("--enable-spot-check", action="store_true",
                       help="Active spot-check strategic QA")
    parser.add_argument("--spot-sample", type=int, default=3,
                       help="Nombre échantillons spot-check")
    parser.add_argument("--output", "-o", help="Répertoire de sortie")
    parser.add_argument("--no-save", action="store_true", help="Ne sauvegarde pas")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    parser.add_argument("--quiet", "-q", action="store_true", help="Mode silencieux")

    args = parser.parse_args()

    # Configuration logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Vérification fichier
    if not Path(args.transcription_file).exists():
        logger.error(f"❌ Fichier transcription non trouvé: {args.transcription_file}")
        sys.exit(1)

    try:
        # Mode extraction seule (rapide)
        if args.extraction_only:
            logger.info("🎯 Mode extraction seule activé")
            start_time = datetime.now()

            # Extraction cascade uniquement
            cascade = ExtractionCascade()

            # Lecture transcription simple
            with open(args.transcription_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "TRANSCRIPTION COMPLÈTE" in content:
                    transcription = content.split("TRANSCRIPTION COMPLÈTE")[1].split("="*70)[0].strip()
                else:
                    transcription = content

            extraction_result = cascade.extract(transcription)
            processing_time = (datetime.now() - start_time).total_seconds()

            # Affichage simplifié
            print("\n" + "="*50)
            print("🎯 EXTRACTION SEULE - RÉSULTATS")
            print("="*50)

            print(f"\n📁 Source: {args.transcription_file}")
            print(f"⏱️ Temps: {processing_time:.2f}s")
            print(f"🎯 Méthode: {extraction_result.get('cascade_method', 'unknown')}")

            if extraction_result.get('success') and extraction_result.get('extraction'):
                extraction_data = extraction_result['extraction']
                topics = extraction_data.get('topics_principaux', [])
                points = extraction_data.get('points_a_retenir', [])

                print(f"\n📊 CONTENU EXTRAIT:")
                print(f"   • Topics: {len(topics)}")
                print(f"   • Points clés: {len(points)}")

                print(f"\n📋 APERÇU:")
                if topics:
                    print(f"TOP TOPIC: {topics[0]}")
                if points:
                    print(f"POINT CLÉ: {points[0]}")

                resume = extraction_data.get('resume_abstractif', '')
                if resume:
                    print(f"RÉSUMÉ: {resume[:100]}...")
            else:
                print("❌ Extraction échouée")

            print(f"\n✅ Extraction terminée !")
            return 0

        # Pipeline complet
        results = extract_from_transcription(
            args.transcription_file,
            enable_recommendations=args.with_recommendations,
            enable_spot_check=args.enable_spot_check,
            enable_bertscore_eval=args.with_eval,
            spot_sample_size=args.spot_sample
        )

        if "error" in results:
            logger.error(f"❌ Erreur pipeline: {results['error']}")
            sys.exit(1)

        # Sauvegarde
        if not args.no_save:
            output_dir = args.output or "output/extractions"
            saver = ExtractionSaver(output_dir)
            main_file = saver.save_results(results)

        # Affichage résultats (si pas silencieux)
        if not args.quiet:
            print_results_summary(results)

        return 0

    except KeyboardInterrupt:
        logger.info("🛑 Analyse interrompue")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)


# === FONCTIONS HELPER POUR COMPATIBILITÉ ===
def extract_with_cascade(transcription: str) -> Dict:
    """Helper function pour compatibilité avec l'API existante."""
    cascade = ExtractionCascade()
    return cascade.extract(transcription)


if __name__ == "__main__":
    sys.exit(main())
