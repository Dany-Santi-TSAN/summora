"""
Summora - Module d'Analyse Audio
Analyse des propriétés audio avec visualisations pour meetings
Usage: python main_visual.py audio.mp3 --plots --model-comparison
"""
import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Path setup pour imports Summora
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Imports Summora
try:
    from src.core.audio_analyzer import analyze_meeting_audio_file
    from src.core.utils import validate_audio_path, get_supported_formats
    SUMMORA_AVAILABLE = True
except ImportError as e:
    print(f"❌ Erreur import Summora modules: {e}")
    SUMMORA_AVAILABLE = False

def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure le logging."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def save_analysis_results(analysis_result: Dict, audio_path: Path,
                         output_file: Optional[str] = None) -> str:
    """
    Sauvegarde les résultats d'analyse audio.

    Args:
        analysis_result: Résultats de l'analyse
        audio_path: Chemin du fichier audio source
        output_file: Nom du fichier de sortie (optionnel)

    Retourne:
        str: Chemin du fichier sauvegardé
    """
    if output_file:
        save_path = Path(output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path('output/audio_analysis') / f"audio_analysis_{audio_path.stem}_{timestamp}.json"

    # Créer le dossier si nécessaire
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Préparation des données pour JSON
    json_data = {
        "source": {
            "file_name": audio_path.name
            ,"file_path": str(audio_path.absolute())
            ,"analysis_timestamp": datetime.now().isoformat()
        },
        "analysis": analysis_result,
        "metadata": {
            "summora_version": "1.0"
            ,"analysis_type": "meeting_audio_properties"
        }
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return str(save_path)

def print_analysis_summary(analysis_result: Dict, audio_path: Path):
    """Affiche un résumé des résultats d'analyse audio."""

    if "error" in analysis_result:
        print(f"❌ Erreur analyse: {analysis_result['error']}")
        return

    print("\n" + "="*60)
    print("🎵 SUMMORA - ANALYSE AUDIO MEETING")
    print("="*60)

    # Infos générales
    print(f"📁 Fichier           : {audio_path.name}")
    print(f"⏱️  Durée            : {analysis_result.get('duration_formatted', 'N/A')}")
    print(f"🎛️  Sample Rate      : {analysis_result.get('sample_rate', 0):,} Hz")
    print(f"📊 Nombre échantillons: {analysis_result.get('samples_count', 0):,}")

    # Métriques d'amplitude
    print(f"\n🔊 MÉTRIQUES AMPLITUDE")
    print("-" * 30)
    print(f"Amplitude max       : {analysis_result.get('max_amplitude', 0):.3f}")
    print(f"Énergie RMS         : {analysis_result.get('rms_energy', 0):.3f}")
    print(f"Amplitude moyenne   : {analysis_result.get('mean_amplitude', 0):.3f}")

    # Métriques spécifiques meetings
    print(f"\n🎤 MÉTRIQUES MEETING")
    print("-" * 30)
    silence_ratio = analysis_result.get('silence_ratio', 0)
    speech_ratio = analysis_result.get('speech_ratio', 0)
    print(f"Ratio silence       : {silence_ratio*100:.1f}%")
    print(f"Ratio parole        : {speech_ratio*100:.1f}%")
    print(f"Dynamic range       : {analysis_result.get('dynamic_range_db', 0):.1f} dB")
    print(f"Zero crossing rate  : {analysis_result.get('zero_crossing_rate', 0):.3f}")

    # Score qualité meeting
    quality_score = analysis_result.get('meeting_quality_score', 0)
    quality_grade = analysis_result.get('meeting_quality_grade', 'N/A')
    print(f"\n📈 QUALITÉ MEETING")
    print("-" * 25)
    print(f"Score global        : {quality_score}/100")
    print(f"Grade               : {quality_grade}")

    # Status détaillés
    meeting_status = analysis_result.get('meeting_status', {})
    if meeting_status:
        print(f"\n📋 STATUS DÉTAILLÉS")
        print("-" * 25)
        for key, status in meeting_status.items():
            print(f"• {status}")

    # Recommandations
    recommendations = analysis_result.get('recommendations', [])
    if recommendations:
        print(f"\n💡 RECOMMANDATIONS")
        print("-" * 25)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")

    # Analyse vocale
    mfcc_coeffs = analysis_result.get('mfcc_coefficients', [])
    vocal_clarity = analysis_result.get('vocal_clarity_score', 0)
    if mfcc_coeffs:
        print(f"\n🎵 ANALYSE VOCALE")
        print("-" * 20)
        print(f"Clarté vocale       : {vocal_clarity:.3f}")
        print(f"MFCC (5 premiers)   : {[f'{x:.3f}' for x in mfcc_coeffs]}")

    print("="*60)

def compare_audio_models(audio_path: Path, models: list = None):
    """
    Compare différents modèles d'analyse audio.

    Args:
        audio_path: Chemin vers le fichier audio
        models: Liste des modèles à comparer
    """
    print(f"\n🔬 COMPARAISON MODÈLES AUDIO")
    print("-" * 40)
    print(f"📁 Fichier: {audio_path.name}")

    if models is None:
        models = ["librosa_standard", "librosa_meeting_optimized"]

    results = {}

    for model in models:
        print(f"\n⚙️ Test modèle: {model}")

        try:
            if model == "librosa_standard":
                # Configuration standard
                analysis = analyze_meeting_audio_file(
                    audio_path
                    ,generate_plots=False
                    ,silence_percentile=20.0  # Plus conservateur
                )
            elif model == "librosa_meeting_optimized":
                # Configuration optimisée meetings
                analysis = analyze_meeting_audio_file(
                    audio_path
                    ,generate_plots=False
                    ,silence_percentile=15.0  # Plus sensible
                    ,min_meeting_duration=30.0
                )
            else:
                print(f"⚠️ Modèle inconnu: {model}")
                continue

            if "error" not in analysis:
                results[model] = {
                    "quality_score": analysis.get('meeting_quality_score', 0)
                    ,"speech_ratio": analysis.get('speech_ratio', 0)
                    ,"dynamic_range": analysis.get('dynamic_range_db', 0)
                }
                print(f"   ✅ Score qualité: {results[model]['quality_score']}/100")
                print(f"   🎤 Ratio parole: {results[model]['speech_ratio']*100:.1f}%")
            else:
                print(f"   ❌ Erreur: {analysis['error']}")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

    # Résumé comparaison
    if len(results) > 1:
        print(f"\n📊 RÉSUMÉ COMPARAISON")
        print("-" * 30)
        best_model = max(results.keys(), key=lambda x: results[x]['quality_score'])
        print(f"🏆 Meilleur modèle: {best_model}")
        print(f"   Score: {results[best_model]['quality_score']}/100")

def main():
    """Point d'entrée principal du module d'analyse audio."""

    parser = argparse.ArgumentParser(
        description="Summora - Module d'Analyse Audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Exemples d'usage:
    python main_visual.py audio.mp3                           # Analyse basique
    python main_visual.py audio.wav --plots                   # Avec visualisations
    python main_visual.py audio.mp3 --model-comparison        # Comparaison modèles
    python main_visual.py audio.mp3 --output analysis.json    # Sortie personnalisée
    python main_visual.py audio.mp3 --no-plots --quiet        # Minimal output
        """
    )

    # Arguments obligatoires
    parser.add_argument(
        "audio_file"
        ,type=str
        ,help="Fichier audio à analyser"
    )

    # Options d'analyse
    parser.add_argument(
        "--plots"
        ,action="store_true"
        ,help="Génère les visualisations (spectrogrammes, etc.)"
    )

    parser.add_argument(
        "--no-plots"
        ,action="store_true"
        ,help="Désactive les visualisations"
    )

    parser.add_argument(
        "--model-comparison"
        ,action="store_true"
        ,help="Compare différents modèles d'analyse"
    )

    # Configuration analyse
    parser.add_argument(
        "--silence-threshold"
        ,type=float
        ,default=15.0
        ,help="Seuil de détection silence (défaut: 15.0)"
    )

    parser.add_argument(
        "--min-duration"
        ,type=float
        ,default=60.0
        ,help="Durée minimum meeting en secondes (défaut: 60.0)"
    )

    # Options de sauvegarde
    parser.add_argument(
        "--output", "-o"
        ,type=str
        ,help="Fichier de sortie JSON"
    )

    parser.add_argument(
        "--no-save"
        ,action="store_true"
        ,help="Ne sauvegarde pas les résultats"
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

    # Utilitaires
    parser.add_argument(
        "--list-formats"
        ,action="store_true"
        ,help="Liste les formats audio supportés"
    )

    args = parser.parse_args()

    # Vérification des modules Summora
    if not SUMMORA_AVAILABLE:
        print("❌ Modules Summora non disponibles")
        print("💡 Vérifiez l'installation et les paths")
        return 1

    # Gestion des options utilitaires
    if args.list_formats:
        formats = get_supported_formats()
        print("📁 Formats audio supportés par Summora:")
        for fmt in sorted(formats):
            print(f"   • {fmt}")
        return 0

    # Setup logging
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)

    try:
        # Validation du fichier audio
        audio_path = Path(args.audio_file)
        if not validate_audio_path(audio_path):
            print(f"❌ Fichier audio invalide: {audio_path}")
            return 1

        # Affichage info de démarrage (sauf mode quiet)
        if not args.quiet:
            print("🎵 SUMMORA - MODULE ANALYSE AUDIO")
            print(f"📁 Fichier: {audio_path.name}")
            print(f"🎛️ Seuil silence: {args.silence_threshold}%")
            if args.plots and not args.no_plots:
                print("📊 Visualisations: Activées")
            print("-" * 50)

        # Configuration analyse
        generate_plots = args.plots and not args.no_plots

        # Analyse audio
        if not args.quiet:
            logger.info(f"🎵 Démarrage analyse audio...")

        analysis_result = analyze_meeting_audio_file(
            audio_path
            ,generate_plots=generate_plots
            ,silence_percentile=args.silence_threshold
            ,min_meeting_duration=args.min_duration
        )

        # Vérification des erreurs
        if "error" in analysis_result:
            print(f"❌ Erreur analyse: {analysis_result['error']}")
            if analysis_result.get('suggestion'):
                print(f"💡 Suggestion: {analysis_result['suggestion']}")
            return 1

        # Comparaison modèles si demandée
        if args.model_comparison:
            compare_audio_models(audio_path)

        # Sauvegarde si demandée
        if not args.no_save:
            saved_path = save_analysis_results(analysis_result, audio_path, args.output)
            if not args.quiet:
                logger.info(f"💾 Résultats sauvegardés: {saved_path}")

        # Affichage des résultats
        if not args.quiet:
            print_analysis_summary(analysis_result, audio_path)
        else:
            # Mode quiet : juste le score principal
            quality_score = analysis_result.get('meeting_quality_score', 0)
            print(f"{quality_score}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Analyse interrompue par l'utilisateur")
        return 1
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
