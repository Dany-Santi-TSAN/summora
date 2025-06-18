#!/usr/bin/env python3
"""
Summora Test CLI - Tests rapides de transcription meeting
Usage: python main_test.py audio.mp3 --model small --verbose
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Import des modules Summora
from src.core.transcriber import create_meeting_transcriber, transcribe_meeting_audio
from src.core.audio_analyzer import analyze_meeting_audio_file
from src.core.utils import validate_audio_path, get_supported_formats

def setup_logging(verbose: bool = False):
    """Configure le logging selon le niveau de verbosité."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def print_results(transcription_result: dict, audio_analysis: dict = None):
    """Affiche les résultats de manière propre."""
    print("\n" + "="*60)
    print("🎤 RÉSULTATS SUMMORA TEST")
    print("="*60)

    if "error" in transcription_result:
        print(f"❌ Erreur: {transcription_result['message']}")
        return

    # Info générale
    print(f"📝 Mots transcrits: {transcription_result['word_count']}")
    print(f"⏱️  Durée: {transcription_result['duration_formatted']}")
    print(f"🎯 Confiance: {transcription_result['meeting_confidence']['meeting_confidence']:.3f}")
    print(f"📊 Grade: {transcription_result['meeting_confidence']['confidence_grade']}")
    print(f"💬 Débit: {transcription_result['speaking_rate']:.1f} mots/min")

    # Métriques meeting
    content = transcription_result['meeting_content']
    print(f"🔥 Densité meeting: {content['meeting_density']:.1f}%")
    print(f"⚡ Actions détectées: {content['keyword_counts'].get('action', 0)}")
    print(f"✅ Décisions détectées: {content['keyword_counts'].get('decision', 0)}")

    # Aperçu du texte
    print(f"\n📖 Aperçu transcription:")
    print(f"'{transcription_result['preview']}'")

    # Audio analysis si disponible
    if audio_analysis and "error" not in audio_analysis:
        print(f"\n🎵 Analyse audio:")
        print(f"   • Sample rate: {audio_analysis['sample_rate']} Hz")
        print(f"   • Ratio parole: {audio_analysis['speech_ratio']*100:.1f}%")
        print(f"   • Score qualité: {audio_analysis['meeting_quality_score']}/100")
        print(f"   • Grade audio: {audio_analysis['meeting_quality_grade']}")

def main():
    """Point d'entrée principal du test CLI."""
    parser = argparse.ArgumentParser(
        description="Summora Test CLI - Tests rapides transcription meeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
  python main_test.py audio.mp3                    # Test avec modèle base
  python main_test.py audio.wav --model small      # Test avec modèle small
  python main_test.py audio.m4a --model medium -v  # Test verbose avec medium
  python main_test.py audio.mp3 --no-audio         # Transcription uniquement
        """
    )

    # Arguments positionnels
    parser.add_argument(
        "audio_file",
        type=str,
        help="Chemin vers le fichier audio à transcrire"
    )

    # Options de transcription
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Modèle Whisper à utiliser (défaut: base)"
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        default="fr",
        help="Langue de transcription (défaut: fr)"
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Température Whisper (0.0-1.0, défaut: 0.0)"
    )

    # Options d'analyse
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip l'analyse audio (transcription uniquement)"
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Désactive les visualisations audio"
    )

    # Options d'affichage
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux (logs détaillés)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Mode silencieux (résultats uniquement)"
    )

    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Affiche le texte complet (pas seulement l'aperçu)"
    )

    # Utilitaires
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="Liste les formats audio supportés"
    )

    args = parser.parse_args()

    # Gestion des options utilitaires
    if args.list_formats:
        formats = get_supported_formats()
        print("📁 Formats audio supportés:")
        for fmt in sorted(formats):
            print(f"   • {fmt}")
        return 0

    # Configuration logging
    if not args.quiet:
        setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        # Validation du fichier
        audio_path = Path(args.audio_file)
        if not validate_audio_path(audio_path):
            print(f"❌ Fichier audio invalide: {audio_path}")
            return 1

        if not args.quiet:
            print(f"🚀 Démarrage test Summora")
            print(f"📁 Fichier: {audio_path.name}")
            print(f"🤖 Modèle: {args.model}")
            print(f"🌍 Langue: {args.language}")
            print("-" * 50)

        start_time = datetime.now()

        # 1. Transcription
        if not args.quiet:
            logger.info(f"🎤 Transcription avec modèle {args.model}...")

        transcription_result = transcribe_meeting_audio(
            audio_path,
            model_size=args.model,
            language=args.language,
            temperature=args.temperature
        )

        # 2. Analyse audio (optionnelle)
        audio_analysis = None
        if not args.no_audio and "error" not in transcription_result:
            if not args.quiet:
                logger.info("🎵 Analyse des propriétés audio...")

            from src.core.audio_analyzer import MeetingAudioAnalyzer, MeetingAudioConfig

            audio_config = MeetingAudioConfig(generate_plots=not args.no_plots)
            audio_analyzer = MeetingAudioAnalyzer(audio_config)
            audio_analysis = audio_analyzer.analyze_meeting_audio_property(audio_path)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # 3. Affichage des résultats
        if not args.quiet:
            print_results(transcription_result, audio_analysis)
            print(f"\n⏱️  Temps total: {processing_time:.2f}s")
        else:
            # Mode quiet : juste le texte
            if "error" not in transcription_result:
                text = transcription_result['text'] if args.full_text else transcription_result['preview']
                print(text)

        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
        return 1
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
