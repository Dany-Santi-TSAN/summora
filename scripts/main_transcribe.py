#!/usr/bin/env python3
"""
Summora - Module de Transcription
Transcription audio vers texte avec Whisper optimisé meetings
Usage: python main_transcribe.py audio.mp3 --model small
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import sys
sys.path.append('..')

# Imports Summora
from src.core.transcriber import transcribe_meeting_audio
from src.core.utils import validate_audio_path, get_supported_formats, format_duration

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

def format_transcription_time(seconds: float) -> str:
    """Formate le temps de transcription de manière lisible."""
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        return f"{seconds:.1f}s"

def save_transcription(transcription_result: dict, audio_path: Path, model: str,
                      processing_time: float, language: str, output_file: str = None) -> str:
    """
    Sauvegarde la transcription dans un fichier texte structuré.

    Args:
        transcription_result: Résultat de la transcription
        audio_path: Chemin du fichier audio source
        model: Modèle Whisper utilisé
        processing_time: Temps de traitement
        language: Langue de transcription
        output_file: Nom du fichier de sortie (optionnel)

    Returns:
        str: Chemin du fichier sauvegardé
    """
    if output_file:
        save_path = Path(output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path('output/transcriptions') / (f"transcription_{audio_path.stem}_{model}_{timestamp}.txt")

    # Métadonnées
    metadata = {
        "file_name": audio_path.name,
        "file_path": str(audio_path.absolute()),
        "model": model,
        "language": language,
        "processing_time": processing_time,
        "processing_time_formatted": format_transcription_time(processing_time),
        "timestamp": datetime.now().isoformat(),
        "word_count": transcription_result.get('word_count', 0),
        "duration": transcription_result.get('duration_formatted', 'N/A'),
        "confidence": transcription_result.get('meeting_confidence', {}).get('meeting_confidence', 0),
        "confidence_grade": transcription_result.get('meeting_confidence', {}).get('confidence_grade', 'N/A'),
        "meeting_density": transcription_result.get('meeting_content', {}).get('meeting_density', 0),
        "speaking_rate": transcription_result.get('speaking_rate', 0)
    }

    # Écriture du fichier
    with open(save_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("SUMMORA - TRANSCRIPTION MEETING\n")
        f.write("=" * 70 + "\n\n")

        # Métadonnées principales
        f.write("📁 INFORMATIONS FICHIER\n")
        f.write("-" * 30 + "\n")
        f.write(f"Fichier source    : {metadata['file_name']}\n")
        f.write(f"Chemin complet    : {metadata['file_path']}\n")
        f.write(f"Durée audio       : {metadata['duration']}\n")
        f.write(f"Généré le         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Métadonnées transcription
        f.write("🤖 CONFIGURATION TRANSCRIPTION\n")
        f.write("-" * 35 + "\n")
        f.write(f"Modèle Whisper    : {metadata['model']}\n")
        f.write(f"Langue            : {metadata['language']}\n")
        f.write(f"Temps traitement  : {metadata['processing_time_formatted']}\n\n")

        # Métriques qualité
        f.write("📊 MÉTRIQUES QUALITÉ\n")
        f.write("-" * 25 + "\n")
        f.write(f"Nombre de mots    : {metadata['word_count']}\n")
        f.write(f"Débit parole      : {metadata['speaking_rate']:.1f} mots/min\n")
        f.write(f"Confiance         : {metadata['confidence']:.3f}\n")
        f.write(f"Grade confiance   : {metadata['confidence_grade']}\n")
        f.write(f"Densité meeting   : {metadata['meeting_density']:.1f}%\n\n")

        # Contenu meeting si disponible
        meeting_content = transcription_result.get('meeting_content', {})
        if meeting_content:
            f.write("🎯 ANALYSE CONTENU MEETING\n")
            f.write("-" * 30 + "\n")
            keywords = meeting_content.get('keyword_counts', {})
            f.write(f"Actions détectées : {keywords.get('action', 0)}\n")
            f.write(f"Décisions         : {keywords.get('decision', 0)}\n")
            f.write(f"Questions         : {keywords.get('question', 0)}\n")
            f.write(f"Planning          : {keywords.get('planning', 0)}\n\n")

        # Transcription complète
        f.write("📝 TRANSCRIPTION COMPLÈTE\n")
        f.write("=" * 70 + "\n\n")
        f.write(transcription_result.get('text', ''))

        # Footer
        f.write(f"\n\n{'=' * 70}\n")
        f.write("Généré par Summora - Speech In, Sense Out\n")
        f.write(f"{'=' * 70}\n")

    return str(save_path)

def save_metadata_json(transcription_result: dict, audio_path: Path, model: str,
                      processing_time: float, language: str, save_path: str):
    """Sauvegarde les métadonnées en JSON pour usage programmatique."""
    metadata_path = Path(save_path).with_suffix('.json')

    metadata = {
        "source": {
            "file_name": audio_path.name,
            "file_path": str(audio_path.absolute()),
            "duration": transcription_result.get('duration_formatted', 'N/A')
        },
        "transcription": {
            "model": model,
            "language": language,
            "processing_time": processing_time,
            "processing_time_formatted": format_transcription_time(processing_time),
            "timestamp": datetime.now().isoformat()
        },
        "metrics": {
            "word_count": transcription_result.get('word_count', 0),
            "confidence": transcription_result.get('meeting_confidence', {}).get('meeting_confidence', 0),
            "confidence_grade": transcription_result.get('meeting_confidence', {}).get('confidence_grade', 'N/A'),
            "meeting_density": transcription_result.get('meeting_content', {}).get('meeting_density', 0),
            "speaking_rate": transcription_result.get('speaking_rate', 0)
        },
        "content": {
            "text": transcription_result.get('text', ''),
            "preview": transcription_result.get('preview', ''),
            "meeting_keywords": transcription_result.get('meeting_content', {}).get('keyword_counts', {})
        }
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return str(metadata_path)

def print_summary(transcription_result: dict, audio_path: Path, model: str,
                 processing_time: float, show_preview: bool = True, show_full: bool = False):
    """Affiche un résumé des résultats de transcription."""
    print("\n" + "="*50)
    print("📝 RÉSUMÉ TRANSCRIPTION SUMMORA")
    print("="*50)

    # Infos générales
    confidence = transcription_result.get('meeting_confidence', {})
    print(f"📁 Fichier        : {audio_path.name}")
    print(f"🤖 Modèle         : {model}")
    print(f"⏱️  Temps         : {format_transcription_time(processing_time)}")
    print(f"📊 Mots           : {transcription_result.get('word_count', 0)}")
    print(f"🎯 Confiance      : {confidence.get('meeting_confidence', 0):.3f} ({confidence.get('confidence_grade', 'N/A')})")
    print(f"💬 Débit          : {transcription_result.get('speaking_rate', 0):.1f} mots/min")

    # Métriques meeting
    meeting_content = transcription_result.get('meeting_content', {})
    if meeting_content:
        keywords = meeting_content.get('keyword_counts', {})
        print(f"🔥 Densité meeting: {meeting_content.get('meeting_density', 0):.1f}%")
        print(f"⚡ Actions        : {keywords.get('action', 0)}")
        print(f"✅ Décisions      : {keywords.get('decision', 0)}")

    # Aperçu du texte
    if show_full:
        print(f"\n📖 TRANSCRIPTION COMPLÈTE:")
        print("-" * 50)
        print(transcription_result.get('text', ''))
    elif show_preview:
        print(f"\n📖 Aperçu:")
        print(f"'{transcription_result.get('preview', '')}'")

    print("="*50)

def main():
    """
    Point d'entrée principal du module de transcription.
    """
    parser = argparse.ArgumentParser(
        description="Summora - Module de Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Exemples d'usage:
    python main_transcribe.py audio.mp3                     # Transcription basique
    python main_transcribe.py audio.wav --model small       # Modèle spécifique
    python main_transcribe.py audio.mp3 --full-text         # Affichage complet
    python main_transcribe.py audio.mp3 --output result.txt # Fichier personnalisé
    python main_transcribe.py audio.mp3 --no-save --quiet   # Juste la transcription
    """
    )

    # Arguments obligatoires
    parser.add_argument(
        "audio_file"
        ,type=str
        ,help="Fichier audio à transcrire"
    )

    # Configuration transcription
    parser.add_argument(
        "--model", "-m"
        ,type=str
        ,choices=["tiny", "base", "small", "medium", "large"]
        ,default="base"
        ,help="Modèle Whisper (défaut: base)"
    )

    parser.add_argument(
        "--language", "-l"
        ,type=str
        ,default="fr"
        ,help="Langue de transcription (défaut: fr)"
    )

    parser.add_argument(
        "--temperature", "-t"
        ,type=float
        ,default=0.0
        ,help="Température Whisper (0.0-1.0, défaut: 0.0)"
    )

    # Options d'affichage
    parser.add_argument(
        "--full-text"
        ,action="store_true"
        ,help="Affiche la transcription complète"
    )

    parser.add_argument(
        "--no-preview"
        ,action="store_true"
        ,help="N'affiche pas l'aperçu de transcription"
    )

    # Options de sauvegarde
    parser.add_argument(
        "--no-save"
        ,action="store_true"
        ,help="Ne sauvegarde pas la transcription"
    )

    parser.add_argument(
        "--output", "-o"
        ,type=str
        ,help="Nom du fichier de sortie"
    )

    parser.add_argument(
        "--json-metadata"
        ,action="store_true"
        ,help="Sauvegarde aussi les métadonnées en JSON"
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
        ,help="Mode silencieux (transcription uniquement)"
    )

    # Utilitaires
    parser.add_argument(
        "--list-formats"
        ,action="store_true"
        ,help="Liste les formats audio supportés"
    )

    args = parser.parse_args()

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
            print("🎤 SUMMORA - MODULE TRANSCRIPTION")
            print(f"📁 Fichier: {audio_path.name}")
            print(f"🤖 Modèle: {args.model}")
            print(f"🌍 Langue: {args.language}")
            if args.temperature > 0:
                print(f"🌡️  Température: {args.temperature}")
            print("-" * 50)

        # Transcription
        start_time = datetime.now()

        if not args.quiet:
            logger.info(f"🎤 Démarrage transcription avec modèle '{args.model}'...")

        transcription_result = transcribe_meeting_audio(
            audio_path,
            model_size=args.model,
            language=args.language,
            temperature=args.temperature
        )

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Vérification des erreurs
        if "error" in transcription_result:
            print(f"❌ Erreur transcription: {transcription_result['message']}")
            return 1

        # Sauvegarde si demandée
        saved_files = []
        if not args.no_save:
            # Sauvegarde texte
            saved_path = save_transcription(
                transcription_result, audio_path, args.model,
                processing_time, args.language, args.output
            )
            saved_files.append(saved_path)

            # Sauvegarde JSON si demandée
            if args.json_metadata:
                json_path = save_metadata_json(
                    transcription_result, audio_path, args.model,
                    processing_time, args.language, saved_path
                )
                saved_files.append(json_path)

        # Affichage des résultats
        if args.quiet:
            # Mode silencieux : juste le texte
            if args.full_text:
                print(transcription_result.get('text', ''))
            else:
                print(transcription_result.get('preview', ''))
        else:
            # Mode normal : résumé complet
            print_summary(
                transcription_result, audio_path, args.model, processing_time,
                show_preview=not args.no_preview,
                show_full=args.full_text
            )

            # Info sauvegarde
            if saved_files:
                print(f"\n💾 Fichiers sauvegardés:")
                for file_path in saved_files:
                    print(f"   • {file_path}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Transcription interrompue par l'utilisateur")
        return 1
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
