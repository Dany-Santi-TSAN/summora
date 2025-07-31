#!/usr/bin/env python3
"""
Summora - Module de Transcription
Transcription audio vers texte avec Whisper optimisé meetings
Backbone: Medium (défaut) | Fallback: Small
Usage: python main_transcribe.py audio.mp3 --model medium
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import sys

# Setup path pour imports Summora
sys.path.append(str(Path(__file__).parent.parent))

# Imports Summora
from src.core.transcriber import transcribe_meeting_audio
from src.core.utils import validate_audio_path, get_supported_formats

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

def get_optimal_model(audio_path: Path, requested_model: str) -> str:
    """
    Détermine le modèle optimal selon la taille du fichier et la demande.
    Backbone: Medium | Fallback: Small
    """
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    # Si un modèle spécifique est demandé, on le respecte
    if requested_model in ["small", "medium", "large"]:
        return requested_model

    # Sélection automatique selon la taille
    if file_size_mb > 50:  # > 50MB
        recommended = "medium"
        logger = logging.getLogger(__name__)
        logger.info(f"📊 Fichier volumineux ({file_size_mb:.1f}MB) → Modèle Medium recommandé")
    elif file_size_mb > 20:  # 20-50MB
        recommended = "medium"
        logger = logging.getLogger(__name__)
        logger.info(f"📊 Fichier moyen ({file_size_mb:.1f}MB) → Modèle Medium optimal")
    else:  # < 20MB
        recommended = "small"
        logger = logging.getLogger(__name__)
        logger.info(f"📊 Fichier léger ({file_size_mb:.1f}MB) → Modèle Small suffisant")

    return recommended

def transcribe_with_fallback(audio_path: Path, preferred_model: str, language: str, temperature: float) -> dict:
    """
    Transcription avec fallback automatique en cas d'erreur.
    Ordre: Medium → Small → Erreur
    """
    logger = logging.getLogger(__name__)

    models_to_try = []

    # Construction de la liste des modèles à tester
    if preferred_model == "large":
        models_to_try = ["large", "medium", "small"]
    elif preferred_model == "medium":
        models_to_try = ["medium", "small"]
    else:  # small ou auto
        models_to_try = ["small"]

    last_error = None

    for model in models_to_try:
        try:
            logger.info(f"🎤 Tentative transcription avec modèle '{model}'...")

            result = transcribe_meeting_audio(
                audio_path,
                model_size=model,
                language=language,
                temperature=temperature
            )

            if "error" not in result:
                if model != preferred_model:
                    logger.warning(f"⚠️ Fallback vers modèle '{model}' (échec '{preferred_model}')")
                else:
                    logger.info(f"✅ Transcription réussie avec modèle '{model}'")

                # Ajout info modèle utilisé
                result["model_used"] = model
                result["was_fallback"] = (model != preferred_model)
                return result
            else:
                last_error = result
                logger.warning(f"❌ Modèle '{model}' échoué: {result.get('message', 'Erreur inconnue')}")

        except Exception as e:
            last_error = {"error": "exception", "message": str(e)}
            logger.error(f"❌ Exception modèle '{model}': {str(e)}")

    # Tous les modèles ont échoué
    logger.error("❌ Tous les modèles de transcription ont échoué")
    return last_error or {"error": "all_models_failed", "message": "Aucun modèle n'a pu traiter ce fichier"}

def save_transcription(transcription_result: dict, audio_path: Path, model: str,
                      processing_time: float, language: str, output_file: str = None) -> str:
    """
    Sauvegarde la transcription dans un fichier texte structuré.
    """
    if output_file:
        save_path = Path(output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path('output/transcriptions') / (f"transcription_{audio_path.stem}_{model}_{timestamp}.txt")

    # Créer le répertoire si nécessaire
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Métadonnées enrichies
    metadata = {
        "file_name": audio_path.name,
        "file_path": str(audio_path.absolute()),
        "file_size_mb": round(audio_path.stat().st_size / (1024 * 1024), 2),
        "model": model,
        "model_used": transcription_result.get('model_used', model),
        "was_fallback": transcription_result.get('was_fallback', False),
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

    # Écriture du fichier avec header amélioré
    with open(save_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("SUMMORA - TRANSCRIPTION MEETING\n")
        f.write("=" * 70 + "\n\n")

        # Informations fichier
        f.write("📁 INFORMATIONS FICHIER\n")
        f.write("-" * 30 + "\n")
        f.write(f"Fichier source    : {metadata['file_name']}\n")
        f.write(f"Taille fichier    : {metadata['file_size_mb']} MB\n")
        f.write(f"Durée audio       : {metadata['duration']}\n")
        f.write(f"Généré le         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration transcription
        f.write("🤖 CONFIGURATION TRANSCRIPTION\n")
        f.write("-" * 35 + "\n")
        f.write(f"Modèle demandé    : {metadata['model']}\n")
        f.write(f"Modèle utilisé    : {metadata['model_used']}")
        if metadata['was_fallback']:
            f.write(" ⚠️ (fallback)")
        f.write("\n")
        f.write(f"Langue            : {metadata['language']}\n")
        f.write(f"Temps traitement  : {metadata['processing_time_formatted']}\n\n")

        # Métriques qualité
        f.write("📊 MÉTRIQUES QUALITÉ\n")
        f.write("-" * 25 + "\n")
        f.write(f"Nombre de mots    : {metadata['word_count']:,}\n")
        f.write(f"Débit parole      : {metadata['speaking_rate']:.1f} mots/min\n")
        f.write(f"Confiance globale : {metadata['confidence']:.3f}\n")
        f.write(f"Grade confiance   : {metadata['confidence_grade']}\n")
        f.write(f"Densité meeting   : {metadata['meeting_density']:.1f}%\n")

        # Recommandations selon la qualité
        if metadata['confidence'] >= 0.8:
            f.write("✅ Excellente qualité de transcription\n")
        elif metadata['confidence'] >= 0.7:
            f.write("👍 Bonne qualité de transcription\n")
        elif metadata['confidence'] >= 0.6:
            f.write("⚠️ Qualité acceptable - vérifier l'audio\n")
        else:
            f.write("❌ Qualité faible - améliorer l'enregistrement\n")
        f.write("\n")

        # Contenu meeting si disponible
        meeting_content = transcription_result.get('meeting_content', {})
        if meeting_content:
            f.write("🎯 ANALYSE CONTENU MEETING\n")
            f.write("-" * 30 + "\n")
            keywords = meeting_content.get('keyword_counts', {})
            f.write(f"Actions détectées : {keywords.get('action', 0)}\n")
            f.write(f"Décisions         : {keywords.get('decision', 0)}\n")
            f.write(f"Questions         : {keywords.get('question', 0)}\n")
            f.write(f"Planning          : {keywords.get('planning', 0)}\n")
            f.write(f"Structure meeting : {'✅' if meeting_content.get('has_structure', False) else '❌'}\n\n")

        # Transcription complète
        f.write("📝 TRANSCRIPTION COMPLÈTE\n")
        f.write("=" * 70 + "\n\n")
        f.write(transcription_result.get('text', ''))

        # Footer
        f.write(f"\n\n{'=' * 70}\n")
        f.write("Généré par Summora V3 - Speech In, Sense Out\n")
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
            "file_size_mb": round(audio_path.stat().st_size / (1024 * 1024), 2),
            "duration": transcription_result.get('duration_formatted', 'N/A')
        },
        "transcription": {
            "model_requested": model,
            "model_used": transcription_result.get('model_used', model),
            "was_fallback": transcription_result.get('was_fallback', False),
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
        },
        "summora_version": "3.0"
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return str(metadata_path)

def print_summary(transcription_result: dict, audio_path: Path, model: str,
                 processing_time: float, show_preview: bool = True, show_full: bool = False):
    """Affiche un résumé des résultats de transcription."""
    print("\n" + "="*60)
    print("📝 RÉSUMÉ TRANSCRIPTION SUMMORA V3")
    print("="*60)

    # Informations générales
    confidence = transcription_result.get('meeting_confidence', {})
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    print(f"📁 Fichier        : {audio_path.name} ({file_size_mb:.1f}MB)")
    print(f"🤖 Modèle demandé : {model}")

    model_used = transcription_result.get('model_used', model)
    was_fallback = transcription_result.get('was_fallback', False)
    print(f"🤖 Modèle utilisé : {model_used}" + (" ⚠️ (fallback)" if was_fallback else ""))

    print(f"⏱️  Temps         : {format_transcription_time(processing_time)}")
    print(f"📊 Mots           : {transcription_result.get('word_count', 0):,}")
    print(f"🎯 Confiance      : {confidence.get('meeting_confidence', 0):.3f} ({confidence.get('confidence_grade', 'N/A')})")
    print(f"💬 Débit          : {transcription_result.get('speaking_rate', 0):.1f} mots/min")

    # Métriques meeting
    meeting_content = transcription_result.get('meeting_content', {})
    if meeting_content:
        keywords = meeting_content.get('keyword_counts', {})
        print(f"🔥 Densité meeting: {meeting_content.get('meeting_density', 0):.1f}%")
        print(f"⚡ Actions        : {keywords.get('action', 0)}")
        print(f"✅ Décisions      : {keywords.get('decision', 0)}")
        print(f"📋 Structure      : {'✅' if meeting_content.get('has_structure', False) else '❌'}")

    # Qualité globale
    conf_score = confidence.get('meeting_confidence', 0)
    if conf_score >= 0.8:
        print("🏆 Qualité: Excellente")
    elif conf_score >= 0.7:
        print("👍 Qualité: Bonne")
    elif conf_score >= 0.6:
        print("⚠️ Qualité: Acceptable")
    else:
        print("❌ Qualité: Faible")

    # Aperçu du texte
    if show_full:
        print(f"\n📖 TRANSCRIPTION COMPLÈTE:")
        print("-" * 60)
        print(transcription_result.get('text', ''))
    elif show_preview:
        print(f"\n📖 Aperçu:")
        print(f"'{transcription_result.get('preview', '')}'")

    print("="*60)

def main():
    """
    Point d'entrée principal du module de transcription.
    Backbone: Medium | Fallback: Small
    """
    parser = argparse.ArgumentParser(
        description="Summora V3 - Module de Transcription (Backbone: Medium)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Exemples d'usage:
    python main_transcribe.py audio.mp3                     # Auto (Medium/Small selon taille)
    python main_transcribe.py audio.wav --model medium      # Force Medium
    python main_transcribe.py audio.mp3 --model large       # Force Large (avec fallback)
    python main_transcribe.py audio.mp3 --full-text         # Affichage complet
    python main_transcribe.py audio.mp3 --output result.txt # Fichier personnalisé
    python main_transcribe.py audio.mp3 --no-save --quiet   # Juste la transcription

    Modèles disponibles: small, medium (défaut), large
    Note: tiny supprimé (qualité insuffisante pour meetings)
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
        ,choices=["small", "medium", "large", "auto"]
        ,default="auto"
        ,help="Modèle Whisper (défaut: auto=medium/small selon taille)"
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

        # Détermination du modèle optimal
        if args.model == "auto":
            optimal_model = get_optimal_model(audio_path, "auto")
        else:
            optimal_model = args.model

        # Affichage info de démarrage (sauf mode quiet)
        if not args.quiet:
            print("🎤 SUMMORA V3 - MODULE TRANSCRIPTION")
            print(f"📁 Fichier: {audio_path.name}")
            print(f"🤖 Modèle: {optimal_model}" + (" (auto-sélectionné)" if args.model == "auto" else ""))
            print(f"🌍 Langue: {args.language}")
            if args.temperature > 0:
                print(f"🌡️  Température: {args.temperature}")
            print("-" * 50)

        # Transcription avec fallback
        start_time = datetime.now()

        if not args.quiet:
            logger.info(f"🎤 Démarrage transcription avec fallback automatique...")

        transcription_result = transcribe_with_fallback(
            audio_path, optimal_model, args.language, args.temperature
        )

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Vérification des erreurs
        if "error" in transcription_result:
            print(f"❌ Erreur transcription: {transcription_result.get('message', 'Erreur inconnue')}")
            return 1

        # Sauvegarde si demandée
        saved_files = []
        if not args.no_save:
            # Sauvegarde texte
            saved_path = save_transcription(
                transcription_result, audio_path, optimal_model,
                processing_time, args.language, args.output
            )
            saved_files.append(saved_path)

            # Sauvegarde JSON si demandée
            if args.json_metadata:
                json_path = save_metadata_json(
                    transcription_result, audio_path, optimal_model,
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
                transcription_result, audio_path, optimal_model, processing_time,
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
