#!/usr/bin/env python3
"""
Summora - Version test avec TEXTE COMPLET
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from src.core.utils import validate_audio_path
from src.core.transcriber import transcribe_meeting_audio
from src.meeting.extractor import extract_meeting_content

def test_transcription_with_full_text(audio_path: str, save_to_file: bool = True):
    """Test avec affichage du texte complet."""
    print(f"🎤 Test transcription COMPLÈTE: {Path(audio_path).name}")
    print("-" * 50)

    # Validation fichier
    validated_path = validate_audio_path(audio_path)
    if not validated_path:
        print(f"❌ Fichier audio invalide: {audio_path}")
        return

    # Transcription
    print("🔄 Transcription en cours...")
    result = transcribe_meeting_audio(validated_path, model_size="tiny")

    if "error" in result:
        print(f"❌ Erreur: {result['message']}")
        return

    # Métadonnées
    text = result.get("text", "")
    word_count = result.get("word_count", 0)
    duration = result.get("duration_formatted", "N/A")
    confidence = result.get("meeting_confidence", {}).get("meeting_confidence", 0)

    print(f"✅ Transcription réussie!")
    print(f"   📝 Mots: {word_count}")
    print(f"   ⏱️ Durée: {duration}")
    print(f"   🎯 Confiance: {confidence:.3f}")
    print()

    # === TEXTE COMPLET ===
    print("=" * 80)
    print("📄 TRANSCRIPTION COMPLÈTE")
    print("=" * 80)
    print(text)
    print("=" * 80)
    print(f"FIN - {len(text)} caractères, {word_count} mots")
    print("=" * 80)

    # Sauvegarde optionnelle
    if save_to_file:
        output_file = f"transcription_{Path(audio_path).stem}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== TRANSCRIPTION SUMMORA ===\n")
            f.write(f"Fichier source: {audio_path}\n")
            f.write(f"Mots transcrits: {word_count}\n")
            f.write(f"Durée: {duration}\n")
            f.write(f"Confiance: {confidence:.3f}\n")
            f.write(f"Modèle: tiny\n")
            f.write("=" * 50 + "\n\n")
            f.write(text)
            f.write(f"\n\n=== FIN TRANSCRIPTION ===\n")

        print(f"💾 Transcription sauvée dans: {output_file}")

    return result

def main():
    """Point d'entrée pour voir le texte complet."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main_test_full.py <audio_file> [--no-save]")
        print()
        print("Options:")
        print("  --no-save    Ne pas sauvegarder dans un fichier")
        print()
        print("Exemples:")
        print("  python main_test_full.py audio-om-mercato-test.mp3")
        print("  python main_test_full.py audio-om-mercato-test.mp3 --no-save")
        return

    audio_path = sys.argv[1]
    save_file = "--no-save" not in sys.argv

    if not Path(audio_path).exists():
        print(f"❌ Fichier non trouvé: {audio_path}")
        return

    try:
        test_transcription_with_full_text(audio_path, save_to_file=save_file)

    except KeyboardInterrupt:
        print("\n❌ Test interrompu")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")

if __name__ == "__main__":
    main()
