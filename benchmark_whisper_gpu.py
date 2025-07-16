#!/usr/bin/env python3
"""
Benchmark Whisper Medium vs Large sur GPU Tesla T4
Script optimisé pour les 2h30 de session AWS
"""
import time
import json
from pathlib import Path
from datetime import datetime
import logging

# Import Summora
from src.core.transcriber import MeetingTranscriber, MeetingTranscriptionConfig
from src.core.metrics.evaluator import SummoraEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_whisper_models():
    """
    Benchmark Whisper Medium vs Large sur fichiers audio test
    """
    print("🚀 Benchmark Whisper Medium vs Large - GPU Tesla T4")
    print("=" * 60)

    # Configuration modèles à tester
    models = ["medium", "large"]
    audio_files = [
        "data/audio-om-mercato-test.mp3",
        "data/test-reunion.mp3"
    ]

    results = {}

    for model_size in models:
        print(f"\n📊 Test modèle Whisper {model_size.upper()}")
        print("-" * 40)

        # Configuration Whisper
        config = MeetingTranscriptionConfig(
            model_size=model_size,
            language="fr",
            temperature=0.0,
            word_timestamps=True
        )

        # Création transcripteur
        transcriber = MeetingTranscriber(config)

        model_results = {}

        for audio_file in audio_files:
            if not Path(audio_file).exists():
                logger.warning(f"⚠️ Fichier non trouvé: {audio_file}")
                continue

            print(f"🎤 Transcription: {Path(audio_file).name}")

            # Mesure temps et transcription
            start_time = time.time()
            result = transcriber.transcribe_meeting(audio_file)
            end_time = time.time()

            # Extraction métriques
            processing_time = end_time - start_time

            if "error" not in result:
                file_results = {
                    "audio_file": audio_file,
                    "processing_time": processing_time,
                    "duration": result["duration"],
                    "word_count": result["word_count"],
                    "meeting_confidence": result["meeting_confidence"]["meeting_confidence"],
                    "confidence_grade": result["meeting_confidence"]["confidence_grade"],
                    "speaking_rate": result["meeting_confidence"]["speaking_rate"],
                    "meeting_density": result["meeting_content"]["meeting_density"],
                    "text_preview": result["preview"][:100] + "...",
                    "realtime_factor": processing_time / result["duration"]
                }

                # Affichage résultats
                print(f"  ✅ Durée: {result['duration_formatted']}")
                print(f"  ⏱️ Temps: {processing_time:.1f}s")
                print(f"  🎯 Confiance: {result['meeting_confidence']['meeting_confidence']:.3f}")
                print(f"  🏆 Grade: {result['meeting_confidence']['confidence_grade']}")
                print(f"  ⚡ RTF: {processing_time / result['duration']:.2f}x")
                print(f"  💬 Débit: {result['meeting_confidence']['speaking_rate']:.1f} wpm")

            else:
                file_results = {
                    "audio_file": audio_file,
                    "error": result["error"],
                    "processing_time": processing_time
                }
                print(f"  ❌ Erreur: {result['error']}")

            model_results[Path(audio_file).name] = file_results

        results[model_size] = model_results

    # Comparaison finale
    print("\n📈 COMPARAISON FINALE")
    print("=" * 60)

    for audio_name in [Path(f).name for f in audio_files]:
        if audio_name in results.get("medium", {}) and audio_name in results.get("large", {}):
            medium_res = results["medium"][audio_name]
            large_res = results["large"][audio_name]

            if "error" not in medium_res and "error" not in large_res:
                print(f"\n🎤 {audio_name}:")
                print(f"  Medium: {medium_res['processing_time']:.1f}s | Confiance: {medium_res['meeting_confidence']:.3f}")
                print(f"  Large:  {large_res['processing_time']:.1f}s | Confiance: {large_res['meeting_confidence']:.3f}")

                # Recommandation
                if large_res['meeting_confidence'] - medium_res['meeting_confidence'] > 0.05:
                    print(f"  🎯 Recommandation: Large (+{large_res['meeting_confidence'] - medium_res['meeting_confidence']:.3f} confiance)")
                else:
                    print(f"  ⚡ Recommandation: Medium (gain temps minimal confiance)")

    # Sauvegarde résultats
    output_file = f"output/benchmark_whisper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("output").mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Résultats sauvés: {output_file}")

    return results

if __name__ == "__main__":
    benchmark_whisper_models()
