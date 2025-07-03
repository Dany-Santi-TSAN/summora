"""
Test de l'évaluateur Summora sur fichiers de transcription réels.
Benchmark Primary + Experimental sur données Whisper.
"""

import os
from pathlib import Path
from src.core.metrics.evaluator import create_summora_evaluator

def load_transcription_file(filepath: str) -> str:
    """
    Charge un fichier de transcription.

    Args:
        filepath: Chemin vers le fichier

    Retourne:
        str: Contenu du fichier nettoyé
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content
    except Exception as e:
        print(f"❌ Erreur lecture {filepath}: {e}")
        return ""

def test_reunion_files():
    """Test sur les fichiers test-reunion (tiny vs small vs original)."""

    print("🎯 TEST RÉUNION - Comparaison modèles Whisper")
    print("=" * 60)

    # Chemins fichiers
    base_path = Path("output/transcriptions")
    files = {
        "original": base_path / "transcription_test-reunion.txt",
        "tiny": base_path / "transcription_test-reunion_tiny_20250618_184255.txt",
        "small": base_path / "transcription_test-reunion_small_20250618_190856.txt"
    }

    # Chargement transcriptions
    transcriptions = {}
    for model, filepath in files.items():
        if filepath.exists():
            content = load_transcription_file(filepath)
            if content:
                transcriptions[model] = content
                print(f"✅ {model}: {len(content)} caractères")
            else:
                print(f"❌ {model}: fichier vide")
        else:
            print(f"⚠️ {model}: fichier non trouvé")

    if len(transcriptions) < 2:
        print("❌ Pas assez de fichiers pour comparaison")
        return

    # Évaluateur Summora
    evaluator = create_summora_evaluator()

    # On prend 'original' comme référence (ou le plus long)
    reference_model = max(transcriptions.keys(), key=lambda k: len(transcriptions[k]))
    reference_text = transcriptions[reference_model]

    print(f"\n📋 Référence: {reference_model} ({len(reference_text)} caractères)")

    # Comparaison des autres vs référence
    comparison_transcriptions = {
        model: text for model, text in transcriptions.items()
        if model != reference_model
    }

    if not comparison_transcriptions:
        print("❌ Pas d'autres modèles à comparer")
        return

    # Évaluation complète
    print(f"\n🔬 Évaluation complète (Primary + Experimental)")
    reports = evaluator.compare_models(
        reference_text, comparison_transcriptions, include_experimental=True
    )

    # Affichage résultats
    print(f"\n📊 RÉSULTATS COMPARAISON")
    print("-" * 40)
    for model, report in reports.items():
        print(f"{model.upper():8} | Grade: {report.overall_grade:2} | "
              f"Primary: {report.primary_composite_score:.3f} | "
              f"Exp: {report.experimental_composite_score:.3f}")

        # Détail des métriques
        print(f"         | CER: {report.cer_result.score:.3f} | "
              f"WER: {report.wer_result.score:.3f} | "
              f"BERT: {report.bert_result.score:.3f}")
        print(f"         | PER: {report.per_result.score:.3f} | "
              f"SemDist: {report.semdist_result.score:.3f}")
        print()

    # Meilleur modèle
    best_model, best_score = evaluator.get_best_model(reports)
    print(f"🏆 Meilleur modèle: {best_model.upper()} (score: {best_score:.3f})")

    # Recommandations du meilleur
    best_report = reports[best_model]
    print(f"\n💡 Recommandations pour {best_model}:")
    for rec in best_report.recommendations:
        print(f"   • {rec}")

def test_om_mercato_files():
    """Test sur les fichiers OM mercato."""

    print("\n🎯 TEST OM MERCATO - Comparaison modèles")
    print("=" * 60)

    # Chemins fichiers OM
    base_path = Path("output/transcriptions")
    files = {
        "original": base_path / "transcription_audio-om-mercato-test.txt",
        "small": base_path / "transcription_audio-om-mercato-test_small_20250618_210617.txt"
    }

    # Chargement
    transcriptions = {}
    for model, filepath in files.items():
        if filepath.exists():
            content = load_transcription_file(filepath)
            if content:
                transcriptions[model] = content
                print(f"✅ {model}: {len(content)} caractères")

    if len(transcriptions) < 2:
        print("❌ Pas assez de fichiers OM pour comparaison")
        return

    # Même processus que réunion
    evaluator = create_summora_evaluator()
    reference_model = "original"  # On prend original comme ref
    reference_text = transcriptions[reference_model]

    comparison_transcriptions = {
        model: text for model, text in transcriptions.items()
        if model != reference_model
    }

    # Évaluation
    reports = evaluator.compare_models(
        reference_text, comparison_transcriptions, include_experimental=True
    )

    # Affichage simple pour OM
    print(f"\n📊 RÉSULTATS OM MERCATO")
    print("-" * 30)
    for model, report in reports.items():
        print(f"{model}: Grade {report.overall_grade} | "
              f"Primary: {report.primary_composite_score:.3f}")

def test_cross_comparison():
    """Test croisé : comparer réunion vs OM mercato."""

    print("\n🎯 TEST CROISÉ - Réunion vs OM Mercato")
    print("=" * 60)

    base_path = Path("output/transcriptions")

    # Chargement d'un échantillon de chaque
    reunion_file = base_path / "transcription_test-reunion.txt"
    om_file = base_path / "transcription_audio-om-mercato-test.txt"

    if not reunion_file.exists() or not om_file.exists():
        print("❌ Fichiers manquants pour test croisé")
        return

    reunion_text = load_transcription_file(reunion_file)
    om_text = load_transcription_file(om_file)

    evaluator = create_summora_evaluator()

    # Test : à quel point les transcriptions sont différentes ?
    report = evaluator.evaluate_complete(reunion_text, om_text, include_experimental=True)

    print(f"📊 Différence Réunion vs OM Mercato:")
    print(f"   Grade: {report.overall_grade}")
    print(f"   Primary score: {report.primary_composite_score:.3f}")
    print(f"   CER: {report.cer_result.score:.3f}")
    print(f"   WER: {report.wer_result.score:.3f}")
    print(f"   SemDist: {report.semdist_result.score:.3f}")

if __name__ == "__main__":
    """Lancement des tests sur fichiers."""

    print("🚀 SUMMORA EVALUATOR - TEST FICHIERS RÉELS")
    print("=" * 80)

    # Vérification dossier
    transcription_dir = Path("output/transcriptions")
    if not transcription_dir.exists():
        print(f"❌ Dossier {transcription_dir} non trouvé")
        print("💡 Lancez depuis la racine du projet summora/")
        exit(1)

    # Tests séquentiels
    test_reunion_files()
    test_om_mercato_files()
    test_cross_comparison()

    print("\n✅ Tests sur fichiers terminés !")
