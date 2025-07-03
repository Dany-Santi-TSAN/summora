"""
Test complet de l'évaluateur Summora modulaire.
Test des 3 classes : Primary + Experimental + Aggregateur.
"""

import pytest

def test_primary_evaluator_only():
    """Test PrimaryMetricsEvaluator seul."""
    from src.core.metrics.evaluator import PrimaryMetricsEvaluator

    evaluator = PrimaryMetricsEvaluator()

    reference = "nous validons le budget planning"
    hypothesis = "nous validons le budget planning"

    # Test Primary uniquement
    results = evaluator.evaluate(reference, hypothesis)

    assert 'cer' in results
    assert 'wer' in results
    assert 'bert' in results
    assert results['cer'].score == pytest.approx(0.0, abs=1e-6)
    assert results['wer'].score == pytest.approx(0.0, abs=1e-6)

    # Test score composite
    composite_score = evaluator.calculate_composite_score(results)
    assert composite_score == pytest.approx(1.0, abs=1e-6)  # Score très élevé pour identique

    print("✅ PrimaryMetricsEvaluator OK")
    print(f"   CER: {results['cer'].score:.3f}")
    print(f"   WER: {results['wer'].score:.3f}")
    print(f"   BERT: {results['bert'].score:.3f}")
    print(f"   Composite: {composite_score:.3f}")

def test_experimental_evaluator_only():
    """Test ExperimentalMetricsEvaluator seul."""
    from src.core.metrics.evaluator import ExperimentalMetricsEvaluator

    evaluator = ExperimentalMetricsEvaluator()

    reference = "nous validons le budget planning"
    hypothesis = "nous validons le budget planning"

    # Test Experimental uniquement
    results = evaluator.evaluate(reference, hypothesis)

    assert 'per' in results
    assert 'semdist' in results
    assert results['per'].score == pytest.approx(0.0, abs=1e-6)
    assert results['semdist'].score == pytest.approx(0.0, abs=1e-6)

    # Test score composite
    composite_score = evaluator.calculate_composite_score(results)
    assert composite_score == pytest.approx(1.0, abs=1e-6)  # Score très élevé pour identique

    print("✅ ExperimentalMetricsEvaluator OK")
    print(f"   PER: {results['per'].score:.3f}")
    print(f"   SemDist: {results['semdist'].score:.3f}")
    print(f"   Composite: {composite_score:.3f}")

def test_summora_aggregator():
    """Test SummoraEvaluator (aggregateur)."""
    from src.core.metrics.evaluator import SummoraEvaluator

    evaluator = SummoraEvaluator()

    reference = "nous validons le budget planning avec deadline"
    hypothesis = "nous validons le budget planing avec échéance"  # Erreurs réalistes

    # Test évaluation complète
    report = evaluator.evaluate_complete(reference, hypothesis, include_experimental=True)

    # Vérifications structure
    assert report.wer_result is not None
    assert report.cer_result is not None
    assert report.bert_result is not None
    assert report.per_result is not None
    assert report.semdist_result is not None

    # Vérifications scores
    assert 0.0 <= report.primary_composite_score <= 1.0
    assert 0.0 <= report.experimental_composite_score <= 1.0
    assert report.overall_grade in ['A+', 'A', 'B', 'C', 'D']

    # Vérifications méta
    assert len(report.recommendations) > 0
    assert len(report.metrics_used) >= 5  # 3 primary + 2 experimental
    assert report.processing_time_total > 0

    print("✅ SummoraEvaluator (Aggregateur) OK")
    print(f"   Primary score: {report.primary_composite_score:.3f}")
    print(f"   Experimental score: {report.experimental_composite_score:.3f}")
    print(f"   Grade: {report.overall_grade}")
    print(f"   Métriques: {report.metrics_used}")

def test_primary_only_mode():
    """Test mode Primary uniquement (sans experimental)."""
    from src.core.metrics.evaluator import SummoraEvaluator

    evaluator = SummoraEvaluator()

    reference = "nous validons le budget"
    hypothesis = "nous validons le budget"

    # Primary seulement
    report = evaluator.evaluate_complete(reference, hypothesis, include_experimental=False)

    # Vérifications Primary
    assert report.wer_result is not None
    assert report.cer_result is not None
    assert report.bert_result is not None

    # Vérifications Experimental désactivées
    assert report.per_result is None
    assert report.semdist_result is None
    assert report.experimental_composite_score == 0.0

    # Métriques utilisées
    assert len(report.metrics_used) == 3  # Seulement primary
    assert 'per' not in report.metrics_used
    assert 'semdist' not in report.metrics_used

    print("✅ Mode Primary uniquement OK")
    print(f"   Métriques: {report.metrics_used}")
    print(f"   Primary score: {report.primary_composite_score:.3f}")

def test_model_comparison_modular():
    """Test comparaison de modèles avec architecture modulaire."""
    from src.core.metrics.evaluator import SummoraEvaluator

    evaluator = SummoraEvaluator()

    reference = "nous validons le budget planning"
    transcriptions = {
        'perfect': "nous validons le budget planning",           # Parfait
        'typo': "nous validons le budget planing",             # Erreur légère
        'error': "nous approuvons le budget stratégie"         # Erreur importante
    }

    # Comparaison avec experimental
    reports = evaluator.compare_models(reference, transcriptions, include_experimental=True)

    # Vérifications
    assert len(reports) == 3
    for model_name, report in reports.items():
        assert isinstance(report.primary_composite_score, float)
        assert isinstance(report.experimental_composite_score, float)
        assert report.overall_grade in ['A+', 'A', 'B', 'C', 'D']
        assert len(report.metrics_used) >= 5  # Primary + Experimental

    # Le modèle perfect doit être le meilleur
    best_model, best_score = evaluator.get_best_model(reports)
    assert best_model == 'perfect'
    assert best_score > 0.9  # Score très élevé attendu

    print("✅ Comparaison modèles modulaire OK")
    for model, report in reports.items():
        print(f"   {model}: Primary={report.primary_composite_score:.3f}, "
              f"Exp={report.experimental_composite_score:.3f} ({report.overall_grade})")
    print(f"   Meilleur: {best_model} ({best_score:.3f})")

def test_factory_function():
    """Test factory function."""
    from src.core.metrics.evaluator import create_summora_evaluator

    # Factory avec défauts
    evaluator = create_summora_evaluator()

    # Vérifications
    assert evaluator.business_vocab is not None
    assert len(evaluator.business_vocab) > 0
    assert 'budget' in evaluator.business_vocab
    assert 'meeting' in evaluator.business_vocab

    # Test rapide
    reference = "budget meeting"
    hypothesis = "budget meeting"

    report = evaluator.evaluate_complete(reference, hypothesis)
    assert report.overall_grade in ['A+', 'A', 'B', 'C', 'D']

    print("✅ Factory function OK")
    print(f"   Business vocab: {len(evaluator.business_vocab)} mots")
    print(f"   Grade test: {report.overall_grade}")

def test_business_vocabulary_modular():
    """Test vocabulaire business avec architecture modulaire."""
    from src.core.metrics.evaluator import SummoraEvaluator

    business_vocab = {'budget', 'planning', 'validation', 'meeting', 'deadline'}
    evaluator = SummoraEvaluator(business_vocab)

    reference = "validation du budget meeting avec deadline"
    hypothesis = "validation du budget réunion avec échéance"  # Synonymes

    report = evaluator.evaluate_complete(reference, hypothesis)

    # Vérifications business focus
    assert report.wer_result.business_focused == True
    assert report.cer_result.business_focused == True
    assert report.per_result.business_focused == True
    assert report.semdist_result.business_focused == True

    # Score doit être correct malgré synonymes
    assert report.primary_composite_score > 0.5  # Pas trop pénalisé

    print("✅ Business vocabulary modulaire OK")
    print(f"   WER business: {report.wer_result.score:.3f}")
    print(f"   Primary score: {report.primary_composite_score:.3f}")

def test_error_handling_modular():
    """Test gestion d'erreurs avec architecture modulaire."""
    from src.core.metrics.evaluator import SummoraEvaluator

    evaluator = SummoraEvaluator()

    # Test avec textes vides (peut causer erreurs)
    reference = ""
    hypothesis = "test"

    # L'évaluateur doit gérer gracieusement
    report = evaluator.evaluate_complete(reference, hypothesis)

    # Vérifications robustesse
    assert report is not None
    assert report.overall_grade is not None
    assert isinstance(report.recommendations, list)

    print("✅ Gestion d'erreurs modulaire OK")
    print(f"   Grade avec texte vide: {report.overall_grade}")
