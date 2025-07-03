"""
Tests de découverte pour SemDist.
Validation du concept avant optimisation.
"""

import pytest
from src.core.metrics.semdist import SemDistCalculator

def test_semdist_import():
    """Test d'import basique."""
    from src.core.metrics.semdist import SemDistCalculator
    print("✅ Import SemDistCalculator réussi")

def test_semdist_creation():
    """Test création instance SemDist."""
    from src.core.metrics.semdist import SemDistCalculator

    # Création instance
    semdist = SemDistCalculator()
    assert semdist.get_metric_name() == "semdist"
    print("✅ Instance SemDist créée")

def test_semdist_identical_simple():
    """Test SemDist phrases identiques - version simple."""
    from src.core.metrics.semdist import SemDistCalculator

    semdist = SemDistCalculator()
    result = semdist.calculate("bonjour", "bonjour")

    # Vérifications basiques
    assert result.score == pytest.approx(0.0, abs=1e-6)  # Distance nulle avec ajustement de tolérance
    assert result.metric_name == "semdist"
    assert "semantic_similarity" in result.details

    print(f"✅ Test identique: score={result.score}, similarité={result.details['semantic_similarity']}")

def test_sentence_embeddings_basic():
    """Test génération d'embeddings de base."""
    from src.core.metrics.utils.utils_semdist import get_sentence_embeddings

    # Test avec phrases simples
    sentences = ["bonjour le monde", "hello world"]
    embeddings = get_sentence_embeddings(sentences)

    # Vérifications de base
    assert embeddings.shape[0] == 2  # 2 phrases
    assert embeddings.shape[1] > 0   # Dimension > 0
    assert embeddings.dtype == 'float32' or embeddings.dtype == 'float64'

    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"✅ Type: {embeddings.dtype}")

def test_semdist_identical():
    """Test SemDist avec phrases identiques - doit donner score parfait."""
    semdist = SemDistCalculator()
    result = semdist.calculate("bonjour le monde", "bonjour le monde")

    # Score parfait (distance nulle / ajustement avec tolérance)
    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert result.details["semantic_similarity"] == pytest.approx(1.0, abs=1e-6)
    print(f"🎯 Identique: SemDist={result.score:.3f}, Similarité={result.details['semantic_similarity']:.3f}")

def test_semdist_synonymes():
    """Test SemDist avec synonymes - doit préserver le sens."""
    semdist = SemDistCalculator()
    result = semdist.calculate("c'est une excellente idée", "c'est une très bonne idée")

    # Sens préservé (distance faible)
    assert result.score < 0.3  # Distance faible attendue
    assert result.details["semantic_similarity"] > 0.7  # Similarité élevée
    print(f"🔄 Synonymes: SemDist={result.score:.3f}, Similarité={result.details['semantic_similarity']:.3f}")

def test_semdist_sens_oppose():
    """Test SemDist avec sens opposés - doit donner distance élevée."""
    semdist = SemDistCalculator()
    result = semdist.calculate("j'approuve cette décision", "je rejette cette décision")

    # Sens opposé (distance élevée)
    assert result.score > 0.4  # Distance significative
    assert result.details["semantic_similarity"] < 0.6  # Similarité faible
    print(f"❌ Opposé: SemDist={result.score:.3f}, Similarité={result.details['semantic_similarity']:.3f}")

def test_semdist_non_sens():
    """Test SemDist avec sens opposés - doit donner distance élevée."""
    semdist = SemDistCalculator()
    result = semdist.calculate("je travaille sur l'ordinateur", "les chats sont des bengales")

    # Non sens (distance très élevée)
    assert result.score > 0.9  # Distance significative
    assert result.details["semantic_similarity"] < 0.1  # Similarité faible
    print(f"❌ Opposé: SemDist={result.score:.3f}, Similarité={result.details['semantic_similarity']:.3f}")

def test_semdist_business_context():
    """Test SemDist avec vocabulaire business."""
    business_vocab = {"budget", "planning", "validation", "meeting"}
    semdist = SemDistCalculator(business_vocab=business_vocab)

    ref = "nous validons le budget final"
    hyp = "nous approuvons le budget définitif"  # Sens similaire

    result = semdist.calculate(ref, hyp)

    # Vérifications business
    assert result.business_focused == True
    assert "business_semantic_preservation" in result.details
    assert result.details["meaning_preserved"] == True  # Sens préservé

    print(f"💼 Business: SemDist={result.score:.3f}")
    print(f"   Préservation business: {result.details['business_semantic_preservation']:.3f}")
    print(f"   Sens préservé: {result.details['meaning_preserved']}")

def test_semdist_precision_table():
    """Table de précision SemDist - benchmark de découverte."""
    semdist = SemDistCalculator()

    # Cas de test variés pour évaluer la pertinence
    test_cases = [
        # (référence, hypothèse, attente_humaine)
        ("le meeting est annulé", "la réunion est annulée", "excellent")  # Synonyme
        ,("budget approuvé", "budget validé", "excellent")  # Business synonyme
        ,("nous commençons", "nous débutons", "bon") # Synonyme simple
        ,("projet fini", "projet commencé", "mauvais")  # Sens opposé
        ,("bonjour", "au revoir", "très_mauvais")  # Totalement différent
    ]

    results = []
    for ref, hyp, expected in test_cases:
        result = semdist.calculate(ref, hyp)
        results.append({
            "reference": ref
            ,"hypothesis": hyp
            ,"expected": expected
            ,"semdist_score": result.score
            ,"similarity": result.details["semantic_similarity"]
            ,"grade": result.details.get("semantic_quality_grade", "N/A")
        })

    # Affichage table de précision
    print("\n📊 TABLE DE PRÉCISION SEMDIST")
    print("=" * 80)
    for r in results:
        print(f"'{r['reference']}' → '{r['hypothesis']}'")
        print(f"   Attendu: {r['expected']} | SemDist: {r['semdist_score']:.3f} | Sim: {r['similarity']:.3f} | Grade: {r['grade']}")
        print()

    # Validation logique générale
    # Les cas "excellent" doivent avoir une distance faible
    excellent_cases = [r for r in results if r["expected"] == "excellent"]
    for case in excellent_cases:
        assert case["semdist_score"] < 0.4, f"Cas excellent avec distance trop élevée: {case}"
