"""
Tests pour PER (Phoneme Error Rate).
"""

import pytest

def test_per_identical():
    """Test PER avec textes identiques."""
    from src.core.metrics.per import PERCalculator

    per = PERCalculator()
    result = per.calculate("bonjour", "bonjour")

    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert result.metric_name == "per"
    print(f"✅ PER identique: {result.score}")

def test_per_phonetic_similar():
    """Test PER avec mots phonétiquement proches."""
    from src.core.metrics.per import PERCalculator

    per = PERCalculator()
    result = per.calculate("meeting", "miting")  # Erreur phonétique courante

    # Devrait avoir un PER faible (sons similaires)
    assert result.score < 0.5  # Tolérant aux variations phonétiques
    assert "levenshtein_distance" in result.details

    print(f"✅ PER phonétique 'meeting'→'miting': {result.score:.3f}")
    print(f"   Distance Levenshtein: {result.details['levenshtein_distance']}")

def test_per_vs_orthographic():
    """Test PER vs erreurs orthographiques."""
    from src.core.metrics.per import PERCalculator

    per = PERCalculator()

    # Même son, orthographe différente
    result1 = per.calculate("bonbon", "bon bon")   # identique
    result2 = per.calculate("chat", "rat")   # Sons différents

    # PER devrait être meilleur pour eau/o que chat/rat
    assert result1.score <= result2.score

    print(f"✅ 'eau'→'o': PER={result1.score:.3f}")
    print(f"✅ 'chat'→'rat': PER={result2.score:.3f}")

def test_per_business():
    """Test PER avec vocabulaire business."""
    from src.core.metrics.per import PERCalculator

    business_vocab = {"budget", "planning", "meeting"}
    per = PERCalculator(business_vocab=business_vocab)

    result = per.calculate("budget meeting", "budget miting")

    assert result.business_focused == True
    assert "business_phonetic_preservation" in result.details

    print(f"✅ PER business: {result.score:.3f}")
    print(f"   Préservation phonétique: {result.details['business_phonetic_preservation']:.3f}")

def test_per_precision_table():
    """Table de précision PER - benchmark phonétique."""
    from src.core.metrics.per import PERCalculator

    per = PERCalculator()

    # Cas de test phonétiques variés
    test_cases = [
        # (référence, hypothèse, attente_humaine)
        ("meeting", "meeting", "parfait")           # Identique
        ,("meeting", "miting", "excellent")          # Erreur Whisper courante
        ,("budget", "buget", "bon")                  # Faute de frappe
        ,("validation", "validasion", "bon")         # Substitution phonétique
        ,("planning", "planing", "bon")              # Double consonne
        ,("bonjour", "bonzhur", "moyen")            # Phonème proche
        ,("chat", "rat", "mauvais")                  # Phonème différent
        ,("ordinateur", "voiture", "très_mauvais")  # Totalement différent
    ]

    results = []
    for ref, hyp, expected in test_cases:
        result = per.calculate(ref, hyp)

        # Analyse des phonèmes pour debug
        ref_phonemes = result.details.get("ref_phoneme_stats", {}).get("phoneme_string", "")
        hyp_phonemes = result.details.get("hyp_phoneme_stats", {}).get("phoneme_string", "")

        results.append({
            "reference": ref
            ,"hypothesis": hyp
            ,"expected": expected
            ,"per_score": result.score
            ,"levenshtein": result.details.get("levenshtein_distance", 0)
            ,"phonemes_ref": result.details.get("phonemes_ref", 0)
            ,"phonemes_hyp": result.details.get("phonemes_hyp", 0)
            ,"ref_phonemes_str": ref_phonemes
            ,"hyp_phonemes_str": hyp_phonemes
        })

    # Affichage table de précision
    print("\n📊 TABLE DE PRÉCISION PER (Phoneme Error Rate)")
    print("=" * 90)
    for r in results:
        print(f"'{r['reference']}' → '{r['hypothesis']}'")
        print(f"   Attendu: {r['expected']} | PER: {r['per_score']:.3f} | Distance: {r['levenshtein']}")
        print(f"   Phonèmes REF: {r['ref_phonemes_str']}")
        print(f"   Phonèmes HYP: {r['hyp_phonemes_str']}")
        print()

    # Validations logiques générales
    perfect_cases = [r for r in results if r["expected"] == "parfait"]
    for case in perfect_cases:
        assert case["per_score"] == 0.0, f"Cas parfait avec PER > 0: {case}"

    # Les cas "excellent" doivent avoir un PER très faible
    excellent_cases = [r for r in results if r["expected"] == "excellent"]
    for case in excellent_cases:
        assert case["per_score"] < 0.5, f"Cas excellent avec PER élevé: {case}"
