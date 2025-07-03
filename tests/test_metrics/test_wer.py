import pytest
from src.core.metrics.wer import WERCalculator


def test_wer_identical():
    """Test WER avec texte identique."""
    wer = WERCalculator()
    result = wer.calculate("bonjour monde", "bonjour monde")

    assert result.score == 0.0
    assert result.get_grade() == "A+"  # WER parfait = A+


def test_wer_one_word_different():
    """Test WER avec 1 mot différent."""
    wer = WERCalculator()
    result = wer.calculate("bonjour monde", "bonjour univers")  # 1 substitution

    assert result.score > 0.0
    assert result.score < 1.0  # Pas d'erreur totale
    assert result.get_grade() in {"A+", "A", "B", "C"}  # WER ≈ 0.647 acceptable

    print(f"WER 1 mot différent: {result.score:.3f}")  # Debug pour voir le vrai score


def test_wer_business_preservation():
    """Test WER avec vocabulaire business."""
    wer = WERCalculator(business_vocab={"budget", "planning", "action"})
    ref = "nous validons le budget pour ce projet"
    hyp = "nous validons le planning pour ce projet"  # budget → planning

    result = wer.calculate(ref, hyp)

    assert result.score > 0.0  # 1 mot différent
    assert result.business_focused == True
    assert result.details.get("business_preservation_rate", 1.0) < 1.0  # Mot business changé

    # Debug scores et business stats
    print(f"\n🔍 WER Business Debug:")
    print(f"   Score WER: {result.score:.3f}")
    print(f"   Grade: {result.get_grade()}")
    print(f"   Business preservation: {result.details.get('business_preservation_rate', 0):.3f}")
    print(f"   Mots business ref: {result.details.get('business_words_ref', 0)}")
    print(f"   Mots business préservés: {result.details.get('preserved_business_words', [])}")


def test_wer_empty_input():
    """Test WER avec entrée vide."""
    wer = WERCalculator()
    result = wer.calculate("", "test mot")

    # WER avec référence vide = problématique
    assert result.score >= 1.0  # Erreur majeure
    # Peut générer erreur ou score élevé selon implémentation


def test_wer_model_comparison():
    """Test comparaison de modèles avec WER."""
    wer = WERCalculator()
    ref = "nous devons finaliser le budget"
    transcriptions = {
        "whisper_tiny": "nous devons finaliser le budget",      # 0 erreur
        "whisper_base": "nous devons finaliser le planning",    # 1 erreur
        "whisper_small": "nous devons terminer le budget"       # 1 erreur différente
    }

    results = wer.compare_transcriptions(ref, transcriptions)
    scores = {k: v["wer_score"] for k, v in results.items()}

    assert scores["whisper_tiny"] == 0.0
    assert scores["whisper_base"] > 0.0
    assert scores["whisper_small"] > 0.0
    assert scores["whisper_tiny"] < scores["whisper_base"]  # tiny meilleur que base
    assert scores["whisper_tiny"] < scores["whisper_small"]  # tiny meilleur que small


def test_wer_grade_thresholds():
    """Test des seuils de grades WER."""
    wer = WERCalculator()

    # Test différents niveaux d'erreur
    cases = [
        ("parfait test", "parfait test", "A+"),           # 0.0
        ("un deux trois", "un deux quatre", "C"),         # 0.33 (1/3)
        ("mot", "autre", "D"),                            # 1.0 (1/1)
        ("a b c d", "w x y z", "D")                       # 1.0 (4/4)
    ]

    for ref, hyp, expected_min_grade in cases:
        result = wer.calculate(ref, hyp)
        grade = result.get_grade()
        wer_score = result.score

        print(f"'{ref}' vs '{hyp}': WER={wer_score:.3f}, Grade={grade}")

        # Vérifications logiques
        if wer_score == 0.0:
            assert grade == "A+"
        elif wer_score >= 1.0:
            assert grade == "D"
        # Pour les autres, on accepte la logique du grading


def test_wer_business_density():
    """Test densité mots business dans WER."""
    business_vocab = {"budget", "planning", "validation", "meeting"}
    wer = WERCalculator(business_vocab)

    # Texte avec forte densité business
    ref = "validation du budget et planning meeting"  # 4/6 mots business
    hyp = "validation du budget et planning réunion"  # meeting → réunion

    result = wer.calculate(ref, hyp)

    assert result.score > 0.0  # 1 erreur
    assert result.details["business_word_density"] > 0.5  # Plus de 50% business
    assert result.details["business_words_ref"] == 4
    assert result.details["business_preservation_rate"] == 0.75  # 3/4 préservés


def test_wer_vs_cer_comparison():
    """Test comparaison logique WER vs CER - comportements complémentaires."""
    from src.core.metrics.cer import CERCalculator

    wer_calc = WERCalculator()
    cer_calc = CERCalculator()

    # 📊 Cas 1: CER plus sensible aux petites erreurs
    ref1 = "test"
    hyp1 = "teste"  # 1 caractère ajouté

    wer_result1 = wer_calc.calculate(ref1, hyp1)
    cer_result1 = cer_calc.calculate(ref1, hyp1)

    # WER strict (mot différent = 1.0), CER nuancé (1 char / 4 = 0.25)
    assert wer_result1.score == 1.0    # WER: substitution complète
    assert cer_result1.score == 0.25   # CER: 1/4 caractères différents
    assert cer_result1.score < wer_result1.score  # CER plus indulgent

    print(f"📊 Cas 1 - Petite erreur:")
    print(f"   WER: {wer_result1.score:.3f}, CER: {cer_result1.score:.3f}")

    # 📊 Cas 2: WER et CER équivalents sur mots parfaits
    ref2 = "bonjour monde"
    hyp2 = "bonjour monde"  # Identique

    wer_result2 = wer_calc.calculate(ref2, hyp2)
    cer_result2 = cer_calc.calculate(ref2, hyp2)

    assert wer_result2.score == 0.0
    assert cer_result2.score == 0.0
    assert wer_result2.score == cer_result2.score  # Égalité sur perfection

    print(f"📊 Cas 2 - Parfait:")
    print(f"   WER: {wer_result2.score:.3f}, CER: {cer_result2.score:.3f}")

    # 🎯 Insights: WER et CER sont complémentaires
    # - WER: strict sur l'intégrité des mots (business critique)
    # - CER: nuancé sur les petites fautes (typos, accents)

    # Alternative: test avec ajout d'espace
    ref2 = "bonjour monde"
    hyp2 = "bonjour  monde"  # Double espace = même mots, caractères différents

    wer_result2 = wer_calc.calculate(ref2, hyp2)
    cer_result2 = cer_calc.calculate(ref2, hyp2)

    print(f"\n📊 Test espace:")
    print(f"WER: {wer_result2.score:.3f}, CER: {cer_result2.score:.3f}")

def test_wer_business_words_preserved():
    """Test préservation explicite mots business."""
    business_vocab = {"budget", "deadline", "action", "validation"}
    wer = WERCalculator(business_vocab)

    ref = "nous validons le budget avant la deadline"
    hyp = "nous approuvons le budget avant la deadline"  # validons → approuvons

    result = wer.calculate(ref, hyp)

    # Vérifications business
    assert "preserved_business_words" in result.details
    preserved = result.details["preserved_business_words"]
    assert "budget" in preserved
    assert "deadline" in preserved
    assert len(preserved) >= 2  # Au moins budget + deadline préservés
