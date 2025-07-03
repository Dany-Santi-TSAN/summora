import pytest
from src.core.metrics.cer import CERCalculator

def test_cer_identical():
    cer = CERCalculator()
    result = cer.calculate("bonjour", "bonjour")
    assert result.score == 0.0
    assert result.get_grade() == "A+"

def test_cer_one_char_added():
    cer = CERCalculator()
    result = cer.calculate("bonjour", "bonjoure")
    assert result.score > 0.0
    assert result.get_grade() in {"A"}  # CER = 1/7 = 14,28% soit grade A

def test_cer_business_preservation():
    cer = CERCalculator(business_vocab={"budget"})
    ref = "nous validons le budget pour ce projet"
    hyp = "nous validons le budjet pour ce projet"
    result = cer.calculate(ref, hyp)
    assert result.score > 0.0
    assert result.details.get("business_preservation_rate", 1.0) < 1.0

def test_cer_empty_input():
    cer = CERCalculator()
    result = cer.calculate("", "test")
    assert result.score == 1.0  # 100% erreur
    assert "error" in result.details or result.details.get("empty_ref")

def test_cer_model_comparison():
    cer = CERCalculator()
    ref = "nous devons finaliser le budget"
    trans = {
        "A": "nous devons finaliser le budget"
        ,"B": "nous devons finaliser le budjet"
        ,"C": "nous devons finaliser le budgett"
    }
    results = cer.compare_transcriptions(ref, trans)
    scores = {k: v["cer_score"] for k, v in results.items()}
    assert scores["A"] == 0.0
    assert scores["B"] > 0.0
    assert scores["C"] > 0.0
