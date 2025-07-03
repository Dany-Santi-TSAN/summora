#!/usr/bin/env python3
"""
Test simple des 3 modèles BERTScore disponibles
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.core.metrics.bertscore import BERTScoreCalculator

def test_multilingue():
    """Test BERT multilingue."""
    print("🤖 BERT Multilingue")
    bert = BERTScoreCalculator(model_name="bert-base-multilingual-cased")
    result = bert.calculate("nous validons le budget", "je mange des pommes")
    print(f"   Sens différent: F1={result.score:.3f} | Grade={result.get_grade()}")
    return result.score

def test_mt5():
    """Test Google mT5-base."""
    print("🤖 Google mT5-base")
    bert = BERTScoreCalculator(model_name="google/mt5-base")
    result = bert.calculate("nous validons le budget", "je mange des pommes")
    print(f"   Sens différent: F1={result.score:.3f} | Grade={result.get_grade()}")
    return result.score

def test_roberta():
    """Test XLM-RoBERTa."""
    print("🤖 XLM-RoBERTa")
    bert = BERTScoreCalculator(model_name="xlm-roberta-base")
    result = bert.calculate("nous validons le budget", "je mange des pommes")
    print(f"   Sens différent: F1={result.score:.3f} | Grade={result.get_grade()}")
    return result.score

def test_distilbert():
    """Test DistilBERT multilingue."""
    print("🤖 DistilBERT Multilingue")
    bert = BERTScoreCalculator(model_name="distilbert-base-multilingual-cased")
    result = bert.calculate("nous validons le budget", "je mange des pommes")
    print(f"   Sens différent: F1={result.score:.3f} | Grade={result.get_grade()}")
    return result.score

def benchmark_simple():
    """Benchmark simple des 3 modèles."""
    print("🔬 BENCHMARK SIMPLE - Sens différent")
    print("=" * 50)
    print("📝 'nous validons le budget' vs 'je mange des pommes'")
    print("-" * 50)

    try:
        score_bert = test_multilingue()
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        score_bert = None

    try:
        score_roberta = test_roberta()
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        score_roberta = None

    try:
        score_distil = test_distilbert()
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        score_distil = None

    print("-" * 50)
    print("🎯 RÉSULTATS:")

    # Classement par rigueur (score le plus bas = le plus strict)
    results = []
    if score_bert is not None:
        results.append(("BERT Multilingue", score_bert))
    if score_roberta is not None:
        results.append(("XLM-RoBERTa", score_roberta))
    if score_distil is not None:
        results.append(("DistilBERT", score_distil))

    # Tri par score croissant (plus strict en premier)
    results.sort(key=lambda x: x[1])

    for i, (model, score) in enumerate(results, 1):
        status = "✅ STRICT" if score < 0.5 else ("⚠️ MOYEN" if score < 0.7 else "❌ PERMISSIF")
        print(f"   {i}. {model}: {score:.3f} {status}")

    if results:
        winner = results[0]
        print(f"\n🏆 Plus strict: {winner[0]} (F1={winner[1]:.3f})")

if __name__ == "__main__":
    benchmark_simple()
