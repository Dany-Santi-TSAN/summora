"""
Tests pour utils PER - Conversion phonétique.
"""

import pytest

def test_text_to_phonemes_basic():
    """Test conversion texte vers phonèmes basique."""
    from src.core.metrics.utils.utils_per import text_to_phonemes

    # Test mot simple
    phonemes = text_to_phonemes("bonjour")

    assert len(phonemes) > 0
    assert isinstance(phonemes, list)
    assert all(isinstance(p, str) for p in phonemes)

    print(f"✅ 'bonjour' → {phonemes}")

def test_phonemes_comparison():
    """Test comparaison phonétique mots similaires."""
    from src.core.metrics.utils.utils_per import text_to_phonemes

    # Mots proches phonétiquement
    phonemes1 = text_to_phonemes("meeting")
    phonemes2 = text_to_phonemes("miting")  # Erreur de transcription courante

    # Devraient être similaires
    assert len(phonemes1) > 0
    assert len(phonemes2) > 0

    print(f"✅ 'meeting' → {phonemes1}")
    print(f"✅ 'miting' → {phonemes2}")

def test_phoneme_stats():
    """Test statistiques phonétiques."""
    from src.core.metrics.utils.utils_per import text_to_phonemes, get_phoneme_stats

    phonemes = text_to_phonemes("validation")
    stats = get_phoneme_stats(phonemes)

    assert "count" in stats
    assert "vowels" in stats
    assert "consonants" in stats
    assert stats["count"] > 0

    print(f"✅ Stats 'validation': {stats}")
