"""
Utilitaires pour PER (Phoneme Error Rate).
Conversion texte vers phonèmes pour français selon approche HATS.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

def text_to_phonemes(text: str, method: str = "simple") -> List[str]:
    """
    Convertit un texte français en séquence de phonèmes.

    Implémentation basique pour démarrer, basée sur les règles
    phonétiques françaises courantes.

    Args:
        text: Texte français à convertir
        method: Méthode de conversion ("simple", "advanced")

    Retourne:
        List[str]: Séquence de phonèmes
    """
    if not text or not text.strip():
        return []

    # Nettoyage du texte
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Supprime ponctuation

    if method == "simple":
        return _simple_french_phonemes(text)
    else:
        # Pour version avancée ultérieure
        return _simple_french_phonemes(text)

def _simple_french_phonemes(text: str) -> List[str]:
    """
    Conversion phonétique française simplifiée.

    Règles de base pour démarrer - à enrichir selon besoins.

    Args:
        text: Texte nettoyé

    Retourne:
        List[str]: Phonèmes approximatifs
    """
    # Dictionnaire de base français (échantillon)
    phoneme_rules = {
        # Voyelles
        'a': 'a', 'e': 'ə', 'i': 'i', 'o': 'o', 'u': 'y',
        'é': 'e', 'è': 'ɛ', 'ê': 'ɛ', 'à': 'a', 'ù': 'y',

        # Consonnes courantes
        'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
        'h': '', 'j': 'ʒ', 'k': 'k', 'l': 'l', 'm': 'm',
        'n': 'n', 'p': 'p', 'q': 'k', 'r': 'ʁ', 's': 's',
        't': 't', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'i', 'z': 'z',

        # Digrammes courants
        'ch': 'ʃ', 'ph': 'f', 'th': 't', 'qu': 'k',
        'ou': 'u', 'on': 'ɔ̃', 'an': 'ɑ̃', 'en': 'ɑ̃',
        'in': 'ɛ̃', 'un': 'œ̃'
    }

    phonemes = []
    i = 0

    while i < len(text):
        if text[i].isspace():
            i += 1
            continue

        # Essaie digrammes d'abord
        if i < len(text) - 1:
            digram = text[i:i+2]
            if digram in phoneme_rules:
                phoneme = phoneme_rules[digram]
                if phoneme:  # Ignore 'h' muet
                    phonemes.append(phoneme)
                i += 2
                continue

        # Caractère simple
        char = text[i]
        if char in phoneme_rules:
            phoneme = phoneme_rules[char]
            if phoneme:  # Ignore 'h' muet
                phonemes.append(phoneme)

        i += 1

    return phonemes

def phonemes_to_string(phonemes: List[str], separator: str = " ") -> str:
    """
    Convertit une liste de phonèmes en chaîne pour debug.

    Args:
        phonemes: Liste de phonèmes
        separator: Séparateur entre phonèmes

    Retourne:
        str: Chaîne de phonèmes
    """
    return separator.join(phonemes)

def get_phoneme_stats(phonemes: List[str]) -> dict:
    """
    Calcule des statistiques sur une séquence de phonèmes.

    Args:
        phonemes: Liste de phonèmes

    Retourne:
        dict: Statistiques phonétiques
    """
    if not phonemes:
        return {"count": 0, "unique": 0, "vowels": 0, "consonants": 0}

    # Classification basique
    vowels = {'a', 'e', 'ə', 'i', 'o', 'u', 'y', 'ɛ', 'ɔ', 'ɑ', 'œ'}

    vowel_count = sum(1 for p in phonemes if any(v in p for v in vowels))
    consonant_count = len(phonemes) - vowel_count

    return {
        "count": len(phonemes),
        "unique": len(set(phonemes)),
        "vowels": vowel_count,
        "consonants": consonant_count,
        "phoneme_string": phonemes_to_string(phonemes)
    }
