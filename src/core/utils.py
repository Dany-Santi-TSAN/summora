"""
Utilitaires généraux pour SMA : Summora Meeting Analyzer
Configuration NLTK, validation des fichiers audio et constantes
"""
import nltk
from pathlib import Path
from typing import Set, Optional
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Formats audio supportés par Whisper
SUPPORTED_AUDIO_FORMATS: Set[str] = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"
}

# Ajout de mots vides courants non inclus dans NLTK
# Liste de +400 mots : https://countwordsfree.com/stopwords/french
# Liste spécifique pour le meeting, enrichissement fréquent

MEETING_STOPWORDS_ADDITIONAL = [
    # Pronoms / Verbes fréquents / Grammaire
    "être", "avoir", "faire", "aller", "venir", "voir", "savoir", "pouvoir", "vouloir",
    "falloir", "devoir", "dire", "prendre", "mettre", "donner", "très", "tout", "tous", "toutes",

    # Connecteurs de réunion
    "alors", "donc", "du coup", "ensuite", "après", "puis", "en fait",
    "enfin", "voilà", "bref", "par contre", "quand même", "de toute façon",

    # Fillers spécifiques réunions
    "heu", "euh", "bah", "ben", "hein", "quoi", "genre", "ouais", "ouai",
    "nan", "d'accord", "ok", "okay", "parfait", "exact", "putain", "de ouf", "fréro", "grave",

    # Expressions meetings
    "on va", "il faut", "je pense", "je crois", "on peut", "ça va",
    "c'est bon", "ok alors", "du coup on", "donc on", "alors on", "bon alors",
    "en vrai", "gros", "c'est relou", "c'est chiant", "tu vois", "je sais pas",
    "je veux dire", "c’est clair", "tu sais", "en gros", "en mode", "ça veut dire",
    "tu m’entends", "si tu veux", "je dirais", "ça marche", "j’sais pas", "c’est genre",
    "du style", "genre de truc", "truc de ouf", "c’est chaud", "tu vois ce que je veux dire",
    "tu comprends", "j’veux dire", "tu me suis", "vois-tu", "c'est abusé",

    # Politesse & social
    "merci", "s'il vous plaît", "s'il te plaît", "excusez-moi", "pardon", "désolé",
    "bonjour", "bonsoir", "au revoir", "à bientôt", "bonne journée", "sorry", "thanks", "ciao", "bye"
]
