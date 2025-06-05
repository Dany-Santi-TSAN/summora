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

def setup_nltk_ressource() -> bool:
    """
    Configure les ressources linguistique NLTK nécessaires.

    Retourne:
        bool : True si le setup s'est bien déroulé
    """
    logger.info("🔄 Configuration des ressources linguistiques...")

    ressources = {
        'punkt': 'tokenizer pour segmentation des phrases'
        ,'stopwords': 'mots vides français'
    }

    success = True
    for ressource, description in ressources.items():
        try:
            nltk.download(ressource, quiet=True)
            logger.info(f"✅ {ressource} téléchargé: {description}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur téléchargement {ressource}: {e}")
            success = False

    if success:
        logger.info("✅ Ressources NLTK configurées avec succès")
    else:
        logger.warning("⚠️ Certaines ressources NLTK ont échoué - continuons quand même")

    return success

def get_meeting_stopwords() -> Set[str]:
    """
    Retourne l'ensemble complet des stopwords français spécifique au meeting.

    Retourne:
        Set[str]: Stopwords français NLTK + extensions spécifiques au meeting
    """
    try:
        from nltk.corpus import stopwords
        french_stopwords = set(stopwords.words('french'))
    except Exception as e:
        logger.warning(f"Erreur chargement stopwords NLTK: {e}")
        french_stopwords = set()

    # Ajout de la liste des stopwords spécifique au meeting
    french_stopwords.update(MEETING_STOPWORDS_ADDITIONAL)
    logger.info(f"📋 {len(french_stopwords)} stopwords français (meeting-optimized)")

    return french_stopwords

"""
Validation des fichiers audio
"""

def is_audio_file(filename: str | Path) -> bool:
    """
    Vérifie si un fichier a un format audio supporté par Whisper

    Args:
        filename: Chemin vers le fichier à vérifier

    Retourne:
        bool: True si le format est supporté.
    """
    return Path(filename).suffix.lower() in SUPPORTED_AUDIO_FORMATS

def validate_audio_path(file_path: str | Path) -> Optional[Path]:
    """
    Valide et normalise un chemin de fichier audio.

    Args:
        file_path: Chemin vers le fichier audio

    Retourne:
        Path: Chemin validé ou None si invalide
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(f"❌ Fichier introuvable: {path}")
        return None

    if not is_audio_file(path):
        logger.error(f"❌ Format non supporté: {path.suffix}")
        logger.info(f"💡 Formats supportés: {SUPPORTED_AUDIO_FORMATS}")
        return None

    logger.info(f"✅ Fichier audio validé: {path.name}")
    return path

def get_supported_formats() -> Set[str]:
    """
    Retourne la liste des formats audio supportés.

    Retourne:
        Set[str]: Ensemble des extensions supportées
    """
    return SUPPORTED_AUDIO_FORMATS.copy()

def get_supported_formats() -> Set[str]:
    """
    Retourne la liste des formats audio supportés.

    Retourne:
        Set[str]: Ensemble des extensions supportées
    """
    return SUPPORTED_AUDIO_FORMATS.copy()

"""
Nettoyage spécifique pour la transcription
"""

def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes vers un format lisible.

    Args:
        seconds: Durée en secondes

    Retourne:
        str: Durée formatée (ex: "2m 30s", "1h 15m")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"

def clean_text_for_meeting(text: str) -> str:
    """
    Nettoie un texte spécifiquement pour l'analyse de meeting.

    Args:
        text: Texte brut à nettoyer

    Retourne:
        str: Texte nettoyé optimisé pour meetings
    """
    import re

    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text)

    # Garde seulement la ponctuation utile pour meetings
    text = re.sub(r'[^\w\s\'\-\.,;:!?\(\)]', '', text)

    # Suppression des répétitions communes en oral
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # "le le" -> "le"

    return text.strip()
