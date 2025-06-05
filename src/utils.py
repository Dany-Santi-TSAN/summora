"""
Utilitaires généraux pour Summora
Gestion des ressources NLTK, validation des fichiers, etc.
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

def setup_nltk_resources() -> bool:
    """
    Configure les ressources linguistiques NLTK nécessaires.

    Returns:
        bool: True si le setup s'est bien passé, False sinon
    """
    print("\n🔄 Configuration des ressources linguistiques...")

    resources = {
        'punkt': 'tokenizer pré-entraîné pour segmentation des phrases',
        'stopwords': 'mots vides en français (le, la, de, etc.)'
    }

    success = True
    for resource, description in resources.items():
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"✅ {resource} téléchargé: {description}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur téléchargement {resource}: {e}")
            success = False

    if success:
        print("✅ Ressources NLTK téléchargées avec succès")
    else:
        print("⚠️ Certaines ressources NLTK ont échoué - continuons quand même")

    return success

def is_audio_file(filename: str | Path) -> bool:
    """
    Vérifie si un fichier a un format audio supporté.

    Args:
        filename: Chemin vers le fichier à vérifier

    Returns:
        bool: True si le format est supporté, False sinon
    """
    return Path(filename).suffix.lower() in SUPPORTED_AUDIO_FORMATS

def validate_audio_path(file_path: str | Path) -> Optional[Path]:
    """
    Valide et normalise un chemin de fichier audio.

    Args:
        file_path: Chemin vers le fichier audio

    Returns:
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

    Returns:
        Set[str]: Ensemble des extensions supportées
    """
    return SUPPORTED_AUDIO_FORMATS.copy()
