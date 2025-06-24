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
    Utilise le module spécialisé utils_stopwords_meeting_fr.

    Retourne:
        Set[str]: Stopwords français NLTK + extensions spécifiques au meeting
    """
    try:
        from nltk.corpus import stopwords
        french_stopwords = set(stopwords.words('french'))
    except Exception as e:
        logger.warning(f"Erreur chargement stopwords NLTK: {e}")
        french_stopwords = set()

    # Import du module spécialisé stopwords meetings
    try:
        from .utils_stopwords_meeting_fr import get_all_meeting_stopwords_fr
        meeting_stopwords = get_all_meeting_stopwords_fr()

        # Combinaison NLTK + meetings spécialisés
        all_stopwords = french_stopwords.union(meeting_stopwords)

        logger.info(f"📋 {len(french_stopwords)} stopwords NLTK français")
        logger.info(f"📋 {len(meeting_stopwords)} stopwords meetings spécialisés")
        logger.info(f"📋 {len(all_stopwords)} stopwords total (optimized)")

        return all_stopwords

    except ImportError as e:
        logger.error(f"❌ Erreur import stopwords meetings: {e}")
        logger.info("🔄 Fallback vers stopwords NLTK uniquement")
        return french_stopwords

def get_stopwords_by_category(category: str) -> Set[str]:
    """
    Retourne les stopwords d'une catégorie spécifique.

    Args:
        category: Catégorie de stopwords ('contractions', 'basic', 'verbs', etc.)

    Retourne:
        Set[str]: Stopwords de la catégorie
    """
    try:
        from .utils_stopwords_meeting_fr import get_category_stopwords
        return get_category_stopwords(category)
    except ImportError:
        logger.warning(f"Module stopwords meetings non disponible pour catégorie '{category}'")
        return set()

def validate_stopwords_coverage(problematic_words: list) -> dict:
    """
    Valide la couverture des stopwords sur des mots problématiques.

    Args:
        problematic_words: Liste des mots qui passent dans les topics

    Returns:
        dict: Statistiques de couverture
    """
    try:
        from .utils_stopwords_meeting_fr import analyze_stopwords_coverage
        return analyze_stopwords_coverage(problematic_words)
    except ImportError:
        logger.warning("Module stopwords meetings non disponible pour validation")
        return {"coverage_rate": 0.0, "error": "module_unavailable"}

"""
Validation des fichiers audio (inchangé)
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

"""
Nettoyage spécifique pour la transcription (inchangé)
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

# === TESTS UTILITAIRES ===
# Note: En prod, ces tests iraient dans tests, mais on teste ici dans un premier temps pour fixer un bug

if __name__ == "__main__":
    # Simple validation pour debug développement
    print("🔍 VALIDATION UTILS STOPWORDS")
    print("=" * 40)

    # Test basique chargement
    stopwords = get_meeting_stopwords()
    print(f"📋 {len(stopwords)} stopwords chargés pour meetings")

    # Test mots problématiques de tes résultats
    problematic = ["c'est", "qu'il", "j'ai", "bon", "oui", "fin"]
    coverage = validate_stopwords_coverage(problematic)

    if coverage.get('coverage_rate', 0) > 0:
        print(f"🎯 Couverture: {coverage['coverage_rate']*100:.1f}%")

        # Test catégorie si module disponible
        contractions = get_stopwords_by_category('contractions')
        if contractions:
            print(f"📝 Contractions: {len(contractions)} mots")

        print("✅ Integration stopwords OK")
    else:
        print("⚠️ Module stopwords meetings pas encore créé")
