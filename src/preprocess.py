"""
Module de préprocessing audio pour Summora
Gestion du chargement et validation des fichiers audio
"""
from pathlib import Path
from typing import Optional, Union
import logging
from dataclasses import dataclass

from .utils import validate_audio_path, setup_nltk_resources

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Configuration pour le traitement audio."""
    upload_dir: Optional[Path] = None
    auto_setup_nltk: bool = True

class AudioLoader:
    """
    Gestionnaire de chargement des fichiers audio.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

        if self.config.auto_setup_nltk:
            setup_nltk_resources()

    def load_local_file(self, filename: str, upload_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Charge un fichier audio local.

        Args:
            filename: Nom du fichier à charger
            upload_dir: Répertoire de base (optionnel)

        Returns:
            Path: Chemin validé du fichier ou None si erreur
        """
        base_dir = upload_dir or self.config.upload_dir

        if base_dir:
            file_path = base_dir / filename
        else:
            file_path = Path(filename)

        return validate_audio_path(file_path)

    def save_uploaded_file(self, filename: str, content: bytes,
                          save_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Sauvegarde un fichier uploadé.

        Args:
            filename: Nom du fichier
            content: Contenu binaire du fichier
            save_dir: Répertoire de sauvegarde

        Returns:
            Path: Chemin du fichier sauvé ou None si erreur
        """
        try:
            save_path = (save_dir or Path.cwd()) / filename

            # Validation du format avant sauvegarde
            if not validate_audio_path(filename):  # Juste le nom pour le format
                return None

            with open(save_path, "wb") as f:
                f.write(content)

            logger.info(f"✅ Fichier sauvegardé: {filename}")
            return save_path

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde {filename}: {e}")
            return None

    def prepare_audio(self, source: str, **kwargs) -> Optional[Path]:
        """
        Point d'entrée principal pour préparer un fichier audio.

        Args:
            source: Type de source ('local', 'upload')
            **kwargs: Arguments spécifiques au type de source

        Returns:
            Path: Chemin du fichier préparé
        """
        if source == "local":
            return self.load_local_file(
                kwargs.get('filename', ''),
                kwargs.get('upload_dir')
            )
        elif source == "upload":
            return self.save_uploaded_file(
                kwargs.get('filename', ''),
                kwargs.get('content', b''),
                kwargs.get('save_dir')
            )
        else:
            logger.error(f"❌ Source inconnue: {source}")
            return None

# Fonction helper pour compatibilité
def load_audio_file(filename: str, upload_dir: Optional[str] = None) -> Optional[Path]:
    """
    Helper function pour charger un fichier audio.

    Args:
        filename: Nom du fichier
        upload_dir: Répertoire de base (optionnel)

    Returns:
        Path: Chemin validé du fichier
    """
    config = AudioConfig(upload_dir=Path(upload_dir) if upload_dir else None)
    loader = AudioLoader(config)
    return loader.load_local_file(filename)
