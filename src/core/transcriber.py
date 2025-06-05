"""
Transcripteur audio spécialisé pour les réunions
Whisper optimisé avec prompts et configuration meeting-specific
"""
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union,List
from dataclasses import dataclass
import logging

import whisper

from utils import clean_text_for_meeting, format_duration

logger = logging.getLogger(__name__)

@dataclass
class MeetingTranscriptionConfig:
    """Configuration de Whisper pour la transcription de meetings."""
    model_size: str = "base"
    language: str = "fr"
    task: str = "transcribe"
    word_timestamps: bool = True
    verbose: bool = False
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True

    # Prompt spécialisé meeting
    initial_prompt: str = (
        "Transcription d'une réunion professionnelle en français."
        "Participants multiples discutant d'actions, décisions, planning et objectifs. "
        "Le vocabulaire inclut des termes business, marketing, finance, logistique et technique, avec quelques anglicismes."
    )

    # Seuils spécifiques meetings
    min_confidence_meeting: float = 0.7
    min_words_per_minute: float = 80    # Débit minimum viable
    max_words_per_minute: float = 200   # Débit maximum viable

class MeetingTranscriber:
    """
    Transcripteur spécialisé pour les réunions d'entreprise.

    Optimisation pour le contexte business:
    - Prompt contextualisé (langage métier, participants multiples)
    - Détection qualité (audio et dynamique de réunion)
    - Métriques spécifiques aux échanges en entreprise
    """
    def __init__(self, config:Optional[MeetingTranscriptionConfig]=None):
        """
        Initialise le transcripteur meeting

        Args:
            config: Configuration de transcription (utilise les défauts si None)

        Retourne:
            Initialise le modèle Whisper avec les paramètres définis
        """
        self.config = config or MeetingTranscriptionConfig()
        self.model_size = self.config.model_size
        self.model = None

        self._load_model()

def _load_model(self) -> None:
    """
    Charge le modèle Whisper optimisé pour meetings.
    """
    logger.info(f"🔄 Chargement modèle Whisper meeting '{self.model_size}'...")

    try:
        self.model = whisper.load_model(self.model_size)
        logger.info(f"✅ Modèle Whisper '{self.model_size}' prêt pour meetings")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle meeting: {e}")
        raise

def _clean_meeting_text(self, text: str) -> str:
    """
    Nettoyage spécialisé pour les transcriptions de meetings.

    Args:
        text: Texte brut de transcription

    Retourne:
        str: Texte nettoyé et optimisé pour meetings
    """
    # Nettoyage de base
    text = clean_text_for_meeting(text)

    # Corrections spécifiques aux meetings
    meeting_corrections = {
        r'\baction item\b': 'action'
        ,r'\bmeeting\b': 'réunion'
        ,r'\bdeadline\b': 'échéance'
        ,r'\bfollow[- ]?up\b': 'suivi'
        ,r'\bfeedback\b': 'retour'
        ,r'\bokay\b': 'ok'
        ,r'\byeah\b': 'oui'
        }

    for pattern, replacement in meeting_corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text
