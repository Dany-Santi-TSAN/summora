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

from .utils import clean_text_for_meeting, format_duration

logger = logging.getLogger(__name__)

# === Configuration du modèle Whisper ===

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
    min_words_per_minute: float = 80    # Débit minimum viable unité wpm
    max_words_per_minute: float = 200   # Débit maximum viable unité wpm

# === Classe Transcripteur pour réunions ===

class MeetingTranscriber:
    """
    Transcripteur spécialisé pour les réunions d'entreprise.

    Optimisation pour le contexte business:
    - Prompt contextualisé (langage métier, participants multiples)
    - Détection qualité (audio et dynamique de réunion)
    - Métriques spécifiques aux échanges en entreprise
    """

    # === Initialisation et chargement du modèle ===

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

    # === Nettoyage du texte pour usage professionnel ===

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

    # === Analyse du contenu transcrit ===

    def _analyze_meeting_content(self, text: str, segments: List[Dict]) -> Dict:
        """
        Analyse le contenu transcrit pour extraire des métriques meeting.

        Args:
            text: Texte transcrit
            segments: Segments Whisper avec timestamps

        Returns:
            Dict: Métriques spécifiques aux meetings (densité, structure, counts par type)
        """
        # Définition des catégories de mots-clés spécifique au réunion
        meeting_keywords = {
            'action': ['action', 'tâche', 'faire', 'réaliser', 'livrer', 'assigner']
            ,'decision': ['décision', 'décider', 'valider', 'trancher', 'approuver']
            ,'planning': ['planning', 'délai', 'échéance', 'deadline', 'calendrier', 'roadmap']
            ,'question': ['question', 'souci', 'problème', 'blocage', 'difficulté', 'bug']
            ,'agreement': ['d’accord', 'ok', 'parfait', 'exactement', 'entendu']
            ,'disagreement': ['non', 'pas d’accord', 'mais', 'cependant', 'toutefois']
            ,'feedback': ['retour', 'commentaire', 'avis', 'point de vue', 'feedback']
            ,'next_step': ['prochaine étape', 'suivant', 'ensuite', 'à faire', 'pour la suite']
            ,'closing': ['en résumé', 'pour conclure', 'synthèse', 'résumé', 'bilan']
        }

        keyword_counts = {}
        text_lower = text.lower()

        for category, keywords in meeting_keywords.items():
            count = sum(text_lower.count(keyword) for keyword in keywords)
            keyword_counts[category] = count

        # Calcul de la "meeting density" (richesse du contenu)
        total_keywords = sum(keyword_counts.values())
        word_count = len(text.split())
        meeting_density = (total_keywords / word_count * 100) if word_count > 0 else 0

        # Détection de structure meeting
        has_structure = any([
            'ordre du jour' in text_lower,
            'agenda' in text_lower,
            'point suivant' in text_lower,
            'première partie' in text_lower,
            'pour conclure' in text_lower
        ])

        return {
            'keyword_counts': keyword_counts,
            'meeting_density': round(meeting_density, 2),
            'has_structure': has_structure,
            'total_meeting_keywords': total_keywords
        }


# === Calcul des scores de confiance ===

    def _calculate_meeting_confidence(self, segments: List[Dict],
                                    duration: float, word_count: int) -> Dict:
        """
        Calcule un score de confiance spécifique aux meetings.

        Args:
            segments: Segments Whisper avec probabilités
            duration: Durée en secondes
            word_count: Nombre de mots transcrits

        Retourne:
            Dict: Métriques de confiance meeting
        """
        # Confiance Whisper standard
        confidences = []
        for segment in segments:
            if "words" in segment:
                for word in segment["words"]:
                    if "probability" in word:
                        confidences.append(word["probability"])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Débit de parole (indicateur de qualité meeting)
        speaking_rate = word_count / (duration / 60) if duration > 0 else 0

        # Score de débit (optimal entre 80-200 mots/minute)
        if self.config.min_words_per_minute <= speaking_rate <= self.config.max_words_per_minute:
            rate_score = 1.0
            rate_status = "✅ Débit optimal pour meeting"
        elif speaking_rate < self.config.min_words_per_minute:
            rate_score = speaking_rate / self.config.min_words_per_minute
            rate_status = "⚠️ Débit lent - meeting peu dense"
        else:
            rate_score = self.config.max_words_per_minute / speaking_rate
            rate_status = "⚠️ Débit rapide - possibles coupures"

        # Score de confiance meeting global
        """
        Le score meeting_confidence combine la confiance moyenne de Whisper (70%) et le débit de parole (30%)
        pour refléter à la fois la précision technique et la fluidité de l’échange – pondérations ajustables si besoin.
        """
        meeting_confidence = (avg_confidence * 0.7) + (rate_score * 0.3)

        # Grade meeting
        if meeting_confidence >= 0.85:
            confidence_grade = "A"
        elif meeting_confidence >= 0.75:
            confidence_grade = "B"
        elif meeting_confidence >= 0.65:
            confidence_grade = "C"
        else:
            confidence_grade = "D"

        return {
            'whisper_confidence': round(avg_confidence, 3),
            'speaking_rate': round(speaking_rate, 1),
            'rate_score': round(rate_score, 3),
            'rate_status': rate_status,
            'meeting_confidence': round(meeting_confidence, 3),
            'confidence_grade': confidence_grade,
            'total_words_analyzed': len(confidences)
        }


# === Fonction principale de transcription ===

    def transcribe_meeting(self, audio_path: Union[str, Path]) -> Dict:
        """
        Transcrit un fichier audio de meeting avec optimisations spécialisées.

        Args:
            audio_path: Chemin vers le fichier audio

        Retourne:
            Dict: Résultats de transcription optimisés pour meetings
        """
        audio_path = Path(audio_path)

        try:
            logger.info(f"🎤 Transcription meeting: {audio_path.name}")
            logger.info(f"⚙️ Modèle: {self.model_size} | Langue: {self.config.language}")
            start_time = datetime.now()

            # Transcription avec configuration meeting
            result = self.model.transcribe(
                str(audio_path)
                ,language=self.config.language
                ,task=self.config.task
                ,word_timestamps=self.config.word_timestamps
                ,verbose=self.config.verbose
                ,temperature=self.config.temperature
                ,compression_ratio_threshold=self.config.compression_ratio_threshold
                ,logprob_threshold=self.config.logprob_threshold
                ,no_speech_threshold=self.config.no_speech_threshold
                ,condition_on_previous_text=self.config.condition_on_previous_text
                ,initial_prompt=self.config.initial_prompt
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Nettoyage spécialisé meeting
            raw_text = result['text']
            cleaned_text = self._clean_meeting_text(raw_text)

            # Métriques de base
            word_count = len(cleaned_text.split())
            duration = result.get('duration', 0)
            segments = result.get('segments', [])

            # Analyse contenu meeting
            content_analysis = self._analyze_meeting_content(cleaned_text, segments)

            # Confiance meeting
            confidence_analysis = self._calculate_meeting_confidence(segments, duration, word_count)

            # Aperçu intelligent (premier et dernier segments)
            if len(cleaned_text) > 200:
                preview = cleaned_text[:150] + " [...] " + cleaned_text[-50:]
            else:
                preview = cleaned_text

            # Log des résultats
            logger.info(f"✅ Transcription meeting terminée en {processing_time:.2f}s")
            logger.info(f"📝 {word_count} mots | {format_duration(duration)}")
            logger.info(f"🎯 Confiance meeting: {confidence_analysis['meeting_confidence']:.3f} ({confidence_analysis['confidence_grade']})")
            logger.info(f"💡 Densité meeting: {content_analysis['meeting_density']:.1f}%")
            logger.info(f"👁️ Aperçu: {preview[:100]}...")

            return {
                # Texte transcrit
                "text": cleaned_text,
                "raw_text": raw_text,
                "preview": preview,

                # Métriques de base
                "word_count": word_count,
                "duration": duration,
                "duration_formatted": format_duration(duration),
                "segments": segments,
                "language": result.get('language', 'fr'),

                # Métriques meeting spécifiques
                "meeting_content": content_analysis,
                "meeting_confidence": confidence_analysis,

                # Métriques temporelles
                "processing_time": round(processing_time, 2),
                "speaking_rate": confidence_analysis['speaking_rate'],
                "speaking_rate_status": confidence_analysis['rate_status'],

                # Configuration utilisée
                "model_info": {
                    "model_size": self.model_size,
                    "initial_prompt": self.config.initial_prompt,
                    "language": self.config.language
                },

                # Métadonnées
                "transcription_timestamp": datetime.now().isoformat(),
                "audio_path": str(audio_path),
                "optimized_for": "meeting_transcription"
            }

        except FileNotFoundError:
            error_msg = f"❌ Fichier audio meeting non trouvé: {audio_path}"
            logger.error(error_msg)
            return {
                "error": "file_not_found",
                "message": error_msg,
                "audio_path": str(audio_path),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"❌ Erreur transcription meeting: {str(e)}"
            logger.error(error_msg)
            logger.error(f"🔍 Type d'erreur: {type(e).__name__}")
            logger.error("📋 Stack trace:", exc_info=True)

            return {
                "error": "meeting_transcription_failed",
                "message": error_msg,
                "error_type": type(e).__name__,
                "audio_path": str(audio_path),
                "timestamp": datetime.now().isoformat()
            }



# === Fonctions utilitaires / Factories ===

def create_meeting_transcriber(model_size: str = "base", **kwargs) -> MeetingTranscriber:
    """
    Factory pour créer un transcripteur meeting.

    Args:
        model_size: Taille du modèle Whisper
        **kwargs: Paramètres de configuration additionnels

    Returns:
        MeetingTranscriber: Instance configurée pour meetings
    """
    config = MeetingTranscriptionConfig(model_size=model_size, **kwargs)
    return MeetingTranscriber(config)

def transcribe_meeting_audio(audio_path: Union[str, Path],
                           model_size: str = "base",
                           **config_kwargs) -> Dict:
    """
    Helper function pour transcription rapide de meeting.

    Args:
        audio_path: Chemin vers le fichier audio
        model_size: Taille du modèle Whisper
        **config_kwargs: Paramètres de configuration

    Returns:
        Dict: Résultats de transcription meeting
    """
    transcriber = create_meeting_transcriber(model_size, **config_kwargs)
    return transcriber.transcribe_meeting(audio_path)
