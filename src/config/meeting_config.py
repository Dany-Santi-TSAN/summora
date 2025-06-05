"""
Configuration spécialisée pour l'analyse de meetings
Paramètres optimisés pour le contexte business et réunions d'entreprise
"""
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path

@dataclass
class MeetingPipelineConfig:
    """Configuration globale pour le pipeline d'analyse de meetings."""

    # Configuration générale
    pipeline_name: str = "Summora Meeting Analyzer"
    version: str = "1.0"
    language: str = "fr"

    # Configuration Whisper pour meetings
    whisper_model: str = "base"
    whisper_temperature: float = 0.0
    whisper_initial_prompt: str = (
        "Transcription d'une réunion professionnelle en français. "
        "Participants multiples discutant d'actions, décisions, planning et objectifs. "
        "Terminologie business et technique française avec quelques termes anglais."
    )

    # Configuration extraction meeting
    extraction_methods: List[str] = None
    extract_actions: bool = True
    extract_decisions: bool = True
    extract_insights: bool = True

    # Configuration audio
    generate_audio_plots: bool = True
    min_meeting_duration: float = 60.0  # 1 minute minimum
    optimal_speech_ratio: float = 0.65   # 65% de parole idéal

    # Configuration outputs
    save_results: bool = False
    output_format: str = "meeting_summary"  # ou "executive_report", "action_list"
    include_timestamps: bool = True
    include_confidence_scores: bool = True

    # Seuils de qualité meeting
    min_confidence_threshold: float = 0.7
    min_efficiency_score: float = 0.5

    def __post_init__(self):
        """Initialisation post-création avec valeurs par défaut intelligentes."""
        if self.extraction_methods is None:
            self.extraction_methods = ['yake', 'tfidf', 'actions', 'decisions']

@dataclass
class MeetingOutputConfig:
    """Configuration pour les formats de sortie meeting."""

    # Format principal
    format_type: str = "meeting_summary"  # meeting_summary, executive_report, action_tracker

    # Sections à inclure
    include_audio_analysis: bool = True
    include_transcription_preview: bool = True
    include_key_topics: bool = True
    include_actions: bool = True
    include_decisions: bool = True
    include_insights: bool = True
    include_recommendations: bool = True

    # Paramètres d'affichage
    max_topics_display: int = 10
    max_actions_display: int = 15
    max_decisions_display: int = 10
    preview_text_length: int = 200

    # Export options
    export_to_json: bool = False
    export_to_markdown: bool = False
    export_to_notion: bool = False  # Future feature
    export_to_slack: bool = False   # Future feature

@dataclass
class MeetingQualityThresholds:
    """Seuils de qualité pour l'évaluation des meetings."""

    # Audio quality
    min_sample_rate: int = 16000
    max_silence_ratio: float = 0.5
    min_speech_ratio: float = 0.3
    min_dynamic_range: float = 15.0

    # Transcription quality
    min_whisper_confidence: float = 0.7
    min_words_per_minute: float = 80
    max_words_per_minute: float = 200

    # Content quality
    min_actionable_density: float = 0.5  # actions+décisions pour 100 mots
    min_business_relevance: float = 0.3
    min_meeting_keywords: int = 5

    # Grades mapping
    grade_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.grade_thresholds is None:
            self.grade_thresholds = {
                "A+": 0.90,
                "A": 0.80,
                "B": 0.70,
                "C": 0.60,
                "D": 0.40,
                "F": 0.0
            }

class MeetingConfigManager:
    """Gestionnaire centralisé des configurations meeting."""

    def __init__(self,
                 pipeline_config: Optional[MeetingPipelineConfig] = None,
                 output_config: Optional[MeetingOutputConfig] = None,
                 quality_thresholds: Optional[MeetingQualityThresholds] = None):
        """
        Initialise le gestionnaire de configuration.

        Args:
            pipeline_config: Configuration du pipeline
            output_config: Configuration des outputs
            quality_thresholds: Seuils de qualité
        """
        self.pipeline = pipeline_config or MeetingPipelineConfig()
        self.output = output_config or MeetingOutputConfig()
        self.quality = quality_thresholds or MeetingQualityThresholds()

    def get_whisper_config(self) -> Dict:
        """Retourne la configuration Whisper pour meetings."""
        return {
            "model_size": self.pipeline.whisper_model,
            "language": self.pipeline.language,
            "temperature": self.pipeline.whisper_temperature,
            "initial_prompt": self.pipeline.whisper_initial_prompt,
            "word_timestamps": True,
            "verbose": False
        }

    def get_extraction_config(self) -> Dict:
        """Retourne la configuration d'extraction pour meetings."""
        return {
            "enabled_methods": self.pipeline.extraction_methods,
            "extract_actions": self.pipeline.extract_actions,
            "extract_decisions": self.pipeline.extract_decisions,
            "yake_top_keywords": 20,
            "tfidf_max_features": 150,
            "action_weight": 2.0,  # Actions prioritaires
            "business_focus": True
        }

    def get_audio_config(self) -> Dict:
        """Retourne la configuration d'analyse audio pour meetings."""
        return {
            "generate_plots": self.pipeline.generate_audio_plots,
            "min_meeting_duration": self.pipeline.min_meeting_duration,
            "optimal_speech_ratio": self.pipeline.optimal_speech_ratio,
            "silence_percentile": 15.0,
            "focus_voice_frequencies": True
        }

    def get_quality_config(self) -> Dict:
        """Retourne la configuration des seuils de qualité."""
        return {
            "min_confidence": self.quality.min_whisper_confidence,
            "min_sample_rate": self.quality.min_sample_rate,
            "min_speech_ratio": self.quality.min_speech_ratio,
            "min_actionable_density": self.quality.min_actionable_density,
            "grade_thresholds": self.quality.grade_thresholds
        }

    def validate_config(self) -> List[str]:
        """
        Valide la cohérence de la configuration.

        Returns:
            List[str]: Liste des erreurs/warnings de configuration
        """
        issues = []

        # Validation pipeline
        if self.pipeline.whisper_model not in ["tiny", "base", "small", "medium", "large"]:
            issues.append(f"Modèle Whisper invalide: {self.pipeline.whisper_model}")

        if not (0.0 <= self.pipeline.whisper_temperature <= 1.0):
            issues.append(f"Température Whisper invalide: {self.pipeline.whisper_temperature}")

        # Validation seuils qualité
        if self.quality.min_speech_ratio >= self.quality.max_silence_ratio:
            issues.append("Seuils speech_ratio et silence_ratio incohérents")

        if self.quality.min_words_per_minute >= self.quality.max_words_per_minute:
            issues.append("Seuils words_per_minute incohérents")

        # Validation output
        valid_formats = ["meeting_summary", "executive_report", "action_tracker"]
        if self.output.format_type not in valid_formats:
            issues.append(f"Format de sortie invalide: {self.output.format_type}")

        return issues

    def create_specialized_config(self, meeting_type: str) -> 'MeetingConfigManager':
        """
        Crée une configuration spécialisée selon le type de meeting.

        Args:
            meeting_type: Type de meeting (sprint, budget, strategic, etc.)

        Returns:
            MeetingConfigManager: Configuration adaptée
        """
        # Configuration de base
        pipeline_config = MeetingPipelineConfig()
        output_config = MeetingOutputConfig()
        quality_config = MeetingQualityThresholds()

        # Adaptations selon le type
        if meeting_type.lower() == "sprint":
            pipeline_config.whisper_initial_prompt = (
                "Transcription de réunion agile/sprint en français. "
                "Discussion de backlog, user stories, vélocité et planning."
            )
            pipeline_config.extraction_methods = ['yake', 'tfidf', 'actions']
            output_config.include_decisions = False  # Moins de décisions en sprint

        elif meeting_type.lower() == "budget":
            pipeline_config.whisper_initial_prompt = (
                "Transcription de réunion budgétaire en français. "
                "Discussion de coûts, investissements, ROI et allocations."
            )
            pipeline_config.extraction_methods = ['yake', 'tfidf', 'decisions']
            quality_config.min_business_relevance = 0.5  # Plus strict

        elif meeting_type.lower() == "strategic":
            pipeline_config.whisper_initial_prompt = (
                "Transcription de réunion stratégique en français. "
                "Discussion de vision, objectifs, roadmap et orientations."
            )
            pipeline_config.extraction_methods = ['yake', 'tfidf', 'lda', 'decisions']
            output_config.format_type = "executive_report"

        elif meeting_type.lower() == "daily":
            pipeline_config.min_meeting_duration = 30.0  # Daily plus courts
            pipeline_config.extraction_methods = ['yake', 'actions']
            output_config.format_type = "action_tracker"
            output_config.max_topics_display = 5

        return MeetingConfigManager(pipeline_config, output_config, quality_config)

    def to_dict(self) -> Dict:
        """Exporte la configuration complète en dictionnaire."""
        return {
            "pipeline": {
                "name": self.pipeline.pipeline_name,
                "version": self.pipeline.version,
                "language": self.pipeline.language,
                "whisper": self.get_whisper_config(),
                "extraction": self.get_extraction_config(),
                "audio": self.get_audio_config()
            },
            "output": {
                "format_type": self.output.format_type,
                "sections": {
                    "audio_analysis": self.output.include_audio_analysis,
                    "transcription": self.output.include_transcription_preview,
                    "topics": self.output.include_key_topics,
                    "actions": self.output.include_actions,
                    "decisions": self.output.include_decisions,
                    "insights": self.output.include_insights
                },
                "limits": {
                    "max_topics": self.output.max_topics_display,
                    "max_actions": self.output.max_actions_display,
                    "max_decisions": self.output.max_decisions_display
                }
            },
            "quality": self.get_quality_config()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MeetingConfigManager':
        """
        Crée une configuration depuis un dictionnaire.

        Args:
            config_dict: Configuration en format dictionnaire

        Returns:
            MeetingConfigManager: Instance configurée
        """
        # Reconstruction des configs depuis le dict
        # Implementation simplifiée - peut être étendue selon besoins
        pipeline_config = MeetingPipelineConfig()
        output_config = MeetingOutputConfig()
        quality_config = MeetingQualityThresholds()

        # Application des valeurs du dictionnaire
        if "pipeline" in config_dict:
            pipeline_data = config_dict["pipeline"]
            if "whisper" in pipeline_data:
                whisper_data = pipeline_data["whisper"]
                pipeline_config.whisper_model = whisper_data.get("model_size", "base")
                pipeline_config.language = whisper_data.get("language", "fr")

        return cls(pipeline_config, output_config, quality_config)

# Configurations prédéfinies pour différents types de meetings
MEETING_CONFIGS = {
    "default": MeetingConfigManager(),
    "sprint": MeetingConfigManager().create_specialized_config("sprint"),
    "budget": MeetingConfigManager().create_specialized_config("budget"),
    "strategic": MeetingConfigManager().create_specialized_config("strategic"),
    "daily": MeetingConfigManager().create_specialized_config("daily")
}

def get_meeting_config(meeting_type: str = "default") -> MeetingConfigManager:
    """
    Retourne une configuration prédéfinie pour un type de meeting.

    Args:
        meeting_type: Type de meeting (default, sprint, budget, strategic, daily)

    Retourne:
        MeetingConfigManager: Configuration appropriée
    """
    return MEETING_CONFIGS.get(meeting_type, MEETING_CONFIGS["default"])
