"""
base.py - Définition des classes de base pour les métriques d’évaluation. Version AWS-compatible

Ce module contient les structures fondamentales utilisées par toutes les métriques :
- `MetricResult`, une classe générique contenant les résultats d’une métrique.
- `BaseMetric`, une interface abstraite que toutes les métriques personnalisées doivent implémenter.

Ces abstractions garantissent une architecture modulaire et extensible, adaptée à des évaluations
classiques ou orientées business (vocabulaires spécifiques, notions métiers, etc.).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class MetricResult:
    """Résultat générique pour toutes les métriques - AWS-compatible."""
    metric_name: str = "unknown"
    score: float = 0.0
    processing_time: float = 0.0
    business_focused: bool = True
    details: Dict[str, Any] = None

    def __post_init__(self):
        """Initialise les champs optionnels."""
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire pour serialization."""
        return {
            "metric_name": self.metric_name
            ,"score": self.score
            ,"processing_time": self.processing_time
            ,"business_focused": self.business_focused
            ,"details": self.details
        }

    def get_display_score(self) -> str:
        """Score formaté pour affichage."""
        if self.metric_name in ["wer", "cer"]:
            # WER/CER : plus bas = mieux
            return f"{self.score:.3f} (↓)"
        else:
            # BERTScore/SemDist : plus haut = mieux
            return f"{self.score:.3f} (↑)"

    def get_grade(self) -> str:
        """Grade simple basé sur le score."""
        if self.metric_name in ["wer", "cer"]:
            # Erreur rates : inversé
            if self.score <= 0.1:
                return "A+"
            elif self.score <= 0.2:
                return "A"
            elif self.score <= 0.3:
                return "B"
            elif self.score <= 0.5:
                return "C"
            else:
                return "D"
        else:
            # Similarity scores
            if self.score >= 0.9:
                return "A+"
            elif self.score >= 0.8:
                return "A"
            elif self.score >= 0.7:
                return "B"
            elif self.score >= 0.6:
                return "C"
            else:
                return "D"

class BaseMetric(ABC):
    """Interface abstraite pour toutes les métriques - AWS-compatible."""

    def __init__(self, business_vocab: Optional[set] = None):
        """
        Initialise la métrique avec vocabulaire business optionnel.

        Args:
            business_vocab: Set de mots-clés business pour analyse spécialisée
        """
        self.business_vocab = business_vocab or set()
        self.metric_name = self.get_metric_name()

    @abstractmethod
    def calculate(self, reference: str, hypothesis: str, **kwargs) -> MetricResult:
        """
        Calcule la métrique entre référence et hypothèse.

        Args:
            reference: Texte de référence (ground truth)
            hypothesis: Texte transcrit à évaluer
            **kwargs: Paramètres additionnels

        Retourne:
            MetricResult: Résultat avec score et métadonnées
        """
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """
        Retourne le nom unique de la métrique.

        Retourne:
            str: Nom de la métrique (ex: "wer", "cer", "bertscore","semdist", "per")
        """
        pass

    def validate_inputs(self, reference: str, hypothesis: str) -> bool:
        """
        Valide les entrées avant calcul de métrique.

        Args:
            reference: Texte de référence
            hypothesis: Texte hypothèse

        Retourne:
            bool: True si entrées valides
        """
        return bool(reference and hypothesis and
                   isinstance(reference, str) and
                   isinstance(hypothesis, str))

    def create_error_result(self, error: Exception) -> MetricResult:
        """
        Crée un résultat d'erreur standardisé.

        Args:
            error: Exception rencontrée

        Retourne:
            MetricResult: Résultat avec erreur documentée
        """
        # Score par défaut selon type de métrique
        default_score = 1.0 if self.metric_name in ["wer", "cer"] else 0.0

        # Construction AWS-safe avec instance puis modification
        result = MetricResult()
        result.metric_name = self.metric_name
        result.score = default_score
        result.processing_time = 0.0
        result.business_focused = False
        result.details = {
            "error": str(error),
            "error_type": type(error).__name__,
            "status": "calculation_failed"
        }
        return result

    def get_business_vocab_size(self) -> int:
        """Retourne la taille du vocabulaire business."""
        return len(self.business_vocab)

    def is_business_word(self, word: str) -> bool:
        """Vérifie si un mot est dans le vocabulaire business."""
        return word.lower() in self.business_vocab

    def __str__(self) -> str:
        """Représentation string de la métrique."""
        vocab_info = f" (vocab: {len(self.business_vocab)} mots)" if self.business_vocab else ""
        return f"{self.metric_name.upper()}Calculator{vocab_info}"

    def __repr__(self) -> str:
        """Représentation détaillée de la métrique."""
        return f"{self.__class__.__name__}(metric_name='{self.metric_name}', business_vocab_size={len(self.business_vocab)})"
