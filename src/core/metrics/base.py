"""
base.py - Définition des classes de base pour les métriques d’évaluation.

Ce module contient les structures fondamentales utilisées par toutes les métriques :
- `MetricResult`, une classe générique contenant les résultats d’une métrique.
- `BaseMetric`, une interface abstraite que toutes les métriques personnalisées doivent implémenter.

Ces abstractions garantissent une architecture modulaire et extensible, adaptée à des évaluations
classiques ou orientées business (vocabulaires spécifiques, notions métiers, etc.).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MetricResult:
    """Résultat générique pour toutes les métriques."""
    metric_name: str
    score: float
    details: Dict[str, Any]
    processing_time: float
    business_focused: bool = True

class BaseMetric(ABC):
    """Interface abstraite pour toutes les métriques."""

    def __init__(self, business_vocab: Optional[set] = None):
        self.business_vocab = business_vocab or set()
        self.metric_name = self.get_metric_name()

    @abstractmethod
    def calculate(self, reference: str, hypothesis: str, **kwargs) -> MetricResult:
        """Calcule la métrique."""
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """Retourne le nom de la métrique."""
        pass

    def validate_inputs(self, reference: str, hypothesis: str) -> bool:
        """Valide les entrées."""
        return bool(reference and hypothesis)
