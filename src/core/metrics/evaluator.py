"""
Orchestrateur d'évaluation modulaire pour Summora.
Architecture séparée : Primary + Experimental + Aggregator.
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .base import MetricResult
from .wer import WERCalculator
from .cer import CERCalculator
from .bertscore import BERTScoreCalculator
from .per import PERCalculator
from .semdist import SemDistCalculator

logger = logging.getLogger(__name__)

@dataclass
class EvaluationReport:
    """Rapport d'évaluation complet Primary + Experimental."""

    # Primary Metrics (Production)
    wer_result: MetricResult = None
    cer_result: MetricResult = None
    bert_result: MetricResult = None

    # Experimental Metrics (R&D)
    per_result: MetricResult = None
    semdist_result: MetricResult = None

    # Scores consolidés
    primary_composite_score: float = 0.0
    experimental_composite_score: float = 0.0
    overall_grade: str = "D"

    # Méta
    recommendations: List[str] = None
    processing_time_total: float = 0.0
    metrics_used: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metrics_used is None:
            self.metrics_used = []

class PrimaryMetricsEvaluator:
    """
    Évaluateur Primary Metrics - Production ready.
    CER + WER + BERTScore pour usage professionnel.
    """

    def __init__(self, business_vocab=None):
        """
        Initialise l'évaluateur Primary.

        Args:
            business_vocab: Vocabulaire métier optionnel
        """
        self.business_vocab = business_vocab or set()

        # Primary Metrics
        self.wer_calculator = WERCalculator(business_vocab)
        self.cer_calculator = CERCalculator(business_vocab)
        self.bert_calculator = BERTScoreCalculator(business_vocab)

        logger.info("📊 PrimaryMetricsEvaluator initialisé (CER + WER + BERT)")

    def evaluate(self, reference: str, hypothesis: str) -> Dict[str, MetricResult]:
        """
        Évaluation des métriques Primary.

        Args:
            reference: Texte de référence (ground truth)
            hypothesis: Texte transcrit (Whisper)

        Retourne:
            Dict: Résultats des métriques primary
        """
        logger.debug("📊 Évaluation Primary Metrics")

        results = {}

        # CER
        try:
            results['cer'] = self.cer_calculator.calculate(reference, hypothesis)
            logger.debug(f"✅ CER: {results['cer'].score:.3f}")
        except Exception as e:
            logger.error(f"❌ Erreur CER: {e}")
            results['cer'] = self._error_result("cer", e)

        # WER
        try:
            results['wer'] = self.wer_calculator.calculate(reference, hypothesis)
            logger.debug(f"✅ WER: {results['wer'].score:.3f}")
        except Exception as e:
            logger.error(f"❌ Erreur WER: {e}")
            results['wer'] = self._error_result("wer", e)

        # BERTScore
        try:
            results['bert'] = self.bert_calculator.calculate(reference, hypothesis)
            logger.debug(f"✅ BERT: {results['bert'].score:.3f}")
        except Exception as e:
            logger.error(f"❌ Erreur BERT: {e}")
            results['bert'] = self._error_result("bert", e)

        return results

    def calculate_composite_score(self, results: Dict[str, MetricResult]) -> float:
        """
        Calcule score composite Primary.

        Pondération: CER 40% + WER 40% + BERT 20%
        """
        scores = []
        weights = []

        # CER (inversé car erreur)
        if 'cer' in results and results['cer'].score is not None:
            scores.append(1.0 - min(1.0, results['cer'].score))
            weights.append(0.4)

        # WER (inversé car erreur)
        if 'wer' in results and results['wer'].score is not None:
            scores.append(1.0 - min(1.0, results['wer'].score))
            weights.append(0.4)

        # BERT (score direct)
        if 'bert' in results and results['bert'].score is not None:
            scores.append(results['bert'].score)
            weights.append(0.2)

        if not scores:
            return 0.0

        # Moyenne pondérée normalisée
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _error_result(self, metric_name: str, error: Exception) -> MetricResult:
        """Crée un MetricResult d'erreur standardisé."""
        result = MetricResult()
        result.metric_name = metric_name
        result.score = 1.0  # Score d'erreur max
        result.processing_time = 0.0
        result.business_focused = bool(self.business_vocab)
        result.details = {"error": str(error)}
        return result

class ExperimentalMetricsEvaluator:
    """
    Évaluateur Experimental Metrics - R&D.
    PER + SemDist pour recherche et optimisation.
    """

    def __init__(self, business_vocab=None):
        """
        Initialise l'évaluateur Experimental.

        Args:
            business_vocab: Vocabulaire métier optionnel
        """
        self.business_vocab = business_vocab or set()

        # Experimental Metrics
        self.per_calculator = PERCalculator(business_vocab)
        self.semdist_calculator = SemDistCalculator(business_vocab)

        logger.info("🧪 ExperimentalMetricsEvaluator initialisé (PER + SemDist)")

    def evaluate(self, reference: str, hypothesis: str) -> Dict[str, MetricResult]:
        """
        Évaluation des métriques Experimental.

        Args:
            reference: Texte de référence (ground truth)
            hypothesis: Texte transcrit (Whisper)

        Retourne:
            Dict: Résultats des métriques experimental
        """
        logger.debug("🧪 Évaluation Experimental Metrics")

        results = {}

        # PER
        try:
            results['per'] = self.per_calculator.calculate(reference, hypothesis)
            logger.debug(f"✅ PER: {results['per'].score:.3f}")
        except Exception as e:
            logger.error(f"❌ Erreur PER: {e}")
            results['per'] = self._error_result("per", e)

        # SemDist
        try:
            results['semdist'] = self.semdist_calculator.calculate(reference, hypothesis)
            logger.debug(f"✅ SemDist: {results['semdist'].score:.3f}")
        except Exception as e:
            logger.error(f"❌ Erreur SemDist: {e}")
            results['semdist'] = self._error_result("semdist", e)

        return results

    def calculate_composite_score(self, results: Dict[str, MetricResult]) -> float:
        """
        Calcule score composite Experimental.

        Pondération: PER 50% + SemDist 50%
        """
        scores = []
        weights = []

        # PER (inversé car erreur)
        if 'per' in results and results['per'].score is not None:
            scores.append(1.0 - min(1.0, results['per'].score))
            weights.append(0.5)

        # SemDist (inversé car distance)
        if 'semdist' in results and results['semdist'].score is not None:
            scores.append(1.0 - min(1.0, results['semdist'].score))
            weights.append(0.5)

        if not scores:
            return 0.0

        # Moyenne pondérée
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _error_result(self, metric_name: str, error: Exception) -> MetricResult:
        """Crée un MetricResult d'erreur standardisé."""
        result = MetricResult()
        result.metric_name = metric_name
        result.score = 1.0  # Score d'erreur max
        result.processing_time = 0.0
        result.business_focused = bool(self.business_vocab)
        result.details = {"error": str(error)}
        return result

class SummoraEvaluator:
    """
    Aggregateur Principal - Combine Primary + Experimental.
    Interface unifiée pour évaluation complète.
    """

    def __init__(self, business_vocab=None):
        """
        Initialise l'aggregateur Summora.

        Args:
            business_vocab: Vocabulaire métier optionnel
        """
        self.business_vocab = business_vocab or set()

        # Évaluateurs spécialisés
        self.primary_evaluator = PrimaryMetricsEvaluator(business_vocab)
        self.experimental_evaluator = ExperimentalMetricsEvaluator(business_vocab)

        logger.info("🎯 SummoraEvaluator initialisé (Primary + Experimental)")

    def evaluate_complete(self, reference: str, hypothesis: str,
                         include_experimental: bool = True) -> EvaluationReport:
        """
        Évaluation complète Primary + Experimental.

        Args:
            reference: Texte de référence (ground truth)
            hypothesis: Texte transcrit (Whisper)
            include_experimental: Inclure métriques experimental

        Retourne:
            EvaluationReport: Rapport complet d'évaluation
        """
        start_time = time.time()

        logger.info("🔬 Évaluation complète Summora")

        # Primary Metrics
        primary_results = self.primary_evaluator.evaluate(reference, hypothesis)
        primary_score = self.primary_evaluator.calculate_composite_score(primary_results)

        # Experimental Metrics (optionnel)
        experimental_results = {}
        experimental_score = 0.0
        if include_experimental:
            experimental_results = self.experimental_evaluator.evaluate(reference, hypothesis)
            experimental_score = self.experimental_evaluator.calculate_composite_score(experimental_results)

        # Grade global
        overall_grade = self._calculate_overall_grade(primary_score, experimental_score)

        # Recommandations
        recommendations = self._generate_recommendations(
            primary_results, experimental_results, primary_score
        )

        # Métriques utilisées
        metrics_used = list(primary_results.keys())
        if include_experimental:
            metrics_used.extend(experimental_results.keys())

        total_time = time.time() - start_time

        return EvaluationReport(
            # Primary
            wer_result=primary_results.get('wer'),
            cer_result=primary_results.get('cer'),
            bert_result=primary_results.get('bert'),
            # Experimental
            per_result=experimental_results.get('per'),
            semdist_result=experimental_results.get('semdist'),
            # Consolidé
            primary_composite_score=primary_score,
            experimental_composite_score=experimental_score,
            overall_grade=overall_grade,
            recommendations=recommendations,
            processing_time_total=total_time,
            metrics_used=metrics_used
        )

    def _calculate_overall_grade(self, primary_score: float, experimental_score: float) -> str:
        """Calcule grade global basé sur primary + experimental."""

        # Grade basé principalement sur primary (70%) + experimental (30%)
        if experimental_score > 0:
            overall_score = (primary_score * 0.7) + (experimental_score * 0.3)
        else:
            overall_score = primary_score

        # Grading
        if overall_score >= 0.90:
            return "A+"
        elif overall_score >= 0.80:
            return "A"
        elif overall_score >= 0.70:
            return "B"
        elif overall_score >= 0.60:
            return "C"
        else:
            return "D"

    def _generate_recommendations(self, primary_results: Dict,
                                experimental_results: Dict,
                                primary_score: float) -> List[str]:
        """Génère recommandations basées sur toutes les métriques."""
        recommendations = []

        # Analyse Primary
        if 'cer' in primary_results:
            cer_score = primary_results['cer'].score
            if cer_score > 0.3:
                recommendations.append("🔴 CER élevé (>30%): problèmes majeurs transcription")
            elif cer_score > 0.15:
                recommendations.append("🟡 CER moyen: optimiser qualité audio")

        if 'wer' in primary_results:
            wer_score = primary_results['wer'].score
            if wer_score > 0.4:
                recommendations.append("🔴 WER élevé (>40%): modèle Whisper inadapté")
            elif wer_score > 0.2:
                recommendations.append("🟡 WER moyen: considérer Whisper plus large")

        # Analyse Experimental
        if 'per' in experimental_results:
            per_score = experimental_results['per'].score
            if per_score < 0.2:
                recommendations.append("🟢 PER excellent: phonétique préservée")
            elif per_score > 0.5:
                recommendations.append("🔍 PER élevé: erreurs phonétiques importantes")

        if 'semdist' in experimental_results:
            semdist_score = experimental_results['semdist'].score
            if semdist_score < 0.3:
                recommendations.append("🟢 SemDist excellent: sens préservé")
            elif semdist_score > 0.6:
                recommendations.append("🔍 SemDist élevé: sens altéré")

        # Recommandation globale
        if primary_score < 0.6:
            recommendations.append("💡 Score primary faible: optimisation majeure requise")
        elif primary_score >= 0.8:
            recommendations.append("🚀 Score primary excellent: prêt production")

        return recommendations or ["Qualité globale satisfaisante"]

    def compare_models(self, reference: str, transcriptions: Dict[str, str],
                      include_experimental: bool = True) -> Dict[str, EvaluationReport]:
        """Compare plusieurs modèles Whisper."""
        logger.info(f"🔀 Comparaison {len(transcriptions)} modèles")

        reports = {}
        for model_name, transcription in transcriptions.items():
            logger.info(f"📊 Évaluation {model_name}...")
            reports[model_name] = self.evaluate_complete(
                reference, transcription, include_experimental
            )

        return reports

    def get_best_model(self, comparison_reports: Dict[str, EvaluationReport]) -> tuple[str, float]:
        """Détermine le meilleur modèle basé sur score primary."""
        if not comparison_reports:
            return "unknown", 0.0

        best_model, best_report = max(
            comparison_reports.items(),
            key=lambda x: x[1].primary_composite_score
        )

        best_score = best_report.primary_composite_score
        logger.info(f"🏆 Meilleur: {best_model} (primary: {best_score:.3f})")

        return best_model, best_score

# === Factory function ===

def create_summora_evaluator(business_vocab: Optional[set] = None) -> SummoraEvaluator:
    """
    Factory pour créer évaluateur Summora complet.

    Args:
        business_vocab: Vocabulaire métier (défaut: meeting)

    Retourne:
        SummoraEvaluator: Instance configurée
    """
    if business_vocab is None:
        # Vocabulaire par défaut meeting
        business_vocab = {
            'budget', 'planning', 'action', 'decision', 'meeting',
            'deadline', 'validation', 'objectif', 'strategie'
        }

    return SummoraEvaluator(business_vocab)
