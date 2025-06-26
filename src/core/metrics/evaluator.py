"""
Orchestrateur d’évaluation pour l’analyse multi-métriques en NLP.

Ce module centralise la logique d’évaluation via une interface unifiée
(BaseMetric) pour chaque métrique implémentée (WER, ROUGE, à venir etc.).

Fonctionnalités :
- Appel centralisé des métriques
- Agrégation des résultats
- Couches d’interprétation adaptées aux besoins métiers

Modulaire, extensible, et compatible avec les sorties NLP extractives ou génératives.
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .base import BaseMetric, MetricResult
from .wer import BusinessWERCalculator
from .rouge import BusinessROUGECalculator

@dataclass
class EvaluationReport:
    """Rapport d'évaluation WER + ROUGE."""
    wer_result: MetricResult
    rouge_result: MetricResult
    composite_score: float
    business_quality_grade: str
    recommendations: List[str]
    processing_time_total: float

class BusinessMetricsEvaluator:
    """Orchestrateur WER + ROUGE pour évaluation qualité business."""

    def __init__(self, business_vocab: set, business_keywords: Dict[str, List[str]]):
        self.business_vocab = business_vocab
        self.business_keywords = business_keywords
        self.logger = logging.getLogger(__name__)

        # Initialisation calculateurs essentiels
        self.wer_calculator = BusinessWERCalculator(business_vocab, business_keywords)
        self.rouge_calculator = BusinessROUGECalculator(business_vocab)

        self.logger.info("🎯 Evaluator WER+ROUGE initialisé")

    def evaluate(self, reference: str, hypothesis: str,
                include_rouge: bool = True) -> EvaluationReport:
        """
        Évaluation WER + ROUGE pour qualité transcription.

        Args:
            reference: Texte de référence (ground truth)
            hypothesis: Texte généré (Whisper)
            include_rouge: Inclure ROUGE (True par défaut)

        Returns:
            EvaluationReport: Rapport avec WER + ROUGE
        """
        start_time = time.time()

        self.logger.info("🔬 Évaluation WER + ROUGE business")

        # Calcul WER (obligatoire)
        try:
            wer_result = self.wer_calculator.calculate(reference, hypothesis)
            self.logger.info(f"✅ WER: {wer_result.score:.3f} ({wer_result.processing_time:.2f}s)")
        except Exception as e:
            self.logger.error(f"❌ Erreur WER: {e}")
            wer_result = MetricResult(
                metric_name="business_wer",
                score=1.0,  # WER max en cas d'erreur
                processing_time=0.0,
                details={"error": str(e)},
                business_focused=True
            )

        # Calcul ROUGE (optionnel)
        if include_rouge:
            try:
                rouge_result = self.rouge_calculator.calculate(reference, hypothesis)
                self.logger.info(f"✅ ROUGE: {rouge_result.score:.3f} ({rouge_result.processing_time:.2f}s)")
            except Exception as e:
                self.logger.error(f"❌ Erreur ROUGE: {e}")
                rouge_result = MetricResult(
                    metric_name="business_rouge",
                    score=0.0,
                    processing_time=0.0,
                    details={"error": str(e)},
                    business_focused=True
                )
        else:
            # ROUGE désactivé
            rouge_result = MetricResult(
                metric_name="business_rouge",
                score=0.0,
                processing_time=0.0,
                details={"status": "disabled"},
                business_focused=True
            )

        # Score composite WER + ROUGE
        composite_score = self._calculate_composite_score(wer_result, rouge_result, include_rouge)

        # Grade qualité
        grade = self._calculate_quality_grade(composite_score, wer_result)

        # Recommandations
        recommendations = self._generate_recommendations(wer_result, rouge_result, composite_score)

        total_time = time.time() - start_time

        return EvaluationReport(
            wer_result=wer_result,
            rouge_result=rouge_result,
            composite_score=composite_score,
            business_quality_grade=grade,
            recommendations=recommendations,
            processing_time_total=total_time
        )

    def _calculate_composite_score(self, wer_result: MetricResult, rouge_result: MetricResult,
                                 include_rouge: bool) -> float:
        """
        Calcule score composite WER + ROUGE.

        Pondération:
        - WER: 70% (métrique principale)
        - ROUGE: 30% (métrique secondaire)
        """
        # WER inversé pour score positif (1 = parfait, 0 = mauvais)
        if hasattr(wer_result, 'business_wer'):
            wer_score = 1 - wer_result.business_wer
        else:
            wer_score = 1 - wer_result.score

        if include_rouge and rouge_result.score > 0:
            # Score pondéré WER 70% + ROUGE 30%
            composite = (wer_score * 0.7) + (rouge_result.score * 0.3)
        else:
            # WER uniquement
            composite = wer_score

        return max(0.0, min(1.0, composite))  # Clamp entre 0 et 1

    def _calculate_quality_grade(self, composite_score: float, wer_result: MetricResult) -> str:
        """Calcule grade qualité business basé sur score composite."""

        # Grading adapté business
        if composite_score >= 0.85:
            return "A+"  # Excellent
        elif composite_score >= 0.75:
            return "A"   # Très bon
        elif composite_score >= 0.65:
            return "B"   # Bon
        elif composite_score >= 0.50:
            return "C"   # Acceptable
        else:
            return "D"   # Insuffisant

    def _generate_recommendations(self, wer_result: MetricResult, rouge_result: MetricResult,
                                composite_score: float) -> List[str]:
        """Génère recommandations basées sur WER + ROUGE."""
        recommendations = []

        # Analyse WER (priorité)
        if hasattr(wer_result, 'business_wer'):
            business_wer = wer_result.business_wer

            if business_wer > 0.5:
                recommendations.append("WER business très élevé (>50%): modèle Whisper inadapté, passer à Large")
            elif business_wer > 0.3:
                recommendations.append("WER business élevé (>30%): considérer modèle Whisper plus performant")
            elif business_wer > 0.15:
                recommendations.append("WER business moyen: optimiser prompt ou qualité audio")
            else:
                recommendations.append("WER business excellent: qualité de transcription optimale")

            # Préservation vocabulaire business
            if hasattr(wer_result, 'business_preservation_rate'):
                preservation = wer_result.business_preservation_rate
                if preservation < 0.6:
                    recommendations.append("Préservation vocabulaire faible (<60%): améliorer contexte métier")
                elif preservation < 0.8:
                    recommendations.append("Préservation vocabulaire moyenne: optimiser prompt business")

        # Analyse ROUGE (secondaire)
        if rouge_result.score > 0:  # ROUGE activé
            if rouge_result.score < 0.4:
                recommendations.append("Score ROUGE faible (<40%): problème cohérence du contenu")
            elif rouge_result.score < 0.6:
                recommendations.append("Score ROUGE moyen: améliorer structure du discours")

        # Évaluation globale
        if composite_score < 0.5:
            recommendations.append("🔴 Qualité globale insuffisante: pipeline nécessite optimisation majeure")
        elif composite_score < 0.7:
            recommendations.append("🟡 Qualité moyenne: améliorations recommandées pour usage professionnel")
        elif composite_score >= 0.8:
            recommendations.append("🟢 Excellente qualité: pipeline prêt pour production business")

        return recommendations if recommendations else ["Qualité satisfaisante pour usage courant"]

    def compare_models(self, reference: str, transcriptions: Dict[str, str],
                      include_rouge: bool = True) -> Dict[str, EvaluationReport]:
        """
        Compare plusieurs transcriptions (modèles Whisper différents).

        Args:
            reference: Texte de référence
            transcriptions: Dict {model_name: transcription}
            include_rouge: Inclure ROUGE dans comparaison

        Returns:
            Dict: Rapports d'évaluation par modèle
        """
        self.logger.info(f"🔀 Comparaison {len(transcriptions)} modèles Whisper")

        reports = {}
        for model_name, transcription in transcriptions.items():
            self.logger.info(f"📊 Évaluation {model_name}...")
            reports[model_name] = self.evaluate(reference, transcription, include_rouge)

        return reports

    def get_best_model(self, comparison_reports: Dict[str, EvaluationReport]) -> tuple[str, float]:
        """
        Détermine le meilleur modèle basé sur score composite.

        Args:
            comparison_reports: Rapports de comparaison

        Returns:
            tuple: (nom_meilleur_modèle, score_composite)
        """
        if not comparison_reports:
            return "unknown", 0.0

        best_model, best_report = max(comparison_reports.items(),
                                     key=lambda x: x[1].composite_score)

        best_score = best_report.composite_score
        self.logger.info(f"🏆 Meilleur modèle: {best_model} (score: {best_score:.3f}, grade: {best_report.business_quality_grade})")

        return best_model, best_score

    def get_wer_only(self, reference: str, hypothesis: str) -> float:
        """
        Évaluation WER uniquement (rapide).

        Args:
            reference: Texte de référence
            hypothesis: Texte transcrit

        Returns:
            float: WER business (0.0 = parfait, 1.0 = totalement faux)
        """
        try:
            wer_result = self.wer_calculator.calculate(reference, hypothesis)
            if hasattr(wer_result, 'business_wer'):
                return wer_result.business_wer
            else:
                return wer_result.score
        except Exception as e:
            self.logger.error(f"❌ Erreur WER rapide: {e}")
            return 1.0  # WER maximum en cas d'erreur

# === Factory function ===

def create_business_evaluator(business_vocab: Optional[set] = None,
                            business_keywords: Optional[Dict[str, List[str]]] = None) -> BusinessMetricsEvaluator:
    """
    Factory pour créer évaluateur WER+ROUGE avec vocabulaire centralisé.

    Args:
        business_vocab: Vocabulaire business (auto-import si None)
        business_keywords: Keywords par catégorie (auto-import si None)

    Returns:
        BusinessMetricsEvaluator: Instance configurée
    """
    if business_vocab is None or business_keywords is None:
        try:
            from ..business_vocabulary import get_all_business_keywords, BUSINESS_KEYWORDS
            business_vocab = business_vocab or get_all_business_keywords()
            business_keywords = business_keywords or BUSINESS_KEYWORDS
        except ImportError:
            logger = logging.getLogger(__name__)
            logger.warning("⚠️ Import vocabulaire centralisé échoué - fallback minimal")
            business_vocab = set(['action', 'décision', 'budget', 'planning'])
            business_keywords = {'actions': ['action'], 'decisions': ['décision']}

    return BusinessMetricsEvaluator(business_vocab, business_keywords)

# === Usage examples ===

def example_single_evaluation():
    """Exemple évaluation simple WER + ROUGE."""
    evaluator = create_business_evaluator()

    reference = "Nous validons le budget pour ce projet avec deadline en décembre"
    hypothesis = "Nous validons le budget pour ce projet avec échéance en décembre"

    # Évaluation complète
    report = evaluator.evaluate(reference, hypothesis)

    print("📊 ÉVALUATION WER + ROUGE")
    print(f"Score composite: {report.composite_score:.3f}")
    print(f"Grade: {report.business_quality_grade}")
    print(f"WER business: {report.wer_result.score:.3f}")
    print(f"ROUGE business: {report.rouge_result.score:.3f}")
    print(f"Recommandations: {report.recommendations}")

def example_model_comparison():
    """Exemple comparaison modèles."""
    evaluator = create_business_evaluator()

    reference = "Nous validons le budget pour ce projet avec deadline en décembre"
    transcriptions = {
        'tiny': "Nous validons le budget pour ce projet avec dead line en décembre",
        'base': "Nous validons le budget pour ce projet avec deadline en décembre",
        'small': "Nous validons le budget pour ce projet avec deadline en décembre"
    }

    reports = evaluator.compare_models(reference, transcriptions)
    best_model, best_score = evaluator.get_best_model(reports)

    print("🔀 COMPARAISON MODÈLES")
    print(f"Meilleur modèle: {best_model} ({best_score:.3f})")
    print()
    for model, report in reports.items():
        wer_score = report.wer_result.score if hasattr(report.wer_result, 'business_wer') else "N/A"
        print(f"{model:8}: {report.composite_score:.3f} ({report.business_quality_grade}) - WER: {wer_score}")

def example_wer_only():
    """Exemple WER rapide."""
    evaluator = create_business_evaluator()

    reference = "Nous validons le budget pour ce projet avec deadline en décembre"
    hypothesis = "Nous validons le budget pour ce projet avec échéance en décembre"

    wer_score = evaluator.get_wer_only(reference, hypothesis)
    print(f"WER business uniquement: {wer_score:.3f}")

if __name__ == "__main__":
    example_single_evaluation()
    print()
    example_model_comparison()
    print()
    example_wer_only()
