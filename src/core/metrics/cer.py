"""
Calculateur du taux d’erreur caractère (CER), basé sur la distance de Levenshtein.

Mesure fine des erreurs de transcription, utile pour le français où l’orthographe est riche
en lettres muettes et variantes morphologiques.

Particulièrement adapté pour évaluer des systèmes de transcription sur des langues à forte
variabilité graphique.
"""
import time
import logging
from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)

class CERCalculator(BaseMetric):
    """CER officiel HuggingFace - Meilleur que WER pour le français."""

    def __init__(self, business_vocab=None):
        super().__init__(business_vocab)
        self._cer_metric = None

    def get_metric_name(self) -> str:
        return "cer"

    def _get_cer_metric(self):
        """Lazy loading de la métrique CER."""
        if self._cer_metric is None:
            try:
                import evaluate
                self._cer_metric = evaluate.load("cer")
                logger.debug("✅ Métrique CER chargée")
            except ImportError:
                raise ImportError("pip install evaluate required")
        return self._cer_metric

    def calculate(self, reference: str, hypothesis: str, **kwargs) -> MetricResult:
        """
        Implémentation du calcul CER officiel

        Args:
            reference: Texte de référence (ground truth)
            hypothesis: Texte transcrit

        Retourne:
            MetricResult: Score CER (0.0 = parfait, 1.0+ = erreurs)
        """
        start_time = time.time()

        try:
            cer_metric = self._get_cer_metric()
            cer_score = cer_metric.compute(
                predictions=[hypothesis]
                ,references=[reference]
            )

            # Analyse business optionnelle
            business_stats = self._get_business_stats(reference, hypothesis) if self.business_vocab else {}

            # Construction AWS-compatible
            result = MetricResult()
            result.metric_name = "cer"
            result.score = cer_score
            result.processing_time = time.time() - start_time
            result.business_focused = bool(self.business_vocab)
            result.details = {
                "implementation": "huggingface_official"
                ,"characters_ref": len(reference)
                ,"characters_hyp": len(hypothesis)
                ,"hats_performance": "64% human agreement (champion français)"
                ,**business_stats
            }
            return result

        except Exception as e:
            # Construction AWS-compatible pour erreur
            result = MetricResult()
            result.metric_name = "cer"
            result.score = 1.0  # CER maximum en cas d'erreur (par défaut)
            result.processing_time = time.time() - start_time
            result.business_focused = False
            result.details = {"error": str(e)}
            return result

    def _get_business_stats(self, reference: str, hypothesis: str) -> dict:
        """Stats business pour caractères spéciaux métier."""
        if not self.business_vocab:
            return {}

        # Analyse des mots business préservés au niveau caractère
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        ref_business = [w for w in ref_words if w in self.business_vocab]
        hyp_business = [w for w in hyp_words if w in self.business_vocab]
        preserved = list(set(ref_business) & set(hyp_business))

        # Caractères business
        ref_business_chars = sum(len(word) for word in ref_business)
        hyp_business_chars = sum(len(word) for word in hyp_business)

        return {
            "business_words_ref": len(ref_business)
            ,"business_chars_ref": ref_business_chars
            ,"business_chars_hyp": hyp_business_chars
            ,"business_preservation_rate": len(preserved) / len(ref_business) if ref_business else 1.0
            ,"business_char_density": ref_business_chars / len(reference) if reference else 0.0
        }

    def compare_transcriptions(self, reference: str, transcriptions: dict) -> dict:
        """Compare plusieurs transcriptions avec CER."""
        results = {}

        for model_name, transcription in transcriptions.items():
            result = self.calculate(reference, transcription)
            results[model_name] = {
                "cer_score": result.score
                ,"processing_time": result.processing_time
                ,"details": result.details
            }

        return results
