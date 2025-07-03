"""
Calculateur du taux d’erreur mot (WER), métrique classique fondée sur la distance de Levenshtein au niveau lexical.

Indique le pourcentage de mots mal transcrits (substitution, insertion, suppression)
par rapport à une référence.

Utilisé comme base standard en ASR, mais à compléter par des métriques plus sensibles au sens.
AWS compatible.
"""

import time
import logging
from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)

class WERCalculator(BaseMetric):
    """WER officiel HuggingFace - Référence universelle pour ASR."""

    def __init__(self, business_vocab=None):
        super().__init__(business_vocab)
        self._wer_metric = None

    def get_metric_name(self) -> str:
        return "wer"

    def _get_wer_metric(self):
        """Lazy loading de la métrique WER."""
        if self._wer_metric is None:
            try:
                import evaluate
                self._wer_metric = evaluate.load("wer")
                logger.debug("✅ Métrique WER chargée")
            except ImportError:
                raise ImportError("pip install evaluate required")
            except Exception as e:
                logger.error(f"❌ Erreur chargement WER: {e}")
                raise RuntimeError(f"Erreur chargement WER: {e}")
        return self._wer_metric

    def calculate(self, reference: str, hypothesis: str, **kwargs) -> MetricResult:
        """
        Calcule WER officiel - Référence universelle ASR.

        Args:
            reference: Texte de référence (ground truth)
            hypothesis: Texte transcrit

        Retourne:
            MetricResult: Score WER (0.0 = parfait, 1.0+ = erreurs)
        """
        start_time = time.time()

        try:
            wer_metric = self._get_wer_metric()
            wer_score = wer_metric.compute(
                predictions=[hypothesis],
                references=[reference]
            )

            # Analyse business optionnelle
            business_stats = self._get_business_stats(reference, hypothesis) if self.business_vocab else {}

            # Construction AWS-safe
            result = MetricResult()
            result.metric_name = "wer"
            result.score = wer_score
            result.processing_time = time.time() - start_time
            result.business_focused = bool(self.business_vocab)
            result.details = {
                "implementation": "huggingface_official",
                "words_ref": len(reference.split()),
                "words_hyp": len(hypothesis.split()),
                **business_stats
            }
            return result

        except Exception as e:
            # Construction AWS-safe pour erreur
            result = MetricResult()
            result.metric_name = "wer"
            result.score = 1.0  # WER maximum en cas d'erreur
            result.processing_time = time.time() - start_time
            result.business_focused = False
            result.details = {"error": str(e)}
            return result

    def _get_business_stats(self, reference: str, hypothesis: str) -> dict:
        """Stats business pour mots-clés métier."""
        if not self.business_vocab:
            return {}

        # Analyse des mots business préservés
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        ref_business = [w for w in ref_words if w in self.business_vocab]
        hyp_business = [w for w in hyp_words if w in self.business_vocab]
        preserved = list(set(ref_business) & set(hyp_business))

        return {
            "business_words_ref": len(ref_business),
            "business_words_hyp": len(hyp_business),
            "business_preservation_rate": len(preserved) / len(ref_business) if ref_business else 1.0,
            "business_word_density": len(ref_business) / len(ref_words) if ref_words else 0.0,
            "preserved_business_words": preserved[:5]  # Top 5 pour debug
        }

    def compare_transcriptions(self, reference: str, transcriptions: dict) -> dict:
        """Compare plusieurs transcriptions avec WER."""
        results = {}

        for model_name, transcription in transcriptions.items():
            result = self.calculate(reference, transcription)
            results[model_name] = {
                "wer_score": result.score,
                "processing_time": result.processing_time,
                "details": result.details
            }

        return results
