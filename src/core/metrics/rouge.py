
"""
ROUGE-L Calculator pour évaluation de main_extractor.py en fallback du LLM juge.
Utilise HuggingFace evaluate pour ROUGE-L (Longest Common Subsequence)
"""
import time
import logging
from typing import Optional
from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)

class ROUGECalculator(BaseMetric):
    """
    Initialise le calculateur Rouge-L

    ROUGE-L mesure la similarité entre deux textes en s’appuyant sur la plus longue sous-séquence commune (LCS).
    Elle est particulièrement utilisée pour évaluer la qualité d’un résumé ou d’une extraction automatique,
    car elle valorise les séquences d’informations conservées dans le bon ordre.
    """

    def __init__(self, business_vocab = None):
        """
        Initialise Rouge-L

        Args:
            business_vocab: Vocabulaire métier optionnel"""
        super().__init__(business_vocab)

        self._rouge_metric = None

    def get_metric_name(self) -> str:
            return "rouge"

    def get_rouge_metric(self):
            """Charge la métrique ROUGE de HuggingFace."""
            if self._rouge_metric is None:
                try:
                    import evaluate
                    self._rouge_metric = evaluate.load("rouge")
                    logger.debug("✅ Métrique ROUGE-L chargée")
                except ImportError:
                    raise ImportError("pip install evaluate required pour ROUGE")
                except Exception as e:
                    logger.error(f"❌ Erreur chargement ROUGE: {e}")
                    raise RuntimeError(f"Erreur chargement ROUGE: {e}")

            return self._rouge_metric

    def calculate(self, reference: str,
                      hypothesis: str, **kwargs) -> MetricResult:

            """
            Calcule ROUGE-L entre référence et hypothèse.

            Args:
                reference: Texte de référence (extraction attendue)
                hypothesis: Texte d'hypothèse (extraction LLM)

            Retourne:
                MetricResult: Score ROUGE-L (0.0-1.0, plus haut = mieux)
            """
            start_time = time.time()

            try:
                rouge_metric = self.get_rouge_metric()

                # Calcul du Rouge-L score
                results = rouge_metric.compute(
                    predictions=[hypothesis]
                    ,references=[reference]
                    ,rouge_types=["rougeL"]
                )

                rouge_l_score = results["rougeL"]

                #   Stats business optionnelles
                business_stats = self._get_business_rouge_stats(
                    reference, hypothesis, rouge_l_score
                ) if self.business_vocab else {}

                #  Construction résultat AWS-safe
                result = MetricResult()
                result.metric_name = self._get_metric_name()
                result.score = rouge_l_score
                result.processing_time = time.time() - start_time
                result.business_focused = bool(self.business_vocab)
                result.details = {
                    "rouge-type":"rougeL"
                    ,"implementation": "huggingface_evaluate"
                    ,"words_ref": len(reference.split())
                    ,"words_hyp": len(hypothesis.split())
                    ,**business_stats
                }

                return result

            except Exception as e:
                logger.error(f"❌ Erreur calcul ROUGE: {e}")
                return self.create_error_result(e)

    def _get_business_rouge_stats(self, reference: str, hypothesis: str, rouge_score: float) -> dict:
        """
        Analyse l'impact business du score ROUGE.

        Args:
            reference: Texte de référence
            hypothesis: Texte d'hypothèse
            rouge_score: Score ROUGE calculé

        Retourne:
            dict: Statistiques business ROUGE
        """
        if not self.business_vocab:
            return {}

        # Mots business dans chaque texte
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())

        ref_business = ref_words & self.business_vocab
        hyp_business = hyp_words & self.business_vocab

        # Préservation vocabulaire business dans l'extraction
        business_overlap = len(ref_business & hyp_business)
        business_total = len(ref_business)

        return {
            "business_preservation": business_overlap / business_total if business_total > 0 else 1.0
            ,"business_words_preserved": list(ref_business & hyp_business)[:3]
            ,"rouge_quality_grade": "A" if rouge_score > 0.6 else "B" if rouge_score > 0.4 else "C"
            ,"extraction_quality": "high" if rouge_score > 0.5 else "medium"
        }
