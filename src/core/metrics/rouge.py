"""
rouge.py - Calcul de la métrique ROUGE avec focus business.

Ce module propose `BusinessROUGECalculator`, une extension de la métrique ROUGE classique
(ROUGE-1, ROUGE-2, ROUGE-L) intégrant :
- une mesure standard de similarité entre n-grammes,
- une mesure business-specific basée sur la présence de mots-clés métiers.

Utilisé pour évaluer la qualité des résumés générés, avec une attention particulière
aux éléments stratégiques à préserver pour garantir la valeur métier du contenu synthétisé.
"""


import time
from typing import Set, Tuple, List
from .base import BaseMetric, MetricResult
from dataclasses import dataclass

@dataclass
class ROUGEResult(MetricResult):
    """Résultat ROUGE spécialisé."""
    rouge_1: float
    rouge_2: float
    rouge_l: float
    business_rouge: float

class BusinessROUGECalculator(BaseMetric):
    """Calculateur ROUGE pour résumés actionables."""

    def get_metric_name(self) -> str:
        return "business_rouge"

    def calculate(self, reference: str, hypothesis: str, **kwargs) -> ROUGEResult:
        """Calcule ROUGE-1, ROUGE-2, ROUGE-L + business-specific."""
        start_time = time.time()

        if not self.validate_inputs(reference, hypothesis):
            raise ValueError("Référence et hypothesis requis")

        # ROUGE standard
        rouge_1 = self._calculate_rouge_n(reference, hypothesis, n=1)
        rouge_2 = self._calculate_rouge_n(reference, hypothesis, n=2)
        rouge_l = self._calculate_rouge_l(reference, hypothesis)

        # ROUGE business (focus mots-clés métier)
        business_rouge = self._calculate_business_rouge(reference, hypothesis)

        processing_time = time.time() - start_time

        return ROUGEResult(
            metric_name="business_rouge",
            score=business_rouge,
            processing_time=processing_time,
            business_focused=True,
            details={
                "rouge_breakdown": {
                    "rouge_1": rouge_1,
                    "rouge_2": rouge_2,
                    "rouge_l": rouge_l
                }
            },
            rouge_1=rouge_1,
            rouge_2=rouge_2,
            rouge_l=rouge_l,
            business_rouge=business_rouge
        )

    def _calculate_rouge_n(self, reference: str, hypothesis: str, n: int) -> float:
        """ROUGE-N standard."""
        ref_ngrams = self._get_ngrams(reference.split(), n)
        hyp_ngrams = self._get_ngrams(hypothesis.split(), n)

        if not ref_ngrams:
            return 0.0

        overlap = len(ref_ngrams.intersection(hyp_ngrams))
        return overlap / len(ref_ngrams)

    def _calculate_rouge_l(self, reference: str, hypothesis: str) -> float:
        """ROUGE-L (Longest Common Subsequence)."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        lcs_length = self._lcs_length(ref_words, hyp_words)

        if not ref_words:
            return 0.0

        return lcs_length / len(ref_words)

    def _calculate_business_rouge(self, reference: str, hypothesis: str) -> float:
        """ROUGE spécialisé mots business."""
        ref_business = [w for w in reference.split() if w.lower() in self.business_vocab]
        hyp_business = [w for w in hypothesis.split() if w.lower() in self.business_vocab]

        if not ref_business:
            return 0.0

        ref_set = set(ref_business)
        hyp_set = set(hyp_business)

        overlap = len(ref_set.intersection(hyp_set))
        return overlap / len(ref_set)

    def _get_ngrams(self, words: List[str], n: int) -> Set[Tuple[str, ...]]:
        """Génère n-grammes."""
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.add(ngram)
        return ngrams

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Longueur plus longue sous-séquence commune."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]
