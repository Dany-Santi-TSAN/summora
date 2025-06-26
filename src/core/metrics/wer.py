"""
wer.py - Calcul de la métrique WER avec adaptation métier.

Ce module implémente `BusinessWERCalculator`, une version enrichie de la Word Error Rate (WER)
incluant :
- un WER global basé sur Levenshtein,
- un WER orienté business (focalisé sur un vocabulaire métier),
- une analyse de préservation des mots clés,
- un WER par catégorie de mots métiers.

Il permet de mesurer l’impact des erreurs de transcription sur les éléments les plus critiques
du discours (enjeux métiers).
"""

import time
from typing import List, Dict
from .base import BaseMetric, MetricResult
from dataclasses import dataclass


@dataclass
class WERResult(MetricResult):
    """Résultat WER spécialisé."""
    global_wer: float
    business_wer: float
    business_preservation_rate: float
    category_wer: Dict[str, float]
    missing_keywords: List[str]
    correctly_transcribed: List[str]

class BusinessWERCalculator(BaseMetric):
    """Calculateur WER business-specific."""

    def __init__(self, business_vocab: set, business_keywords: Dict[str, List[str]]):
        super().__init__(business_vocab)
        self.business_keywords = business_keywords

    def get_metric_name(self) -> str:
        return "business_wer"

    def calculate(self, reference: str, hypothesis: str, **kwargs) -> WERResult:
        """Calcule WER global + business-specific."""
        start_time = time.time()

        if not self.validate_inputs(reference, hypothesis):
            raise ValueError("Référence et hypothesis requis")

        # Nettoyage textes
        ref_clean = self._clean_text(reference)
        hyp_clean = self._clean_text(hypothesis)

        # WER global
        global_wer = self._calculate_levenshtein_wer(ref_clean, hyp_clean)

        # Extraction mots business
        ref_business = self._extract_business_words(ref_clean)
        hyp_business = self._extract_business_words(hyp_clean)

        # WER business
        business_wer = self._calculate_levenshtein_wer(
            ' '.join(ref_business), ' '.join(hyp_business)
        )

        # Analyse par catégorie
        category_wer = self._calculate_category_wer(ref_clean, hyp_clean)

        # Préservation + mots manqués/corrects
        preservation_rate = len(hyp_business) / max(len(ref_business), 1)
        missing = [w for w in ref_business if w not in hyp_business]
        correct = [w for w in ref_business if w in hyp_business]

        processing_time = time.time() - start_time

        return WERResult(
            metric_name="business_wer",
            score=business_wer,
            processing_time=processing_time,
            business_focused=True,
            details={
                "global_wer": global_wer,
                "preservation_rate": preservation_rate,
                "category_breakdown": category_wer
            },
            global_wer=global_wer,
            business_wer=business_wer,
            business_preservation_rate=preservation_rate,
            category_wer=category_wer,
            missing_keywords=missing,
            correctly_transcribed=correct
        )

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte pour comparaison."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\']', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _extract_business_words(self, text: str) -> List[str]:
        """Extrait mots business du texte."""
        words = text.split()
        return [word for word in words if word in self.business_vocab]

    def _calculate_levenshtein_wer(self, reference: str, hypothesis: str) -> float:
        """Distance de Levenshtein pour WER."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        # Matrice distance Levenshtein
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,    # Deletion
                        d[i][j-1] + 1,    # Insertion
                        d[i-1][j-1] + 1   # Substitution
                    )

        return min(d[len(ref_words)][len(hyp_words)] / len(ref_words), 1.0)

    def _calculate_category_wer(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """WER par catégorie business."""
        category_wer = {}

        for category, keywords in self.business_keywords.items():
            ref_cat = self._extract_category_words(reference, keywords)
            hyp_cat = self._extract_category_words(hypothesis, keywords)

            if ref_cat:
                wer = self._calculate_levenshtein_wer(' '.join(ref_cat), ' '.join(hyp_cat))
                category_wer[category] = wer
            else:
                category_wer[category] = 0.0

        return category_wer

    def _extract_category_words(self, text: str, keywords: List[str]) -> List[str]:
        """Extrait mots d'une catégorie."""
        words = text.split()
        keywords_lower = [kw.lower() for kw in keywords]
        return [word for word in words if word in keywords_lower]
