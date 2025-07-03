"""
Calculateur du taux d'erreur phonème (PER), métrique phonétique basée sur Levenshtein.

Calcule la distance entre séquences de phonèmes de référence et hypothèse.
Inspiré de l'étude HATS - plus proche de la perception auditive que WER/CER.
Métrique experimentale.
AWS compatible.
"""

import time
import logging
from typing import List
from .base import BaseMetric, MetricResult
from .utils.utils_per import text_to_phonemes, get_phoneme_stats

logger = logging.getLogger(__name__)

class PERCalculator(BaseMetric):
    """
    PER = Taux d'erreur au niveau phonétique.

    Métrique innovante qui compare les sons plutôt que l'orthographe,
    potentiellement plus corrélée à la perception humaine selon l'étude.
    """

    def __init__(self, business_vocab=None, phoneme_method="simple"):
        """
        Initialise le calculateur PER.

        Args:
            business_vocab: Vocabulaire métier optionnel
            phoneme_method: Méthode de conversion phonétique
        """
        super().__init__(business_vocab)
        self.phoneme_method = phoneme_method

    def get_metric_name(self) -> str:
        return "per"

    def _calculate_levenshtein_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Calcule la distance de Levenshtein entre deux séquences de phonèmes.

        Args:
            seq1: Séquence de phonèmes de référence
            seq2: Séquence de phonèmes d'hypothèse

        Retourne:
            int: Distance de Levenshtein (nombre d'opérations)
        """
        if not seq1:
            return len(seq2)
        if not seq2:
            return len(seq1)

        # Matrice de programmation dynamique
        len1, len2 = len(seq1), len(seq2)
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialisation
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        # Calcul distances
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if seq1[i-1] == seq2[j-1]:
                    cost = 0
                else:
                    cost = 1

                matrix[i][j] = min(
                    matrix[i-1][j] + 1      # Suppression
                    ,matrix[i][j-1] + 1      # Insertion
                    ,matrix[i-1][j-1] + cost  # Substitution
                )

        return matrix[len1][len2]

    def calculate(self, reference: str, hypothesis: str, **kwargs) -> MetricResult:
        """
        Calcule le PER entre référence et hypothèse via phonèmes.

        Processus :
        1. Conversion texte → phonèmes
        2. Distance Levenshtein sur séquences phonétiques
        3. PER = distance / longueur_référence

        Args:
            reference: Texte de référence (ground truth)
            hypothesis: Texte transcrit à évaluer

        Retourne:
            MetricResult: Score PER (0.0 = parfait, 1.0+ = erreurs)
        """
        start_time = time.time()

        try:
            # Conversion texte → phonèmes
            ref_phonemes = text_to_phonemes(reference, self.phoneme_method)
            hyp_phonemes = text_to_phonemes(hypothesis, self.phoneme_method)

            # Distance Levenshtein sur phonèmes
            levenshtein_distance = self._calculate_levenshtein_distance(ref_phonemes, hyp_phonemes)

            # Calcul PER
            ref_length = len(ref_phonemes)
            if ref_length == 0:
                per_score = 1.0 if len(hyp_phonemes) > 0 else 0.0
            else:
                per_score = levenshtein_distance / ref_length

            # Stats phonétiques
            ref_stats = get_phoneme_stats(ref_phonemes)
            hyp_stats = get_phoneme_stats(hyp_phonemes)

            # Stats business optionnelles
            business_stats = self._get_business_phonetic_stats(
                reference, hypothesis, ref_phonemes, hyp_phonemes
            ) if self.business_vocab else {}

            # Construction résultat AWS-safe
            result = MetricResult()
            result.metric_name = "per"
            result.score = per_score
            result.processing_time = time.time() - start_time
            result.business_focused = bool(self.business_vocab)
            result.details = {
                "implementation": f"levenshtein_phonemes_{self.phoneme_method}"
                ,"levenshtein_distance": levenshtein_distance
                ,"phonemes_ref": len(ref_phonemes)
                ,"phonemes_hyp": len(hyp_phonemes)
                ,"ref_phoneme_stats": ref_stats
                ,"hyp_phoneme_stats": hyp_stats
                ,"hats_inspiration": "Phoneme Error Rate from HATS study"
                ,**business_stats
            }

            return result

        except Exception as e:
            # Résultat d'erreur AWS-safe
            result = MetricResult()
            result.metric_name = "per"
            result.score = 1.0  # PER maximum en cas d'erreur
            result.processing_time = time.time() - start_time
            result.business_focused = False
            result.details = {"error": str(e)}
            return result

    def _get_business_phonetic_stats(self, reference: str, hypothesis: str,
                                   ref_phonemes: List[str], hyp_phonemes: List[str]) -> dict:
        """
        Analyse l'impact business de la distance phonétique.

        Args:
            reference: Texte de référence
            hypothesis: Texte d'hypothèse
            ref_phonemes: Phonèmes de référence
            hyp_phonemes: Phonèmes d'hypothèse

        Retourne:
            dict: Statistiques business phonétiques
        """
        if not self.business_vocab:
            return {}

        # Mots business dans les textes
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())

        ref_business = ref_words & self.business_vocab
        hyp_business = hyp_words & self.business_vocab

        # Analyse phonétique des mots business
        business_phonetic_preservation = len(ref_business & hyp_business) / len(ref_business) if ref_business else 1.0

        return {
            "business_phonetic_preservation": business_phonetic_preservation
            ,"business_words_phonetic": list(ref_business & hyp_business)[:3]
            ,"phonetic_complexity": (len(ref_phonemes) + len(hyp_phonemes)) / 2
            ,"business_phonetic_density": len(ref_business) / len(ref_words) if ref_words else 0.0
        }

    def compare_transcriptions(self, reference: str, transcriptions: dict) -> dict:
        """
        Compare plusieurs transcriptions avec PER.

        Args:
            reference: Texte de référence
            transcriptions: Dict {modèle: transcription}

        Retourne:
            dict: Résultats PER par modèle
        """
        results = {}

        for model_name, transcription in transcriptions.items():
            result = self.calculate(reference, transcription)
            results[model_name] = {
                "per_score": result.score
                ,"levenshtein_distance": result.details.get("levenshtein_distance", 0)
                ,"phonemes_ref": result.details.get("phonemes_ref", 0)
                ,"phonemes_hyp": result.details.get("phonemes_hyp", 0)
                ,"processing_time": result.processing_time
            }

        return results
