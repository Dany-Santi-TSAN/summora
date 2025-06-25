"""
Métriques d'évaluation spécialisées pour Summora Meeting Analyzer
Focus business context : WER sur termes métier, ROUGE pour résumés actionnable
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Import vocabulaire business centralisé
from .business_vocabulary import get_all_business_keywords, BUSINESS_KEYWORDS

logger = logging.getLogger(__name__)

@dataclass
class BusinessWERResult:
    """Résultat WER spécialisé business."""
    global_wer: float
    business_wer: float
    business_preservation_rate: float
    category_wer: Dict[str, float]
    missing_keywords: List[str]
    correctly_transcribed: List[str]
    total_business_words: int

class BusinessMetricsCalculator:
    """Calculateur de métriques orientées business pour meetings."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.business_vocab = get_all_business_keywords()  # ✅ Import centralisé

        self.logger.info(f"📋 Vocabulaire business chargé: {len(self.business_vocab)} mots")

    def calculate_business_wer(self, reference: str, hypothesis: str) -> BusinessWERResult:
        """
        Calcule le WER spécialisé pour mots business.

        Args:
            reference: Transcription de référence (ground truth)
            hypothesis: Transcription générée par Whisper

        Retourne:
            BusinessWERResult: Métriques détaillées business
        """
        # Nettoyage textes
        ref_clean = self._clean_text(reference)
        hyp_clean = self._clean_text(hypothesis)

        # WER global classique
        global_wer = self._calculate_classic_wer(ref_clean, hyp_clean)

        # Extraction mots business
        ref_business = self._extract_business_words(ref_clean)
        hyp_business = self._extract_business_words(hyp_clean)

        # WER business spécialisé
        business_wer = self._calculate_classic_wer(
            ' '.join(ref_business),
            ' '.join(hyp_business)
        )

        # Analyse par catégorie
        category_wer = self._calculate_category_wer(ref_clean, hyp_clean)

        # Préservation keywords
        preservation_rate = len(hyp_business) / max(len(ref_business), 1)

        # Mots manqués vs correctement transcrits
        missing = [word for word in ref_business if word not in hyp_business]
        correct = [word for word in ref_business if word in hyp_business]

        return BusinessWERResult(
            global_wer=global_wer,
            business_wer=business_wer,
            business_preservation_rate=preservation_rate,
            category_wer=category_wer,
            missing_keywords=missing,
            correctly_transcribed=correct,
            total_business_words=len(ref_business)
        )

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte pour comparaison."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\']', ' ', text)  # Garde apostrophes
        text = re.sub(r'\s+', ' ', text)
        return text

    def _extract_business_words(self, text: str) -> List[str]:
        """Extrait seulement les mots business du texte."""
        words = text.split()
        business_words = []

        for word in words:
            if word in self.business_vocab:
                business_words.append(word)

        return business_words

    def _calculate_classic_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calcule le WER classique (distance de Levenshtein).

        Args:
            reference: Texte de référence
            hypothesis: Texte généré

        Retourne:
            float: WER (0.0 = parfait, 1.0 = totalement faux)
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        # Matrice distance Levenshtein
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

        # Initialisation
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        # Calcul distance
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

        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
        return min(wer, 1.0)  # Cap à 100%

    def _calculate_category_wer(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calcule WER par catégorie business."""
        category_wer = {}

        for category, keywords in BUSINESS_KEYWORDS.items():
            # Extraction mots de cette catégorie
            ref_cat = self._extract_category_words(reference, keywords)
            hyp_cat = self._extract_category_words(hypothesis, keywords)

            if ref_cat:  # Si on a des mots de cette catégorie en référence
                wer = self._calculate_classic_wer(
                    ' '.join(ref_cat),
                    ' '.join(hyp_cat)
                )
                category_wer[category] = wer
            else:
                category_wer[category] = 0.0  # Pas de mots dans cette catégorie

        return category_wer

    def _extract_category_words(self, text: str, keywords: List[str]) -> List[str]:
        """Extrait les mots d'une catégorie spécifique."""
        words = text.split()
        keywords_lower = [kw.lower() for kw in keywords]
        return [word for word in words if word in keywords_lower]

# === Fonction utilitaire benchmark ===

def benchmark_whisper_models_business(audio_path: str,
                                    reference_text: str,
                                    models: List[str] = None) -> Dict:
    """
    Benchmark modèles Whisper avec focus business.

    Args:
        audio_path: Chemin vers fichier audio
        reference_text: Transcription de référence manuelle
        models: Liste modèles à tester

    Retourne:
        Dict: Comparaison détaillée des modèles
    """
    if models is None:
        models = ['tiny', 'base', 'small']  # Large nécessite GPU

    calculator = BusinessMetricsCalculator()
    results = {}

    logger.info(f"🔬 Benchmark business WER sur {len(models)} modèles")

    for model in models:
        try:
            # Transcription avec modèle (placeholder - sera intégré avec transcriber)
            # hypothesis = transcribe_with_model(audio_path, model)
            hypothesis = f"placeholder_transcription_{model}"  # TODO: intégrer vraie transcription

            # Calcul métriques business
            business_result = calculator.calculate_business_wer(reference_text, hypothesis)

            results[model] = {
                'global_wer': business_result.global_wer,
                'business_wer': business_result.business_wer,
                'business_preservation': business_result.business_preservation_rate,
                'category_performance': business_result.category_wer,
                'business_words_total': business_result.total_business_words,
                'missing_critical': business_result.missing_keywords[:5],  # Top 5
            }

            logger.info(f"✅ {model}: Business WER = {business_result.business_wer:.1%}")

        except Exception as e:
            logger.error(f"❌ Erreur benchmark {model}: {e}")
            results[model] = {'error': str(e)}

    return results

def quick_test_business_wer():
    """Test rapide du calculateur business WER."""
    calculator = BusinessMetricsCalculator()

    # Exemples test
    reference = "Nous validons le budget de cinquante mille euros pour ce projet avec une deadline en décembre"
    hypothesis = "Nous validons le budget de 50000 euros pour ce projet avec une deadline en décembre"

    result = calculator.calculate_business_wer(reference, hypothesis)

    print("🧪 TEST BUSINESS WER")
    print(f"Global WER: {result.global_wer:.1%}")
    print(f"Business WER: {result.business_wer:.1%}")
    print(f"Préservation: {result.business_preservation_rate:.1%}")
    print(f"Mots business: {result.total_business_words}")

    return result

if __name__ == "__main__":
    # Test développement
    quick_test_business_wer()
