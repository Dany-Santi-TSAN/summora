"""
Calculateur expérimental SemDist : mesure de distance sémantique entre phrases à l’aide
d'embedding contextuels.

Mesure la similarité cosinus de sens entre deux phrases via embeddings.
Métrique expérimentale pour évaluation transcription meeting.(inspiré par l'étude HATS)

Exploratoire mais prometteur pour estimer la préservation du sens global d’une transcription,
notamment dans des cas d’usage orientés compréhension utilisateur.

AWS compatible
"""

import time
import logging
from typing import Optional
import numpy as np
from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)

class SemDistCalculator(BaseMetric):
    """
    Distance sémantique - Compare le sens global de deux phrases.

    Version découverte basée sur embeddings de phrases pour évaluer
    si une transcription préserve le sens original même avec des erreurs.
    """

    def __init__(self, business_vocab=None, model_name="sentence-transformers"):
        """
        Initialise le calculateur de distance sémantique.

        Args:
            business_vocab: Vocabulaire métier optionnel
            model_name: Modèle d'embeddings ("sentence-transformers", "camembert", etc.)
        """
        super().__init__(business_vocab)
        self.model_name = model_name
        self._embedding_model = None

    def get_metric_name(self) -> str:
        return "semdist"

    def _get_embedding_model(self):
        """
        Charge le modèle d'embeddings

        Retourne:
            Modèle d'embeddings configuré
        """
        if self._embedding_model is None:
            try:
                # Version découverte : sentence-transformers basique
                from sentence_transformers import SentenceTransformer

                # Modèle multilingue léger pour débuter
                self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.debug(f"✅ Modèle SemDist chargé: {self.model_name}")

            except ImportError:
                raise ImportError("pip install sentence-transformers required pour SemDist")
            except Exception as e:
                logger.error(f"❌ Erreur chargement modèle SemDist: {e}")
                raise RuntimeError(f"Erreur chargement SemDist: {e}")

        return self._embedding_model

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité sémantique entre deux phrases.

        Args:
            text1: Première phrase (référence)
            text2: Deuxième phrase (hypothèse)

        Retourne:
            float: Similarité cosinus (0.0 à 1.0, 1.0 = identique)
        """
        try:
            model = self._get_embedding_model()

            # Génération des embeddings
            embeddings = model.encode([text1, text2])

            # Calcul similarité cosinus
            embedding1, embedding2 = embeddings[0], embeddings[1]

            # Normalisation et produit scalaire
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            cosine_similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

            # Conversion en similarité (0-1)
            return float(max(0.0, cosine_similarity))

        except Exception as e:
            logger.warning(f"Erreur calcul similarité sémantique: {e}")
            return 0.0

    def calculate(self, reference: str, hypothesis: str, **kwargs) -> MetricResult:
        """
        Calcule la distance sémantique HATS entre référence et hypothèse.

        Principe: Plus les phrases ont un sens similaire, plus le score est bon.
        Score = 1.0 - similarité_cosinus (0.0 = parfait, 1.0 = opposé)

        Args:
            reference: Texte de référence (ground truth)
            hypothesis: Texte transcrit à évaluer

        Retourne:
            MetricResult: Score SemDist (0.0 = sens identique, 1.0 = sens opposé)
        """
        start_time = time.time()

        try:
            # Calcul similarité sémantique
            similarity = self._compute_semantic_similarity(reference, hypothesis)

            # Conversion en distance (HATS style)
            # Distance = 1 - similarité (0.0 = parfait, 1.0 = très différent)
            semdist_score = 1.0 - similarity

            # Stats business optionnelles
            business_stats = self._get_business_semantic_stats(
                reference, hypothesis, similarity
            ) if self.business_vocab else {}

            # Construction résultat AWS-safe
            result = MetricResult()
            result.metric_name = "semdist"
            result.score = semdist_score
            result.processing_time = time.time() - start_time
            result.business_focused = bool(self.business_vocab)
            result.details = {
                "implementation": f"sentence_transformers_{self.model_name}"
                ,"semantic_similarity": similarity
                ,"words_ref": len(reference.split())
                ,"words_hyp": len(hypothesis.split())
                ,"hats_inspiration": "Semantic Distance metric from HATS study"
                ,**business_stats
            }

            return result

        except Exception as e:
            # Résultat d'erreur AWS-safe
            result = MetricResult()
            result.metric_name = "semdist"
            result.score = 1.0  # Distance maximale en cas d'erreur
            result.processing_time = time.time() - start_time
            result.business_focused = False
            result.details = {"error": str(e)}
            return result

    def _get_business_semantic_stats(self, reference: str, hypothesis: str, similarity: float) -> dict:
        """
        Analyse l'impact business de la distance sémantique.

        Args:
            reference: Texte de référence
            hypothesis: Texte d'hypothèse
            similarity: Similarité calculée

        Retourne:
            dict: Statistiques business spécifiques
        """
        if not self.business_vocab:
            return {}

        # Mots business dans chaque phrase
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())

        ref_business = ref_words & self.business_vocab
        hyp_business = hyp_words & self.business_vocab

        # Impact business de la similarité sémantique
        business_overlap = len(ref_business & hyp_business)
        business_total = len(ref_business)

        return {
            "business_semantic_preservation": business_overlap / business_total if business_total > 0 else 1.0
            ,"business_words_preserved": list(ref_business & hyp_business)[:3]
            ,"semantic_quality_grade": "A" if similarity > 0.8 else "B" if similarity > 0.6 else "C"
            ,"meaning_preserved": similarity > 0.7  # Seuil lecture HATS
        }

    def compare_transcriptions(self, reference: str, transcriptions: dict) -> dict:
        """
        Compare plusieurs transcriptions avec SemDist.

        Args:
            reference: Texte de référence
            transcriptions: Dict {modèle: transcription}

        Retourne:
            dict: Résultats SemDist par modèle
        """
        results = {}

        for model_name, transcription in transcriptions.items():
            result = self.calculate(reference, transcription)
            results[model_name] = {
                "semdist_score": result.score
                ,"semantic_similarity": result.details.get("semantic_similarity", 0.0)
                ,"meaning_preserved": result.details.get("meaning_preserved", False)
                ,"processing_time": result.processing_time
            }

        return results
