"""
🎯 Spot Checker – Module de validation QA (Quality Assurance)

Ce module permet d'effectuer des contrôles qualité manuels
dans une pipeline LLM à deux niveaux stratégiques :

1️. Vérification aléatoire (Random QA) – Post-correction (LLM Corrector)
   - Échantillonnage aléatoire de segments corrigés
   - Validation humaine de la lisibilité, fidélité et absence d'hallucination

2️. Vérification ciblée (Strategic QA) – Extraction de contenus (LLM Extractor)
   - Spot-checks stratégiques sur des cas business ou sensibles
   - Évaluation manuelle de la pertinence des insights extraits

Utilisé dans une boucle Human-in-the-loop pour affiner les prompts, améliorer la qualité,
et constituer un éventuel jeu de données d'entraînement/fine-tuning.
"""

import json
import random
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SpotCheckSample:
    """Structure d'un échantillon type pour validation humaine"""
    sample_id: str
    content: str
    context: Dict
    sample_method: str # "random" ou "strategic"
    char_count: int
    word_count: int
    created_date: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class HumanFeedback:
    """Structure du feedback humain sur un échantillon"""
    sample_id: str
    quality_score: int # note entre 1 et 10 comme à l'école
    corrections: List[str]
    comments: str
    reviewer: str
    date_of_rewiew: str
    improvement_suggestions: List[str] = None

    def __post_init__(self):
        if self.improvement_suggestions is None:
            self.improvement_suggestions = []

class SpotCheckSaver:
    """Gestionnaire de sauvegarde pour les échantillons et feedback QA."""

    @staticmethod
    def save_samples(samples: List[SpotCheckSample],
                    output_dir: str = "output/qa/samples") -> str:
        """
        Sauvegarde les échantillons pour annotation humaine.

        Args:
            samples (List[SpotCheckSample]): Échantillons à sauvegarder
            output_dir (str): Répertoire de sortie

        Retourne:
            str: Chemin du fichier sauvegardé
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spot_check_samples_{timestamp}.json"
        file_path = output_path / filename

        # Conversion en dict pour sérialisation JSON
        samples_data = {
            "metadata": {
                "total_samples": len(samples)
                ,"sample_methods": list(set(s.sample_method for s in samples))
                ,"created_date": datetime.now().isoformat()
                ,"ready_for_annotation": True
            },
            "samples": [
                {
                    "sample_id": s.sample_id
                    ,"content": s.content
                    ,"context": s.context
                    ,"sample_method": s.sample_method
                    ,"char_count": s.char_count
                    ,"word_count": s.word_count
                    ,"created_date": s.created_date
                    ,"metadata": s.metadata
                }
                for s in samples
            ]
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(samples_data, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 {len(samples)} échantillons sauvegardés: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde échantillons: {e}")
            return ""

    @staticmethod
    def save_feedback(feedback: HumanFeedback,
                     output_dir: str = "output/qa/feedback") -> str:
        """
        Sauvegarde le feedback humain.

        Args:
            feedback (HumanFeedback): Feedback à sauvegarder
            output_dir (str): Répertoire de sortie

        Retourne:
            str: Chemin du fichier sauvegardé
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"feedback_{feedback.sample_id}.json"
        file_path = output_path / filename

        feedback_data = {
            "sample_id": feedback.sample_id
            ,"quality_score": feedback.quality_score
            ,"corrections": feedback.corrections
            ,"comments": feedback.comments
            ,"reviewer": feedback.reviewer
            ,"date_of_review": feedback.date_of_rewiew
            ,"improvement_suggestions": feedback.improvement_suggestions
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)

            logger.info(f"📋 Feedback sauvegardé: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde feedback: {e}")
            return ""


class SpotChecker:
    """
    SpotChecker : Contrôle qualité HITL pour valider manuellement les sorties
    LLM (Corrector et Extractor) via des échantillons sélectionnés différemment :

    1. Mode aléatoire : post-correction pour le LLM Corrector
    2. Mode ciblé : autour des insights business pour le LLM Extractor
    """

    def __init__(self, sample_size: int = 3, max_chars: Optional[int] = 10000):
        """
        Initialise le SpotChecker.

        Args:
            sample_size (int): Nombre d'échantillons à extraire par défaut
            max_chars (int, optional): Longueur max d'un échantillon (en caractères)
        """
        self.sample_size = sample_size
        self.max_chars = max_chars
        self.samples: List[SpotCheckSample] = []

        logger.info(f"🎯 SpotChecker initialisé (sample_size={sample_size}, max_chars={max_chars})")


    def random_sample(self, text: str, sample_size: Optional[int] = None) -> List[SpotCheckSample]:
        """
        Sélectionne aléatoirement un sous-ensemble du dataset pour annotation humaine.
        Mode aléatoire : Random QA pour main_corrector.py

        Args:
            text (str): Texte complet à échantillonner
            sample_size (int, optional): Nombre d'échantillons (défaut: self.sample_size)

        Retourne:
            List[SpotCheckSample]: Liste d'échantillons sélectionnés
        """
        if not text:
            logger.warning("⚠️ Texte vide. Aucun échantillon à extraire.")
            return []

        sample_size = sample_size or self.sample_size
        max_chars = self.max_chars or 10000

        # Découpage du texte en segment
        segments = self._segment_text(text, max_chars)

        if len(segments) <= sample_size:
            selected_segments = segments
            logger.info(f"📋 Tous les segments sélectionnés ({len(segments)} <= {sample_size})")
        else:
            selected_segments = random.sample(segments, sample_size)
            logger.info(f"🎲 {sample_size} segments sélectionnés aléatoirement sur {len(segments)}")

        # Création des échantillons structurés
        samples = []
        for i, segment in enumerate(selected_segments):
            sample = SpotCheckSample(
                sample_id=f"random_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                ,content=segment
                ,context={"total_segments": len(segments), "segment_index": segments.index(segment)}
                ,sample_method="random"
                ,char_count=len(segment)
                ,word_count=len(segment.split())
                ,created_date=datetime.now().isoformat()
            )
            samples.append(sample)

        self.samples.extend(samples)
        logger.info(f"✅ {len(samples)} échantillons random créés")
        return samples

    def strategic_sample(self, text: str, keywords: List[str], context_window: int = 500) -> List[SpotCheckSample]:
        """
        Sélectionne stratégiquement des segments autour de mots-clés business.
        Mode ciblé : Strategic QA pour main_extractor.py

        Args:
            text (str): Texte complet
            keywords (List[str]): Mots-clés business à cibler
            context_window (int): Fenêtre de contexte autour des mots-clés

        Retourne:
            List[SpotCheckSample]: Échantillons autour des mots-clés
        """
        if not text or not keywords:
            logger.warning("⚠️ Texte ou mots-clés vides.")
            return []

        samples = []
        text_lower = text.lower()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            start = 0

            while True:
                # Trouve la prochaine occurence du mot-clé
                pos = text_lower.find(keyword_lower, start)
                if pos == -1: # si aucun mot clé n'est trouvé, sortir de la boucle
                    break

                # Extrait le contexte autour du mot-clé
                context_start = max(0, pos - context_window // 2)
                context_end = min(len(text), pos + len(keyword) + context_window // 2)
                context_segment = text[context_start:context_end]

                # Evite le doublons
                if not any(sample.content == context_segment for sample in samples):
                    sample = SpotCheckSample(
                        sample_id=f"strategic_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(samples)}"
                        ,content=context_segment
                        ,context={
                            "target_keyword": keyword
                            ,"keyword_position": pos
                            ,"context_window": context_window
                        },
                        sample_method="strategic"
                        ,char_count=len(context_segment)
                        ,word_count=len(context_segment.split())
                        ,created_date=datetime.now().isoformat()
                        ,metadata={"keyword_highlighted": True}
                    )
                    samples.append(sample)

                start = pos + 1

        self.samples.extend(samples)
        logger.info(f"🎯 {len(samples)} échantillons strategic créés autour de {len(keywords)} mots-clés")
        return samples

    def _segment_text(self, text: str, max_chars: int) -> List[str]:
        """
        Découpe intelligente d'un texte en segments pour échantillonnage
        avec limite 80 caractères par ligne pour la lisibilité.

        Args:
            text (str): Texte à découper
            max_chars (int): Taille maximum d'un segment

        Retourne:
            List[str]: Liste des segments
        """
        if len(text) <= max_chars:
            return [text]

        segments = []

        # Découpage par paragraphes d'abord
        paragraphs = text.split('\n\n')
        current_segment = ""

        for paragraph in paragraphs:
            if len(current_segment + paragraph) <= max_chars:
                current_segment += ("\n\n" if current_segment else "") + paragraph
            else:
                if current_segment:
                    segments.append(current_segment.strip())

                # Si un paragraphe est trop long, découpe par phrases
                if len(paragraph) > max_chars:
                    sentences = paragraph.split('. ')
                    sentence_segment = ""

                    for sentence in sentences:
                        if len(sentence_segment + sentence) <= max_chars:
                            sentence_segment += (". " if sentence_segment else "") + sentence
                        else:
                            if sentence_segment:
                                segments.append(sentence_segment.strip())
                            sentence_segment = sentence

                    current_segment = sentence_segment
                else:
                    current_segment = paragraph

        if current_segment:
            segments.append(current_segment.strip())

        formatted_segments = []
        for segment in segments:
            formatted_segment = self._format_for_readability(segment)
            formatted_segments.append(formatted_segment)

        return formatted_segments

    def _format_for_readability(self, text: str) -> str:
        """
        Formate le texte avec limite 80 caractères par ligne.
        """
        logger.info(f"🔍 DEBUG formatage - Input: {len(text)} chars")
        paragraphs = text.split('\n\n')
        logger.info(f"🔍 DEBUG formatage - Paragraphes détectés: {len(paragraphs)}")

        lines = []
        for paragraph in text.split('\n\n'):
            words = paragraph.split()
            current_line = ""

            for word in words:
                if len(current_line + word) > 80:
                    lines.append(current_line.strip())
                    current_line = word + " "
                else:
                    current_line += word + " "

            if current_line:
                lines.append(current_line.strip())
            lines.append("")

        result = '\n'.join(lines).rstrip()
        newline_count = result.count('\n')
        logger.info(f"🔍 DEBUG formatage - Output: {len(result)} chars, {newline_count} lignes")
        return result

    def save_samples_for_annotation(self, samples: List[SpotCheckSample],
                                   output_dir: str = "output/qa/samples") -> str:
        """Délègue la sauvegarde à SpotCheckPersistence."""
        return SpotCheckSaver.save_samples(samples, output_dir)

    def save_feedback(self, feedback: HumanFeedback,
                     output_dir: str = "output/qa/feedback") -> str:
        """Délègue la sauvegarde à SpotCheckPersistence."""
        return SpotCheckSaver.save_feedback(feedback, output_dir)

    def get_samples_summary(self) -> Dict:
        """
        Retourne un résumé des échantillons créés.

        Retourne:
            Dict: Statistiques des échantillons
        """
        if not self.samples:
            return {"total_samples": 0}

        random_count = sum(1 for s in self.samples if s.sample_method == "random")
        strategic_count = sum(1 for s in self.samples if s.sample_method == "strategic")

        total_chars = sum(s.char_count for s in self.samples)
        total_words = sum(s.word_count for s in self.samples)

        return {
            "total_samples": len(self.samples)
            ,"random_samples": random_count
            ,"strategic_samples": strategic_count
            ,"total_characters": total_chars
            ,"total_words": total_words
            ,"avg_chars_per_sample": total_chars / len(self.samples)
            ,"avg_words_per_sample": total_words / len(self.samples)
        }

# Factory functions
def create_spot_checker(sample_size: int = 3, max_chars: int = 10000) -> SpotChecker:
    """
    Factory pour créer un SpotChecker.

    Args:
        sample_size (int): Nombre d'échantillons par défaut
        max_chars (int): Taille max d'un échantillon

    Retourne:
        SpotChecker: Instance configurée
    """
    return SpotChecker(sample_size, max_chars)

def quick_random_check(text: str, sample_size: int = 3) -> List[SpotCheckSample]:
    """
    Fonction utilitaire pour un spot-check random rapide.

    Args:
        text (str): Texte à échantillonner
        sample_size (int): Nombre d'échantillons

    Retourne:
        List[SpotCheckSample]: Échantillons pour annotation
    """
    checker = create_spot_checker(sample_size)
    return checker.random_sample(text, sample_size)
