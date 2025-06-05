"""
Analyseur audio spécialisé pour les meetings
Analyse des propriétés audio avec focus sur la qualité meeting
"""
import os
from pathlib import Path
from typing import Optional, Union, Dict
import logging
from dataclasses import dataclass

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display

from .utils import validate_audio_path, format_duration

logger = logging.getLogger(__name__)

# === Configuration de l'analyseur audio spécifique au réunion ===

@dataclass
class MeetingAudioConfig:
    """
    Configuration pour l’analyse des réunions audio.
    Adapté aux besoins spécifiques des échanges humains en contexte professionnel.

    """
    frame_length: int = 2048
    hop_length: int = 512
    n_mfcc: int = 13 # Standard théorique
    silence_percentile: float = 15.0 # Seuil adaptatif
    generate_plots: bool = True
    plot_figsize: tuple = (15, 10)

    # Seuils spécifiques aux meetings
    min_meeting_duration: float = 60.0   # 1 minute minimum
    optimal_speech_ratio: float = 0.65   # 65% de parole idéal
    max_silence_gap: float = 10.0        # Silence > 10s = problème

class MeetingAudioAnalyzer:
    """
    Analyseur dédié aux fichiers audio de réunions professionnelles.
    Fournit des métriques utiles et des recommandations pratiques.
    """

    def __init__(self, config: Optional[MeetingAudioConfig] = None):
        """
        Initialise l'analyseur audio pour meetings.

        Args:
            config: Configuration d'analyse (utilise les défauts si None)
        """
        self.config = config or MeetingAudioConfig()

    # === Calcul des scores de qualité de réunion ===

    def _calculate_meeting_quality_score(self, sr: int, duration: float,
                                       speech_ratio: float, dynamic_range_db: float) -> tuple:
        """
        Calcule un score de qualité spécifique aux meetings.

        Args:
            sr: Sample rate
            duration: Durée en secondes
            speech_ratio: Ratio de parole (0-1)
            dynamic_range_db: Dynamic range en dB

        Retourne:
            tuple: (score, grade, statuses, recommendations)
        """
        score = 0
        statuses = {}
        recommendations = []

        # Évaluation sample rate pour meetings (8kHz-48kHz)
        if 16000 <= sr <= 48000:
            score += 25
            statuses['sample_rate'] = "✅ Qualité optimale pour meeting"
        elif 8000 <= sr < 16000:
            score += 15
            statuses['sample_rate'] = "⚠️ Qualité acceptable mais améliorable"
            recommendations.append("Enregistrer en 16kHz minimum pour meilleure qualité")
        else:
            statuses['sample_rate'] = "❌ Qualité insuffisante pour meeting"
            recommendations.append("Utiliser un sample rate entre 16-48kHz")

        # Évaluation durée meeting
        if duration >= self.config.min_meeting_duration:
            score += 25
            if duration <= 3600:  # <= 1h
                statuses['duration'] = f"✅ Durée appropriée ({format_duration(duration)})"
            else:
                statuses['duration'] = f"⚠️ Meeting long ({format_duration(duration)}) - vérifier efficacité"
                recommendations.append("Meetings > 1h peuvent être moins efficaces")
        else:
            statuses['duration'] = f"⚠️ Meeting très court ({format_duration(duration)})"
            recommendations.append("Meetings < 1min peu adaptés à l'analyse")

        # Évaluation ratio de parole pour meetings
        if 0.55 <= speech_ratio <= 0.75:
            score += 25
            statuses['speech_ratio'] = f"✅ Ratio parole optimal ({speech_ratio*100:.1f}%)"
        elif 0.4 <= speech_ratio < 0.55:
            score += 15
            statuses['speech_ratio'] = f"⚠️ Peu de parole ({speech_ratio*100:.1f}%)"
            recommendations.append("Meeting silencieux - vérifier engagement participants")
        elif speech_ratio > 0.8:
            score += 10
            statuses['speech_ratio'] = f"⚠️ Beaucoup de parole ({speech_ratio*100:.1f}%)"
            recommendations.append("Meeting dense - prévoir des pauses")
        else:
            statuses['speech_ratio'] = f"❌ Trop de silence ({speech_ratio*100:.1f}%)"
            recommendations.append("Meeting peu productif - revoir animation")

        # Évaluation dynamic range pour meetings
        if 15 <= dynamic_range_db <= 50:
            score += 25
            statuses['dynamic_range'] = "✅ Qualité audio claire"
        elif dynamic_range_db < 10:
            score += 5
            statuses['dynamic_range'] = "❌ Audio compressé ou faible"
            recommendations.append("Améliorer setup micro pour meetings futurs")
        else:
            score += 15
            statuses['dynamic_range'] = "⚠️ Audio avec variations importantes"
            recommendations.append("Vérifier distance micro et acoustique salle")

        # Calcul du grade meeting-specific
        if score >= 85:
            grade = "A+"
        elif score >= 75:
            grade = "A"
        elif score >= 65:
            grade = "B"
        elif score >= 50:
            grade = "C"
        else:
            grade = "D"

        return score, grade, statuses, recommendations

    # === Génération des visualisations audio ===

    """
    Les visualisations aident à comprendre la dynamique de la réunion :
    - les moments où les participants parlent le plus
    - les périodes de silence
    - l'énergie globale de la discussion.

    Cela peut aider à identifier les réunions productives et celles qui nécessitent des améliorations.

    La fonction peut également aider à diagnostiquer des problèmes techniques dans les enregistrements audio.
    """


    def _generate_meeting_visualizations(self, y: np.ndarray, sr: int,
                                       duration: float, audio_path: str) -> None:
            """
            Génère des visualisations optimisées pour l'analyse de meetings :

            1. Timeline audio avec détection des moments parlés (zones orangées)
            2. Spectrogramme centré sur la bande voix humaine (200–4000Hz)
            3. Histogramme silence vs parole – indicateur de dynamique
            4. Suivi RMS (Root Mean Square) pour détecter les pics d’activité

            Args:
                y: Signal audio
                sr: Sample rate
                duration: Durée
                audio_path: Chemin du fichier pour le titre
            """
            if not self.config.generate_plots:
                return

            try:
                fig, axes = plt.subplots(2, 2, figsize=self.config.plot_figsize)
                fig.suptitle(f'📊 Analyse Meeting Audio - {os.path.basename(audio_path)}'
                            ,fontsize=16, fontweight='bold')

                # 1. Timeline audio avec zones de parole
                time = np.linspace(0, duration, len(y))
                axes[0, 0].plot(time, y, linewidth=0.6, color='steelblue', alpha=0.7)

                # Highlight des zones de forte activité (parole probable)
                rms = librosa.feature.rms(y=y, frame_length=self.config.frame_length
                                        ,hop_length=self.config.hop_length)[0]
                time_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr
                                                ,hop_length=self.config.hop_length)
                speech_threshold = np.percentile(rms, 70)
                speech_zones = rms > speech_threshold

                for i in range(len(time_rms)-1):
                    if speech_zones[i]:
                        axes[0, 0].axvspan(time_rms[i], time_rms[i+1], alpha=0.3, color='orange')

                axes[0, 0].set_title('🎤 Timeline Meeting avec zones de parole')
                axes[0, 0].set_xlabel('Temps (minutes)')
                axes[0, 0].set_ylabel('Amplitude')
                axes[0, 0].grid(True, alpha=0.3)

                # Conversion temps en minutes pour meeting
                time_minutes = time / 60
                axes[0, 0].set_xlim(0, duration/60)

                # 2. Spectrogramme avec focus voix humaine (200-4000Hz)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                img = librosa.display.specshow(D, y_axis='hz', x_axis='time'
                                            ,sr=sr, ax=axes[0, 1], cmap='viridis'
                                            ,fmax=4000)  # Focus fréquences voix
                axes[0, 1].set_title('🎵 Spectrogramme (focus voix humaine)')
                plt.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')

                # 3. Analyse des silences et activité
                silence_threshold = np.percentile(np.abs(y), self.config.silence_percentile)
                is_silence = np.abs(y) < silence_threshold

                # Histogramme silence vs parole
                categories = ['Silence', 'Parole']
                values = [np.mean(is_silence), 1 - np.mean(is_silence)]
                colors = ['lightcoral', 'lightgreen']

                bars = axes[1, 0].bar(categories, values, color=colors, alpha=0.7)
                axes[1, 0].set_title('📊 Répartition Silence / Parole')
                axes[1, 0].set_ylabel('Proportion')
                axes[1, 0].set_ylim(0, 1)

                # Ajout des pourcentages sur les barres
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01
                                ,f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')

                # 4. Énergie RMS avec seuils meeting
                axes[1, 1].plot(time_rms, rms, color='orange', linewidth=2, label='Énergie RMS')
                axes[1, 1].axhline(y=speech_threshold, color='red', linestyle='--'
                                ,alpha=0.7, label='Seuil parole')
                axes[1, 1].fill_between(time_rms, rms, alpha=0.3, color='orange')
                axes[1, 1].set_title('⚡ Énergie Meeting (détection activité)')
                axes[1, 1].set_xlabel('Temps (minutes)')
                axes[1, 1].set_ylabel('Énergie RMS')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            except Exception as e:
                logger.warning(f"Erreur génération visualisations meeting: {e}")

    # === Fonction principale d'analyse audio de réunion ===

    def analyze_meeting_audio_property(self, audio_path: Union[str, Path]) -> Dict:
        """
        Analyse complète d'un audio de réunion

        Pipeline de traitement :
        1. 📂 Validation et chargement du fichier (avec son sample rate natif)
        2. 📊 Extraction des features clés :
            - Durée, Speech/Silence Ratio
            - Plage dynamique (dB)
            - MFCCs (vecteurs vocaux)
            - ZCR (clarté vocale)
        3. 💡 Scoring & recommandations via `_calculate_meeting_quality_score`
        4. 📉 Visualisation (optionnelle)

        Args:
            audio_path: Chemin vers le fichier audio

        Retourne:
            Dict: Propriétés audio et métriques spécifiques aux meetings
        """
        audio_path = Path(audio_path)

        try:
            logger.info(f"🎤 Analyse audio meeting: {audio_path.name}")

            # Validation et chargement
            validated_path = validate_audio_path(audio_path)
            if not validated_path:
                return {"error": "file_validation_failed", "path": str(audio_path)}

            # Chargement avec sample rate original
            y, sr = librosa.load(str(validated_path), sr=None)
            duration = len(y) / sr

            logger.info(f"📈 Propriétés: {sr}Hz, {format_duration(duration)}")

            # Génération des visualisations meeting
            logger.info("🎨 Génération visualisations meeting...")
            self._generate_meeting_visualizations(y, sr, duration, str(audio_path))

            # Calculs des métriques meeting
            logger.info("🔢 Calcul métriques meeting...")

            # Détection de silence adaptatif
            silence_threshold = np.percentile(np.abs(y), self.config.silence_percentile)
            silence_ratio = np.mean(np.abs(y) < silence_threshold)
            speech_ratio = 1 - silence_ratio

            # Dynamic Range
            max_amp = np.max(np.abs(y))
            mean_amp = np.mean(np.abs(y))
            dynamic_range_db = 20 * np.log10(max_amp / (mean_amp + 1e-8))

            # Zero Crossing Rate (indicateur de clarté vocale)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            mean_zcr = np.mean(zcr)

            # MFCC pour analyse vocale meeting
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config.n_mfcc)
            mfcc_mean = np.mean(mfccs, axis=1)

            # Score de qualité meeting
            meeting_score, meeting_grade, meeting_statuses, recommendations = \
                self._calculate_meeting_quality_score(sr, duration, speech_ratio, dynamic_range_db)

            # Affichage des résultats
            logger.info(f"📊 Métriques meeting:")
            logger.info(f"   • Ratio parole: {speech_ratio*100:.1f}%")
            logger.info(f"   • Score qualité: {meeting_score}/100 (Grade {meeting_grade})")
            for status in meeting_statuses.values():
                logger.info(f"   • {status}")

            return {
                # Propriétés de base
                "duration": float(duration)
                ,"duration_formatted": format_duration(duration)
                ,"sample_rate": int(sr)
                ,"samples_count": int(len(y))

                # Métriques d'amplitude
                ,"max_amplitude": float(max_amp)
                ,"rms_energy": float(np.sqrt(np.mean(y**2)))
                ,"mean_amplitude": float(mean_amp)

                # Métriques meeting spécifiques
                ,"silence_ratio": float(silence_ratio)
                ,"speech_ratio": float(speech_ratio)
                ,"dynamic_range_db": float(dynamic_range_db)
                ,"zero_crossing_rate": float(mean_zcr)

                # Score qualité meeting
                ,"meeting_quality_score": int(meeting_score)
                ,"meeting_quality_grade": meeting_grade
                ,"meeting_status": meeting_statuses
                ,"recommendations": recommendations

                # Analyse vocale optimisée meeting
                ,"mfcc_coefficients": [float(x) for x in mfcc_mean[:5]]
                ,"vocal_clarity_score": float(1.0 / (1.0 + mean_zcr * 10))  # Score clarté

                # Métadonnées
                ,"analysis_method": "librosa + meeting_optimization"
                ,"optimized_for": "meeting_analysis"
                ,"config_used": {
                    "silence_percentile": self.config.silence_percentile,
                    "min_duration": self.config.min_meeting_duration
                    }
                }

        except Exception as e:
            logger.error(f"❌ Erreur analyse audio meeting: {e}")
            return {
                "error": str(e),
                "error_type": "meeting_audio_analysis_failure",
                "suggestion": "Vérifiez le format audio et la qualité d'enregistrement"
                }


# Factory function
def analyze_meeting_audio_file(audio_path: Union[str, Path],
                         generate_plots: bool = True,
                         **config_kwargs) -> Dict:
    """
    Helper function pour analyse rapide d'un audio de meeting.

    Args:
        audio_path: Chemin vers le fichier audio
        generate_plots: Générer les visualisations
        **config_kwargs: Paramètres de configuration

    Retourne:
        Dict: Résultats d'analyse audio meeting
    """
    config = MeetingAudioConfig(generate_plots=generate_plots, **config_kwargs)
    analyzer = MeetingAudioAnalyzer(config)
    return analyzer.analyze_meeting_audio(audio_path)
