#!/usr/bin/env python3
"""
Main Recommendation V3 - Pipeline cascade RAG-Enhanced
Input: Transcription TXT + (Optional) Audio Analysis JSON → Output: Recommandations
Architecture simplifiée: Transcription → Extraction → Cascade → Affichage
"""
import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup path pour imports Summora
sys.path.append(str(Path(__file__).parent.parent))

# Import des modules Summora
from src.llm.llm_qwen_recommendation import recommend_and_evaluate_meeting
from src.qa.spot_checker import SpotChecker
from src.core.business_vocabulary import BUSINESS_KEYWORDS, get_keyword_category
from src.rag.rag_meeting_helper import enhance_qwen_with_rag

from dotenv import load_dotenv
load_dotenv()

# === DICTIONNAIRE DE RECOMMANDATIONS (Enhanced) ===
class MeetingRecommendationEngine:
    """Moteur de recommandations pour améliorer les meetings - Enhanced Dict."""

    def __init__(self):
        self.recommendations_db = {
            'structure': [
                "Définir un ordre du jour clair avant le meeting",
                "Commencer par rappeler les objectifs et l'agenda",
                "Structurer en phases : contexte → discussion → décisions → actions",
                "Terminer par un récapitulatif des décisions et next steps"
            ],
            'participation': [
                "Encourager chaque participant à s'exprimer",
                "Poser des questions ouvertes pour stimuler la discussion",
                "Utiliser la technique du 'tour de table' pour les décisions importantes",
                "Limiter les apartés et recentrer sur les objectifs"
            ],
            'efficacite': [
                "Limiter la durée à 45-60 minutes maximum",
                "Prévoir des pauses pour les meetings longs",
                "Désigner un animateur pour maintenir le rythme",
                "Utiliser un timer pour respecter les créneaux"
            ],
            'decisions': [
                "Formuler clairement chaque décision prise",
                "Assigner un responsable et une échéance pour chaque action",
                "Valider la compréhension avec tous les participants",
                "Documenter les décisions pour référence future"
            ],
            'technique': [
                "Tester la qualité audio avant de commencer",
                "Utiliser un micro de qualité pour l'enregistrement",
                "Réduire les bruits de fond (ventilation, notifications)",
                "Positionner le micro à distance optimale des participants"
            ]
        }

    def generate_recommendations(self, quality_analysis: Dict, extraction_result: Dict) -> List[Dict]:
        """Génère des recommandations personnalisées selon les faiblesses détectées."""
        grade = quality_analysis.get('quality_indicators', {}).get('grade', 'D')

        if grade == "A":
            return [{
                'categorie': 'Excellence',
                'titre': 'Meeting exemplaire - Continuez cette excellence !',
                'description': 'Votre meeting atteint un niveau d\'excellence remarquable.',
                'impact': 'inspirational',
                'facilite_implementation': 'immediate',
                'source': 'dict_enhanced'
            }]

        # Sélection recommandations selon grade
        if grade == "B":
            selected_reco = (
                self.recommendations_db['structure'][:1] +
                self.recommendations_db['decisions'][:1]
            )
        elif grade == "C":
            selected_reco = (
                self.recommendations_db['structure'][:2] +
                self.recommendations_db['efficacite'][:1] +
                self.recommendations_db['decisions'][:2]
            )
        else:  # Grade D
            selected_reco = (
                self.recommendations_db['structure'][:2] +
                self.recommendations_db['participation'][:2] +
                self.recommendations_db['efficacite'][:2] +
                self.recommendations_db['technique'][:1]
            )

        # Formatage en dict structuré
        recommendations = []
        for reco_text in selected_reco:
            category = self._get_category_from_text(reco_text)
            recommendations.append({
                'categorie': category,
                'titre': reco_text,
                'description': f"{reco_text}. Recommandation adaptée au grade {grade}.",
                'impact': 'high' if grade == 'D' else 'medium',
                'facilite_implementation': 'easy',
                'source': 'dict_enhanced'
            })

        return recommendations[:8 if grade == 'D' else (5 if grade == 'C' else 3)]

    def _get_category_from_text(self, text: str) -> str:
        """Détermine la catégorie d'une recommandation."""
        text_lower = text.lower()

        if any(word in text_lower for word in ['ordre', 'agenda', 'structurer', 'phases']):
            return 'Structure'
        elif any(word in text_lower for word in ['participant', 'question', 'tour de table']):
            return 'Participation'
        elif any(word in text_lower for word in ['durée', 'pause', 'timer', 'rythme']):
            return 'Efficacité'
        elif any(word in text_lower for word in ['audio', 'micro', 'bruit']):
            return 'Technique'
        else:
            return 'Général'

# === ANALYSEUR QUALITÉ ===
class RecommendationAnalyzer:
    """Analyseur de qualité meeting."""

    def __init__(self):
        logger.info("Analyseur qualité meeting initialisé")

    def analyze_extraction_quality(self, extraction_data: Dict) -> Dict:
        """Analyse la qualité d'un meeting depuis données d'extraction."""
        try:
            transcription_data = extraction_data.get('transcription', {})
            existing_quality = transcription_data.get('quality_analysis', {})
            extraction_results = extraction_data.get('extraction', {})

            transcription_text = transcription_data.get('text', '')
            grade = existing_quality.get('grade', 'C')
            global_score = existing_quality.get('global_score', 70)

            # Comptage topics/points depuis extraction
            topics_count = 0
            points_count = 0

            if extraction_results:
                # Compter topics
                topics = extraction_results.get('topics', {})
                if isinstance(topics, dict) and 'topics' in topics:
                    topics_count = len(topics['topics'])

                # Compter actions + décisions comme points
                actions = extraction_results.get('actions', {})
                if isinstance(actions, dict) and 'actions' in actions:
                    points_count += len(actions['actions'])

                decisions = extraction_results.get('decisions', {})
                if isinstance(decisions, dict) and 'decisions' in decisions:
                    points_count += len(decisions['decisions'])

            logger.info(f"Analyse meeting: Grade {grade} ({global_score}/100), {topics_count} topics, {points_count} points")

            # Construction analyse bienveillante
            analysis = {
                'strengths': [],
                'improvements': [],
                'quality_indicators': {
                    'grade': grade,
                    'global_score': global_score,
                    'topics_count': topics_count,
                    'key_points_count': points_count,
                    'text_length': len(transcription_text)
                },
                'overall_assessment': 'excellent' if grade == 'A' else ('good' if grade == 'B' else 'needs_improvement')
            }

            # Points forts
            if topics_count >= 3:
                analysis['strengths'].append(f"Richesse thématique ({topics_count} sujets abordés)")
            if points_count >= 2:
                analysis['strengths'].append(f"Contenu actionnable ({points_count} éléments détectés)")

            analysis['strengths'].append("Participation active des membres" if grade == 'D' else "Communication efficace")

            # Améliorations suggérées
            if grade == 'D':
                analysis['improvements'].extend([
                    {'category': 'Structure', 'suggestion': 'Définir un ordre du jour précis'},
                    {'category': 'Animation', 'suggestion': 'Désigner un facilitateur'},
                    {'category': 'Décision', 'suggestion': 'Réserver du temps pour les conclusions'}
                ])
            elif grade == 'C':
                analysis['improvements'].extend([
                    {'category': 'Efficacité', 'suggestion': 'Structurer en blocs thématiques'},
                    {'category': 'Participation', 'suggestion': 'Encourager la synthèse collective'}
                ])

            return analysis

        except Exception as e:
            logger.error(f"Erreur analyse extraction: {e}")
            return {
                'strengths': ['Tentative d\'analyse courageuse'],
                'improvements': [{'category': 'Technique', 'suggestion': 'Vérifier données d\'extraction'}],
                'quality_indicators': {'grade': 'C', 'global_score': 50, 'topics_count': 0, 'key_points_count': 0},
                'overall_assessment': 'needs_improvement'
            }

# === UTILITAIRE AUDIO ===
def load_audio_data(audio_file: str) -> Optional[str]:
    """Charge et formate les métriques audio pour le contexte des recommandations."""
    try:
        with open(audio_file, 'r') as f:
            data = json.load(f)

        analysis = data.get('analysis', {})

        # Extraction des métriques clés pour les recommandations
        metrics_summary = (
            f"Durée: {analysis.get('duration_formatted', 'N/A')}, "
            f"Ratio parole: {analysis.get('speech_ratio', 0)*100:.0f}%, "
            f"Qualité: {analysis.get('meeting_quality_score', 0)}/100 ({analysis.get('meeting_quality_grade', 'N/A')}), "
            f"Clarté vocale: {analysis.get('vocal_clarity_score', 0):.2f}"
        )

        # Recommandations audio existantes
        audio_reco = analysis.get('recommendations', [])
        if audio_reco:
            metrics_summary += f", Observations: {', '.join(audio_reco)}"

        return metrics_summary

    except Exception as e:
        logger.warning(f"Erreur lecture audio data: {e}")
        return None

# === CASCADE RECOMMANDATIONS ===
class RecommendationCascade:
    """Pipeline de recommandations: RAG+LLM → LLM → Dict Enhanced."""

    def __init__(self):
        self.dict_engine = MeetingRecommendationEngine()
        logger.info("Pipeline RAG-Enhanced: RAG+LLM → LLM → Dict Enhanced")

    def _build_enhanced_context(self, extraction_data: Dict, audio_analysis_file: Optional[str] = None) -> str:
        """Construit contexte enrichi pour RAG."""
        context_parts = []

        if extraction_data and 'extraction' in extraction_data:
            extraction = extraction_data['extraction']

            # Type de meeting
            meeting_type_data = extraction.get('meeting_type', {})
            if isinstance(meeting_type_data, dict):
                meeting_type = meeting_type_data.get('meeting_type', 'Général')
            elif isinstance(meeting_type_data, str):
                meeting_type = meeting_type_data
            else:
                meeting_type = 'Général'
            context_parts.append(f"TYPE: {meeting_type}")

            # Topics principaux
            topics = extraction.get('topics', {})
            if isinstance(topics, dict) and 'topics' in topics:
                topic_list = topics['topics'][:7]
                topic_keywords = [t.get('keyword', '') for t in topic_list if isinstance(t, dict) and t.get('keyword')]
                if topic_keywords:
                    context_parts.append(f"TOPICS: {', '.join(topic_keywords)}")

            # Actions et décisions
            actions = extraction.get('actions', {})
            if isinstance(actions, dict) and 'actions' in actions:
                actions_count = len(actions['actions'])
                if actions_count > 0:
                    context_parts.append(f"ACTIONS: {actions_count} détectées")

            decisions = extraction.get('decisions', {})
            if isinstance(decisions, dict) and 'decisions' in decisions:
                decisions_count = len(decisions['decisions'])
                if decisions_count > 0:
                    context_parts.append(f"DÉCISIONS: {decisions_count} identifiées")

        # Enrichissement avec analyse audio si disponible
        if audio_analysis_file:
            audio_data = load_audio_data(audio_analysis_file)
            if audio_data:
                context_parts.append(f"AUDIO: {audio_data}")

        return f"CONTEXTE: {' | '.join(context_parts)}" if context_parts else ""

    def _get_meeting_type(self, extraction_data: Dict) -> str:
        """Extrait le type de meeting."""
        if not extraction_data or 'extraction' not in extraction_data:
            return 'Général'

        meeting_type_data = extraction_data['extraction'].get('meeting_type', {})
        if isinstance(meeting_type_data, dict):
            return meeting_type_data.get('meeting_type', 'Général')
        elif isinstance(meeting_type_data, str):
            return meeting_type_data
        return 'Général'

    def get_specialized_prompt_context(self, meeting_type: str) -> str:
        """Retourne contexte spécialisé selon type meeting."""
        few_shots_by_type = {
            'brainstorming': "Brainstorming créatif → Animation: 'Divergence 30min + Convergence', Structure: 'Post-it + Vote dot'",
            'décisionnelle': "Réunion décision → Efficacité: 'Options A/B/C + Vote + Validation'",
            'rétrospective': "Rétrospective agile → Structure: 'Start/Stop/Continue + Actions SMART'",
            'client': "Meeting client → Communication: 'Slides impactants + Démo live + Q&R'",
            'copil': "Copil stratégique → Efficacité: 'Dashboard KPIs + Points bloquants + Arbitrages'",
            'standup': "Stand-up quotidien → Animation: 'Tour de table 2min/pers + Actions jour'"
        }

        return few_shots_by_type.get(meeting_type.lower(),
                                    "Meeting structuré → Organisation: 'Agenda timing + Objectifs clairs + Synthèse'")

    def _recommend_with_rag_qwen(self, transcription_text: str, quality_analysis: Dict,
                                 extraction_data: Dict = None, audio_analysis_file: Optional[str] = None) -> Dict:
        """Méthode 1: RAG + LLM Qwen Enhanced."""

        try:
            logger.info("RAG + LLM: enrichissement avec contexte leadership...")

            # Enrichissement par le RAG
            enhanced_data = enhance_qwen_with_rag(transcription_text, extraction_data, quality_analysis)

            # Construction contexte enrichi (inclut audio si disponible)
            enhanced_context = self._build_enhanced_context(extraction_data, audio_analysis_file)
            meeting_type = self._get_meeting_type(extraction_data)
            specialized_context = self.get_specialized_prompt_context(meeting_type)

            # Enrichissement avec analyse audio si disponible
            if audio_analysis_file:
                audio_data = load_audio_data(audio_analysis_file)
                if audio_data:
                    enhanced_data['audio_data'] = audio_data

            # Ajout du contexte meeting au RAG
            enhanced_data['meeting_context'] = enhanced_context
            enhanced_data['meeting_type'] = meeting_type
            enhanced_data['specialized_context'] = specialized_context

            # Appel du LLM avec RAG + context meeting
            result = recommend_and_evaluate_meeting(transcription_text, enhanced_data)

            if result.get('success'):
                result['method'] = 'rag_enhanced_qwen'
                result['rag_used'] = 'rag_context' in enhanced_data
                result['meeting_type_detected'] = meeting_type
                logger.info(f"RAG + LLM réussi (RAG: {result['rag_used']})")
                return result
            else:
                return {'method': 'rag_enhanced_qwen', 'success': False, 'error': 'qwen_failed'}

        except Exception as e:
            logger.warning(f"RAG + LLM échoué: {str(e)}")
            return {'method': 'rag_enhanced_qwen', 'success': False, 'error': str(e)}

    def _recommend_with_qwen_only(self, transcription_text: str, quality_analysis: Dict,
                                  extraction_data: Dict = None, audio_analysis_file: Optional[str] = None) -> Dict:
        """Méthode 2: LLM Qwen avec cascade interne."""
        try:
            logger.info("Qwen cascade: génération recommandations...")

            # Enrichissement analyse audio si disponible
            enhanced_data = extraction_data or {}
            if audio_analysis_file:
                audio_data = load_audio_data(audio_analysis_file)
                if audio_data:
                    enhanced_data['audio_data'] = audio_data
                    logger.info("Analyse audio disponible pour enrichissement de la data")

            result = recommend_and_evaluate_meeting(transcription_text, enhanced_data)

            if result.get('success'):
                result['method'] = 'qwen_cascade'
                result['rag_used'] = False
                logger.info("Qwen cascade réussi")
                return result
            else:
                return {'method': 'qwen_cascade', 'success': False, 'error': 'qwen_cascade_failed'}

        except Exception as e:
            logger.warning(f"Qwen cascade échoué: {str(e)}")
            return {'method': 'qwen_cascade', 'success': False, 'error': str(e)}

    def _recommend_with_dict_enhanced(self, transcription_text: str, quality_analysis: Dict,
                                      extraction_data: Dict = None, audio_analysis_file: Optional[str] = None) -> Dict:
        """Méthode 3: Dict Enhanced (fallback garanti)."""
        try:
            logger.info("Dict Enhanced: génération recommandations...")

            enhanced_data = extraction_data.get('extraction', {}) if extraction_data else {}
            # Enrichissement analyse audio si disponible
            if audio_analysis_file:
                audio_data = load_audio_data(audio_analysis_file)
                if audio_data:
                    enhanced_data['audio_data'] = audio_data
                    logger.info("Analyse audio disponible pour enrichissement de la data")

            recommendations = self.dict_engine.generate_recommendations(quality_analysis, enhanced_data)

            grade = quality_analysis.get('quality_indicators', {}).get('grade', 'C')
            strengths = quality_analysis.get('strengths', [])

            if grade == 'A':
                conseil_synthese = "Meeting exemplaire ! Ces micro-ajustements le rendront parfait."
            elif grade == 'B':
                conseil_synthese = f"Excellent travail avec {len(strengths)} points forts ! Ces optimisations maximiseront l'impact."
            elif grade == 'C':
                conseil_synthese = "Très bon potentiel détecté ! Ces ajustements simples transformeront vos meetings."
            else:  # Grade D
                conseil_synthese = f"Formidable engagement des participants ! Canaliser cette énergie avec ces {len(recommendations)} optimisations."

            result = {
                'method': 'dict_enhanced',
                'success': True,
                'recommendations': {
                    'recommandations_principales': recommendations,
                    'resume_conseil': conseil_synthese,
                    'nb_recommandations': len(recommendations)
                },
                'ready_for_implementation': True,
                'conseil_quality': 'high' if grade in ['A', 'B'] else 'medium',
                'rag_used': False,
                'approach': 'dict_enhanced_positive',
                'grade_detected': grade
            }

            logger.info(f"Dict Enhanced réussi (grade {grade})")
            return result

        except Exception as e:
            logger.error(f"Erreur Dict Enhanced: {str(e)}")
            return {'method': 'dict_enhanced', 'success': False, 'error': str(e)}

    def recommend(self, transcription_text: str, quality_analysis: Dict,
                  extraction_data: Dict = None, audio_analysis_file: Optional[str] = None) -> Dict:
        """Pipeline cascade avec RAG enhanced."""
        logger.info("Démarrage cascade recommandations RAG-Enhanced")

        methods = [
            ('rag_enhanced_qwen', self._recommend_with_rag_qwen, 'RAG + LLM'),
            ('qwen_cascade', self._recommend_with_qwen_only, 'LLM Qwen cascade'),
            ('dict_enhanced', self._recommend_with_dict_enhanced, 'Dict Enhanced (garanti)')
        ]

        attempts = []

        for method_name, method_func, method_desc in methods:
            logger.info(f"Tentative: {method_desc}")

            try:
                result = method_func(transcription_text, quality_analysis, extraction_data, audio_analysis_file)

                attempts.append({
                    'method': method_name,
                    'success': result.get('success', False),
                    'error': result.get('error') if not result.get('success') else None
                })

                if result.get('success'):
                    logger.info(f"{method_desc} réussi")
                    result['cascade_method'] = method_name
                    result['attempts'] = attempts
                    return result

            except Exception as e:
                logger.warning(f"{method_desc} échoué: {str(e)}")
                attempts.append({
                    'method': method_name,
                    'success': False,
                    'error': str(e)
                })

        # Fallback ultime (ne devrait jamais arriver)
        logger.warning("Toutes méthodes échouées - fallback ultime")
        return {
            'method': 'fallback_ultimate_positive',
            'success': True,
            'recommendations': {
                'recommandations_principales': [{
                    'categorie': 'Encouragement',
                    'titre': 'Continuer cette excellente dynamique de progression',
                    'description': 'Votre démarche d\'amélioration continue est remarquable !',
                    'impact': 'high',
                    'facilite_implementation': 'immediate'
                }],
                'resume_conseil': 'Bravo pour cette analyse ! Votre engagement vers l\'excellence est inspirant.',
                'nb_recommandations': 1
            },
            'cascade_method': 'fallback_ultimate_positive',
            'attempts': attempts
        }

# === SAUVEGARDE ===
class RecommendationSaver:
    """Gestionnaire sauvegarde recommandations."""

    def __init__(self, output_dir: str = "output/recommendations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Sauvegarde recommandations: {self.output_dir}")

    def save_results(self, results: Dict) -> str:
        """Sauvegarde recommandations avec timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcription_name = Path(results.get("transcription_file", "unknown")).stem
        method = results.get("cascade_info", {}).get("method_used", "unknown")

        filename = f"recommendations_{method}_{transcription_name}_{timestamp}.json"
        output_path = self.output_dir / filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Recommandations sauvées: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            return ""

# === FONCTION PRINCIPALE ===
def analyze_and_recommend(transcription_file: str, audio_analysis_file: Optional[str] = None,
                         enable_spot_check: bool = False, spot_sample_size: int = 3) -> Dict:
    """Pipeline principal: Transcription TXT + Audio Analysis JSON → Recommandations cascade RAG-Enhanced."""
    transcription_file = Path(transcription_file)

    if not transcription_file.exists():
        return {"error": "file_not_found", "path": str(transcription_file)}

    # Vérification fichier audio optionnel
    if audio_analysis_file:
        audio_path = Path(audio_analysis_file)
        if not audio_path.exists():
            logger.warning(f"Fichier audio analysis non trouvé: {audio_analysis_file} - continuer sans")
            audio_analysis_file = None

    logger.info(f"Pipeline recommandations cascade: {transcription_file.name}")
    if audio_analysis_file:
        logger.info(f"Audio analysis: {Path(audio_analysis_file).name}")

    start_time = datetime.now()

    # 1. Lecture transcription TXT
    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            transcription_text = f.read().strip()

        if not transcription_text:
            return {"error": "empty_transcription", "file": str(transcription_file)}

        logger.info(f"Transcription: {len(transcription_text)} chars")

    except Exception as e:
        logger.error(f"Erreur lecture transcription: {e}")
        return {"error": "transcription_read_failed", "details": str(e)}

    # 2. Extraction contenu meeting pour enrichir le contexte
    extraction_results = {}
    try:
        from src.meeting.extractor import create_meeting_extractor
        extractor = create_meeting_extractor(
            extract_topics=True,
            extract_actions=True,
            extract_decisions=True,
            extract_meeting_type=True
        )
        extraction_results = extractor.extract_meeting_content(transcription_text)
        logger.info(f"Extraction réussie: {len(extraction_results)} méthodes")

    except Exception as e:
        logger.warning(f"Extraction meeting échouée: {e}")

    # Structure pour compatibilité avec RecommendationAnalyzer
    extraction_data = {
        "transcription": {
            "text": transcription_text,
            "quality_analysis": {"grade": "C", "global_score": 70}
        },
        "extraction": extraction_results
    }

    # 3. Analyse qualité meeting
    analyzer = RecommendationAnalyzer()
    quality_analysis = analyzer.analyze_extraction_quality(extraction_data)

    # 4. Recommandations cascade RAG-Enhanced (avec audio)
    cascade = RecommendationCascade()
    recommendation_result = cascade.recommend(transcription_text, quality_analysis, extraction_data, audio_analysis_file)

    # 5. Spot-check strategic (optionnel)
    strategic_spotcheck = {}
    if enable_spot_check:
        try:
            spot_checker = SpotChecker(sample_size=spot_sample_size)
            reco_text = ' '.join([str(r) for r in recommendation_result.get('recommendations', [])])
            samples = spot_checker.strategic_sample(transcription_text,
                                                  ['recommendation', 'amélioration', 'conseil'],
                                                  context_window=400)
            if samples:
                spot_file = spot_checker.save_samples_for_annotation(samples)
                strategic_spotcheck = {
                    "success": True,
                    "samples_count": len(samples),
                    "spot_check_file": spot_file
                }
        except Exception as e:
            logger.warning(f"SpotCheck échoué: {e}")

    total_duration = (datetime.now() - start_time).total_seconds()

    # Extraction métriques finales
    topics_found = 0
    actions_found = 0
    meeting_type = 'Général'

    if extraction_results:
        topics = extraction_results.get('topics', {})
        if isinstance(topics, dict) and 'topics' in topics:
            topics_found = len(topics['topics'])

        actions = extraction_results.get('actions', {})
        if isinstance(actions, dict) and 'actions' in actions:
            actions_found = len(actions['actions'])

        meeting_type_data = extraction_results.get('meeting_type', {})
        if isinstance(meeting_type_data, dict):
            meeting_type = meeting_type_data.get('meeting_type', 'Général')
        elif isinstance(meeting_type_data, str):
            meeting_type = meeting_type_data

    # Résultats consolidés
    return {
        'transcription_file': str(transcription_file),
        'audio_analysis_file': audio_analysis_file,
        'analysis_timestamp': datetime.now().isoformat(),
        'total_duration': total_duration,

        # Analyse qualité
        'quality_analysis': quality_analysis,

        # Recommandations RAG-Enhanced
        'recommendations': recommendation_result,
        'cascade_info': {
            'method_used': recommendation_result.get('cascade_method', 'unknown'),
            'attempts': recommendation_result.get('attempts', []),
            'success': recommendation_result.get('success', False)
        },

        # Context enrichi
        'extraction_enriched': bool(extraction_results),
        'audio_enriched': audio_analysis_file is not None,
        'strategic_spotcheck': strategic_spotcheck,

        # Métriques extraction
        'extraction_metrics': {
            'topics_found': topics_found,
            'actions_found': actions_found,
            'meeting_type': meeting_type
        }
    }

# === AFFICHAGE RÉSULTATS ===
def print_results_summary(results: Dict):
    """Affichage bienveillant des résultats."""
    print("\n" + "="*70)
    print("SUMMORA V3 - RECOMMANDATIONS RAG-ENHANCED")
    print("="*70)

    # Infos générales
    transcription_file = Path(results["transcription_file"]).name
    audio_file = results.get("audio_analysis_file")
    quality_analysis = results["quality_analysis"]

    print(f"\nMeeting analysé: {transcription_file}")
    if audio_file:
        print(f"Audio analysis: {Path(audio_file).name}")
    print(f"Temps analyse: {results['total_duration']:.2f}s")

    # Grade et score
    indicators = quality_analysis.get('quality_indicators', {})
    grade = indicators.get('grade', 'C')
    score = indicators.get('global_score', 50)
    topics_count = indicators.get('topics_count', 0)
    points_count = indicators.get('key_points_count', 0)

    print(f"Grade meeting: {grade} ({score}/100)")
    print(f"Contenu: {topics_count} topics, {points_count} points clés")
    if results.get('audio_enriched'):
        print("✅ Enrichi avec analyse audio")

    # Points forts
    strengths = quality_analysis.get('strengths', [])
    if strengths:
        print(f"\nPOINTS FORTS IDENTIFIÉS:")
        for i, strength in enumerate(strengths, 1):
            print(f"   {i}. {strength}")

    # Recommandations
    recommendations_data = results["recommendations"]
    if recommendations_data.get('success'):
        recommendations = recommendations_data.get('recommendations', {}).get('recommandations_principales', [])
        conseil_synthese = recommendations_data.get('recommendations', {}).get('resume_conseil', '')
        method_used = results.get('cascade_info', {}).get('method_used', 'unknown')

        print(f"\nRECOMMANDATIONS ({method_used.upper()}):")
        for i, reco in enumerate(recommendations[:5], 1):
            if isinstance(reco, dict):
                title = reco.get('titre', reco.get('description', ''))[:60]
                category = reco.get('categorie', 'Conseil')
                print(f"   {i}. [{category}] {title}")

        if conseil_synthese:
            print(f"\nMessage d'encouragement:")
            print(f"   {conseil_synthese}")

        # Indication RAG
        if recommendations_data.get('rag_used'):
            print(f"\nEnrichi avec documents leadership")

    print("="*70)

def _print_reco_only_summary(results: Dict):
    """Affichage compact pour mode --reco-only."""
    print("\n" + "="*50)
    print("RECOMMANDATIONS CASCADE (RAG + LLM + Context)")
    print("="*50)

    # Métriques rapides
    cascade_info = results.get('cascade_info', {})
    method_used = cascade_info.get('method_used', 'unknown')
    extraction_metrics = results.get('extraction_metrics', {})
    meeting_type = extraction_metrics.get('meeting_type', 'Général')
    audio_enriched = "✅ Audio" if results.get('audio_enriched') else ""

    print(f"Méthode utilisée: {method_used}")
    print(f"Type meeting: {meeting_type} {audio_enriched}")
    print(f"Durée: {results.get('total_duration', 0):.2f}s")

    # Recommandations
    recommendations_data = results.get('recommendations', {})
    if recommendations_data.get('success'):
        recommendations = recommendations_data.get('recommendations', {}).get('recommandations_principales', [])
        resume_conseil = recommendations_data.get('recommendations', {}).get('resume_conseil', '')

        print(f"\nRECOMMANDATIONS ({len(recommendations)}):")
        for i, reco in enumerate(recommendations[:8], 1):
            if isinstance(reco, dict):
                category = reco.get('categorie', 'Conseil')
                title = reco.get('titre', reco.get('description', ''))[:70]
                impact = reco.get('impact', 'medium')
                print(f"   {i}. [{category}] {title} (Impact: {impact})")

        if resume_conseil:
            print(f"\nConseil prioritaire:")
            print(f"   {resume_conseil}")

    print("="*50)

# === CLI PRINCIPAL ===
def main():
    """Interface CLI simplifiée."""
    parser = argparse.ArgumentParser(
        description="Summora V3 - Recommandations cascade RAG-Enhanced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
python scripts/main_reco.py transcription.txt                           # Analyse complète avec sauvegarde
python scripts/main_reco.py transcription.txt --audio-data audio_analysis.json  # Avec analyse audio
python scripts/main_reco.py transcription.txt --reco-only              # Recommandations seulement (RAG + LLM + Context)
python scripts/main_reco.py transcription.txt --enable-spot-check       # Avec QA spot-check
        """
    )

    parser.add_argument("transcription_file", help="Fichier transcription TXT")
    parser.add_argument("--audio-data", help="Fichier analyse audio JSON (optionnel)")
    parser.add_argument("--reco-only", action="store_true", help="Mode recommandations seulement (pas de sauvegarde)")
    parser.add_argument("--enable-spot-check", action="store_true", help="Active spot-check QA")
    parser.add_argument("--spot-sample", type=int, default=3, help="Échantillons spot-check")
    parser.add_argument("--output", "-o", help="Répertoire de sortie")
    parser.add_argument("--no-save", action="store_true", help="Ne sauvegarde pas")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    parser.add_argument("--quiet", "-q", action="store_true", help="Mode silencieux")

    args = parser.parse_args()

    # Configuration logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Vérification fichier transcription
    transcription_path = Path(args.transcription_file)
    if not transcription_path.exists():
        logger.error(f"Fichier non trouvé: {args.transcription_file}")
        return 1

    if not transcription_path.suffix.lower() == '.txt':
        logger.error(f"Format non supporté: {transcription_path.suffix}. Utilisez un fichier .txt")
        return 1

    # Vérification fichier audio optionnel
    audio_analysis_file = None
    if args.audio_data:
        audio_path = Path(args.audio_data)
        if not audio_path.exists():
            logger.error(f"Fichier audio analysis non trouvé: {args.audio_data}")
            return 1
        if not audio_path.suffix.lower() == '.json':
            logger.error(f"Format audio analysis non supporté: {audio_path.suffix}. Utilisez un fichier .json")
            return 1
        audio_analysis_file = str(audio_path)
        logger.info(f"Audio analysis: {audio_path.name}")

    try:
        # Pipeline cascade complet
        results = analyze_and_recommend(
            str(transcription_path),
            audio_analysis_file=audio_analysis_file,
            enable_spot_check=args.enable_spot_check,
            spot_sample_size=args.spot_sample
        )

        if "error" in results:
            logger.error(f"Erreur pipeline: {results['error']}")
            return 1

        # Mode reco-only : affichage compact sans sauvegarde
        if args.reco_only:
            _print_reco_only_summary(results)
            return 0

        # Mode complet : sauvegarde + affichage
        if not args.no_save:
            output_dir = args.output or "output/recommendations"
            saver = RecommendationSaver(output_dir)
            saved_file = saver.save_results(results)
            if saved_file:
                logger.info(f"Fichier sauvé: {saved_file}")

        if not args.quiet:
            print_results_summary(results)

        return 0

    except KeyboardInterrupt:
        logger.info("Analyse interrompue")
        return 1
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
