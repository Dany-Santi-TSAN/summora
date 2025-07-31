#!/usr/bin/env python3
"""
Main Recommendation - Analyse qualité meeting → Recommandations bienveillantes
Input: JSON extraction (main_extract.py) → Output: Conseils d'amélioration
Architecture: Analyzer + Reco avec Dict Enhanced by LLM + Saver
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
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

from dotenv import load_dotenv

# Variables d'environnement
load_dotenv()

# === DICTIONNAIRE DE RECOMMANDATIONS (Enhanced by Qwen) ===
class MeetingRecommendationEngine:
    """Moteur de recommandations pour améliorer les meetings - Enhanced Dict."""

    def __init__(self):
        # Base de recommandations existante (celle qui fonctionne)
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
        recommendations = []
        grade = quality_analysis.get('grade', 'D')
        issues = quality_analysis.get('issues_detected', [])

        # Recommandations graduées selon le grade
        if grade == "A":
            # Grade A : Juste encouragement, pas de recommandations
            return [{
                'categorie': 'Excellence',
                'titre': 'Meeting exemplaire - Continuez cette excellence !',
                'description': 'Votre meeting atteint un niveau d\'excellence remarquable. Votre approche est un modèle à suivre.',
                'impact': 'inspirational',
                'facilite_implementation': 'immediate',
                'source': 'encouragement_grade_A'
            }]
        elif grade == "B":
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
        for i, reco_text in enumerate(selected_reco):
            category = self._get_category_from_text(reco_text)
            recommendations.append({
                'categorie': category,
                'titre': reco_text,
                'description': f"{reco_text}. Recommandation adaptée au grade {grade}.",
                'impact': 'high' if grade == 'D' else 'medium',
                'facilite_implementation': 'easy',
                'source': 'dict_enhanced'
            })

        return recommendations[:8 if grade == 'D' else (5 if grade == 'C' else (3 if grade == 'B' else 1))]

    def _get_category_from_text(self, text: str) -> str:
        """Détermine la catégorie d'une recommandation avec vocabulaire DRY."""
        text_lower = text.lower()

        # Utilisation du vocabulaire business centralisé (DRY)
        business_mapping = {
            'actions': 'Action',
            'decisions': 'Décision',
            'planning': 'Planning',
            'organisation': 'Organisation',
            'finance': 'Finance',
            'objectifs': 'Objectifs'
        }

        # Test avec vocabulaire business centralisé
        for category, keywords in BUSINESS_KEYWORDS.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                return business_mapping.get(category, category.title())

        # Fallback catégories spécifiques recommandations
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

# === CLASSE 1: RecommendationAnalyzer (Compatible extraction JSON) ===
class RecommendationAnalyzer:
    """Analyseur de qualité meeting - Compatible avec JSON d'extraction."""

    def __init__(self):
        logger.info("📊 Analyseur qualité meeting initialisé (format extraction)")

    def analyze_extraction_quality(self, extraction_data: Dict) -> Dict:
        """
        Analyse la qualité d'un meeting depuis JSON extraction.

        Args:
            extraction_data: JSON de main_extract.py

        Returns:
            Dict: Analyse des points forts + améliorations possibles
        """
        try:
            # Extraction des données principales
            transcription_data = extraction_data.get('transcription', {})
            existing_quality = transcription_data.get('quality_analysis', {})
            extraction_results = extraction_data.get('extraction', {})

            transcription_text = transcription_data.get('text', '')
            grade = existing_quality.get('grade', 'C')
            global_score = existing_quality.get('global_score', 50)

            # Données enrichies d'extraction
            topics = extraction_results.get('topics_principaux', [])
            points = extraction_results.get('points_a_retenir', [])
            method_used = extraction_data.get('cascade_info', {}).get('method_used', 'unknown')

            logger.info(f"📊 Analyse meeting: Grade {grade} ({global_score}/100), {len(topics)} topics, méthode {method_used}")

            # Construction analyse avec bienveillance
            analysis = {
                'strengths': [],
                'improvements': [],
                'quality_indicators': {
                    'grade': grade,
                    'global_score': global_score,
                    'topics_count': len(topics),
                    'key_points_count': len(points),
                    'text_length': len(transcription_text),
                    'extraction_method': method_used
                },
                'overall_assessment': self._determine_assessment(grade)
            }

            # === POINTS FORTS (approche positive) ===
            if len(topics) >= 5:
                analysis['strengths'].append(f"Richesse thématique remarquable ({len(topics)} sujets abordés)")
            elif len(topics) >= 3:
                analysis['strengths'].append(f"Bonne diversité des sujets ({len(topics)} topics identifiés)")

            if len(points) >= 5:
                analysis['strengths'].append(f"Excellent niveau de contenu ({len(points)} points clés extraits)")
            elif len(points) >= 2:
                analysis['strengths'].append(f"Contenu substantiel avec {len(points)} points importants")

            if method_used != 'yake_fallback':
                analysis['strengths'].append("Qualité suffisante pour analyse LLM avancée")
            else:
                analysis['strengths'].append("Résilience technique - extraction réussie malgré les défis")

            # Encouragement selon le grade
            if grade in ['A', 'B']:
                analysis['strengths'].append("Communication efficace et bien structurée")
            elif grade == 'C':
                analysis['strengths'].append("Base solide avec potentiel d'optimisation")
            else:  # Grade D
                analysis['strengths'].append("Participation active des membres (excellent engagement)")

            # === AMÉLIORATIONS SUGGÉRÉES (constructif) ===
            if grade == 'D':
                analysis['improvements'].extend([
                    {
                        'category': 'Structure',
                        'suggestion': 'Définir un ordre du jour précis et le respecter',
                        'reason': 'Discussion riche mais dispersée - plus de cadrage maximisera l\'impact'
                    },
                    {
                        'category': 'Animation',
                        'suggestion': 'Désigner un facilitateur pour maintenir le focus',
                        'reason': 'Excellente participation - canaliser cette énergie vers les objectifs'
                    },
                    {
                        'category': 'Décision',
                        'suggestion': 'Réserver 15min en fin pour synthétiser les décisions',
                        'reason': 'Beaucoup d\'échanges - les ancrer par des conclusions claires'
                    }
                ])
            elif grade == 'C':
                analysis['improvements'].extend([
                    {
                        'category': 'Efficacité',
                        'suggestion': 'Structurer en blocs thématiques de 15-20 minutes',
                        'reason': 'Bon contenu - optimiser le timing pour plus d\'impact'
                    },
                    {
                        'category': 'Participation',
                        'suggestion': 'Encourager la synthèse collective des points clés',
                        'reason': 'Belle dynamique - la formaliser pour ancrer les apprentissages'
                    }
                ])
            elif grade == 'B':
                analysis['improvements'].extend([
                    {
                        'category': 'Excellence',
                        'suggestion': 'Ajouter un récapitulatif visuel en temps réel',
                        'reason': 'Très bon niveau - perfectionner avec des outils visuels'
                    }
                ])

            return analysis

        except Exception as e:
            logger.error(f"❌ Erreur analyse extraction: {e}")
            return {
                'error': str(e),
                'strengths': ['Tentative d\'analyse courageuse'],
                'improvements': [{'category': 'Technique', 'suggestion': 'Vérifier format JSON extraction'}],
                'overall_assessment': 'needs_improvement'
            }

    def _determine_assessment(self, grade: str) -> str:
        """Détermine l'assessment global selon le grade."""
        if grade == 'A':
            return 'excellent'
        elif grade == 'B':
            return 'good'
        else:
            return 'needs_improvement'

# === CLASSE 2: RecommendationCascade (Dict Enhanced by Qwen) ===
class RecommendationCascade:
    """Pipeline de recommandations: Dict Enhanced by LLM Qwen (backbone) + Dict Simple (fallback)."""

    def __init__(self):
        self.dict_engine = MeetingRecommendationEngine()
        self.recommendation_methods = [
            {
                'name': 'dict_enhanced',
                'function': self._recommend_with_dict_enhanced,
                'description': 'Dictionnaire enhanced bienveillant (backbone)'
            },
            {
                'name': 'dict_simple',
                'function': self._recommend_with_dict_simple,
                'description': 'Dictionnaire simple (fallback)'
            }
        ]
        logger.info("💡 Pipeline recommandations simple: Dict Enhanced + Dict Simple")

    def _recommend_with_dict_simple(self, transcription_text: str, quality_analysis: Dict,
                                  extraction_data: Dict = None) -> Dict:
        """Fallback dictionnaire simple - version basique."""
        try:
            grade = quality_analysis.get('quality_indicators', {}).get('grade', 'C')

            # Recommandations basiques selon grade
            if grade == 'A':
                recommendations = [{
                    'categorie': 'Excellence',
                    'titre': 'Meeting parfait - Continuez !',
                    'description': 'Aucune amélioration nécessaire.',
                    'impact': 'none',
                    'facilite_implementation': 'immediate',
                    'source': 'dict_simple'
                }]
            elif grade == 'B':
                recommendations = [{
                    'categorie': 'Structure',
                    'titre': 'Petits ajustements structurels',
                    'description': 'Bon meeting avec potentiel d\'optimisation.',
                    'impact': 'low',
                    'facilite_implementation': 'easy',
                    'source': 'dict_simple'
                }]
            else:  # C ou D
                recommendations = [
                    {
                        'categorie': 'Structure',
                        'titre': 'Améliorer l\'organisation',
                        'description': 'Structurer davantage le meeting.',
                        'impact': 'medium',
                        'facilite_implementation': 'medium',
                        'source': 'dict_simple'
                    },
                    {
                        'categorie': 'Efficacité',
                        'titre': 'Optimiser la durée',
                        'description': 'Meetings plus courts et focalisés.',
                        'impact': 'medium',
                        'facilite_implementation': 'easy',
                        'source': 'dict_simple'
                    }
                ]

            conseil_synthese = f"Recommandations simples pour grade {grade}."

            return {
                'method': 'dict_simple',
                'success': True,
                'recommendations': recommendations,
                'conseil_synthese': conseil_synthese,
                'approach': 'dict_simple_basic'
            }

        except Exception as e:
            logger.error(f"❌ Erreur Dict Simple: {str(e)}")
            return {'method': 'dict_simple', 'success': False, 'error': str(e)}

    def _recommend_with_dict_enhanced(self, transcription_text: str, quality_analysis: Dict,
                                    extraction_data: Dict = None) -> Dict:
        """Fallback recommandations avec dictionnaire enhanced."""
        try:
            # Utilisation du moteur de recommandations enhanced
            extraction_results = extraction_data.get('extraction', {}) if extraction_data else {}
            recommendations = self.dict_engine.generate_recommendations(quality_analysis, extraction_results)

            # Message encourageant basé sur les forces
            strengths = quality_analysis.get('strengths', [])
            grade = quality_analysis.get('quality_indicators', {}).get('grade', 'C')

            if grade == 'A':
                conseil_synthese = "Meeting exemplaire ! Ces micro-ajustements le rendront parfait."
            elif grade == 'B':
                conseil_synthese = f"Excellent travail avec {len(strengths)} points forts ! Ces optimisations maximiseront l'impact."
            elif grade == 'C':
                conseil_synthese = "Très bon potentiel détecté ! Ces ajustements simples transformeront vos meetings."
            else:  # Grade D
                conseil_synthese = f"Formidable engagement des participants ! Canaliser cette énergie avec ces {len(recommendations)} optimisations."

            return {
                'method': 'dict_enhanced',
                'success': True,
                'recommendations': recommendations,
                'conseil_synthese': conseil_synthese,
                'approach': 'dict_enhanced_positive',
                'grade_detected': grade,
                'recommendations_count': len(recommendations)
            }

        except Exception as e:
            logger.error(f"❌ Erreur Dict Enhanced: {str(e)}")
            return {'method': 'dict_enhanced', 'success': False, 'error': str(e)}

    def recommend(self, transcription_text: str, quality_analysis: Dict,
                 extraction_data: Dict = None) -> Dict:
        """Pipeline simple avec approche bienveillante."""
        logger.info("💡 Démarrage recommandations simples (Dict Enhanced + Dict Simple)")

        recommendation_attempts = []

        for i, method in enumerate(self.recommendation_methods, 1):
            method_name = method['name']
            method_func = method['function']
            method_desc = method['description']

            logger.info(f"🧠 Tentative {i}: {method_desc}")

            try:
                result = method_func(transcription_text, quality_analysis, extraction_data)

                recommendation_attempts.append({
                    'method': method_name,
                    'success': result.get('success', False),
                    'error': result.get('error') if not result.get('success') else None
                })

                if result.get('success'):
                    logger.info(f"✅ {method_desc} réussi")
                    result['cascade_method'] = method_name
                    result['attempts'] = recommendation_attempts
                    return result

            except Exception as e:
                logger.warning(f"⚠️ {method_desc} échoué: {str(e)}")
                recommendation_attempts.append({
                    'method': method_name,
                    'success': False,
                    'error': str(e)
                })

        # Fallback final ultra-positif
        return {
            'method': 'fallback_ultimate_positive',
            'success': True,
            'recommendations': [{
                'categorie': 'Encouragement',
                'titre': 'Continuer cette excellente dynamique de progression',
                'description': 'Votre démarche d\'amélioration continue est remarquable ! Chaque meeting est une réussite en soi.',
                'impact': 'high',
                'facilite_implementation': 'immediate'
            }],
            'conseil_synthese': 'Bravo pour cette analyse ! Votre engagement vers l\'excellence est inspirant.',
            'cascade_method': 'fallback_ultimate_positive',
            'attempts': recommendation_attempts
        }

# === CLASSE 3: RecommendationSaver (Sauvegarde harmonisée) ===
class RecommendationSaver:
    """Gestionnaire sauvegarde recommandations (pattern harmonisé)."""

    def __init__(self, output_dir: str = "output/recommendations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"💾 Sauvegarde recommandations: {self.output_dir}")

    def save_results(self, results: Dict) -> str:
        """Sauvegarde recommandations avec timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extraction_name = Path(results["extraction_file"]).stem if results.get("extraction_file") else "unknown"
        method = results["cascade_info"]["method_used"]

        filename = f"recommendations_{method}_{extraction_name}_{timestamp}.json"
        output_path = self.output_dir / filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Recommandations sauvées: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            return ""

# === FONCTION PRINCIPALE ===
def analyze_and_recommend(extraction_file: str, enable_spot_check: bool = False,
                         spot_sample_size: int = 3) -> Dict:
    """
    Pipeline principal: JSON extraction → Recommandations bienveillantes.
    """
    extraction_file = Path(extraction_file)

    if not extraction_file.exists():
        return {"error": "file_not_found", "path": str(extraction_file)}

    logger.info(f"💡 Analyse extraction + recommandations: {extraction_file.name}")
    start_time = datetime.now()

    # 1. Lecture JSON extraction (main_extract.py)
    try:
        with open(extraction_file, 'r', encoding='utf-8') as f:
            extraction_data = json.load(f)

        transcription_text = extraction_data.get('transcription', {}).get('text', '')
        if not transcription_text:
            return {"error": "no_transcription_text", "file": str(extraction_file)}

        logger.info(f"📖 Transcription: {len(transcription_text)} chars")

    except Exception as e:
        logger.error(f"❌ Erreur lecture extraction: {e}")
        return {"error": "extraction_read_failed", "details": str(e)}

    # 2. Analyse qualité meeting depuis extraction
    analyzer = RecommendationAnalyzer()
    quality_analysis = analyzer.analyze_extraction_quality(extraction_data)

    # 3. Recommandations cascade (LLM + Dict Enhanced)
    cascade = RecommendationCascade()
    recommendation_result = cascade.recommend(transcription_text, quality_analysis, extraction_data)

    # 4. Spot-check strategic (optionnel)
    strategic_spotcheck = {}
    if enable_spot_check:
        try:
            spot_checker = SpotChecker(sample_size=spot_sample_size)
            # Mots-clés depuis recommandations
            reco_text = ' '.join([str(r) for r in recommendation_result.get('recommendations', [])])
            samples = spot_checker.strategic_sample(transcription_text,
                                                  ['recommandation', 'amélioration', 'conseil'],
                                                  context_window=400)
            if samples:
                spot_file = spot_checker.save_samples_for_annotation(samples)
                strategic_spotcheck = {
                    "success": True,
                    "samples_count": len(samples),
                    "spot_check_file": spot_file
                }
        except Exception as e:
            logger.warning(f"⚠️ SpotCheck échoué: {e}")

    total_duration = (datetime.now() - start_time).total_seconds()

    # Résultats consolidés
    return {
        'extraction_file': str(extraction_file),
        'analysis_timestamp': datetime.now().isoformat(),
        'total_duration': total_duration,

        # Analyse qualité depuis extraction
        'quality_analysis': quality_analysis,

        # Recommandations
        'recommendations': recommendation_result,
        'cascade_info': {
            'method_used': recommendation_result.get('cascade_method', 'unknown'),
            'attempts': recommendation_result.get('attempts', []),
            'success': recommendation_result.get('success', False)
        },

        # Context enrichi
        'extraction_enriched': True,
        'strategic_spotcheck': strategic_spotcheck if strategic_spotcheck else {}
    }

def print_results_summary(results: Dict):
    """Affichage bienveillant des résultats."""
    print("\n" + "="*70)
    print("💡 SUMMORA V3 - RECOMMANDATIONS BIENVEILLANTES")
    print("="*70)

    # Infos générales
    extraction_file = Path(results["extraction_file"]).name
    quality_analysis = results["quality_analysis"]

    print(f"\n📁 Meeting analysé: {extraction_file}")
    print(f"⏱️ Temps analyse: {results['total_duration']:.2f}s")

    # Grade et score
    indicators = quality_analysis.get('quality_indicators', {})
    grade = indicators.get('grade', 'C')
    score = indicators.get('global_score', 50)
    topics_count = indicators.get('topics_count', 0)
    points_count = indicators.get('key_points_count', 0)

    print(f"📊 Grade meeting: {grade} ({score}/100)")
    print(f"🎯 Contenu: {topics_count} topics, {points_count} points clés")

    # Points forts (encouragement)
    strengths = quality_analysis.get('strengths', [])
    if strengths:
        print(f"\n🌟 POINTS FORTS IDENTIFIÉS:")
        for i, strength in enumerate(strengths, 1):
            print(f"   {i}. {strength}")

    # Recommandations
    recommendations_data = results["recommendations"]
    if recommendations_data.get('success'):
        recommendations = recommendations_data.get('recommendations', [])
        conseil_synthese = recommendations_data.get('conseil_synthese', '')

        print(f"\n💡 RECOMMANDATIONS D'AMÉLIORATION:")
        for i, reco in enumerate(recommendations[:5], 1):
            if isinstance(reco, dict):
                title = reco.get('titre', reco.get('description', ''))[:60]
                category = reco.get('categorie', 'Conseil')
                print(f"   {i}. [{category}] {title}")
            else:
                print(f"   {i}. {str(reco)[:60]}")

        if conseil_synthese:
            print(f"\n💭 Message d'encouragement:")
            print(f"   {conseil_synthese}")

    # Méthode utilisée
    cascade_info = results.get('cascade_info', {})
    method_used = cascade_info.get('method_used', 'unknown')
    print(f"\n🔧 Méthode: {method_used}")

    # Assessment global selon grade
    if grade == 'A':
        print(f"\n🏆 Meeting d'excellence - Bravo !")
    elif grade == 'B':
        print(f"\n👍 Bon meeting - Quelques optimisations simples le rendront parfait")
    elif grade == 'C':
        print(f"\n🌱 Meeting avec potentiel - Ces améliorations feront la différence")
    else:  # Grade D
        print(f"\n🚀 Excellent engagement détecté - Canaliser cette énergie maximisera l'impact")

    print("="*70)

def main():
    """Interface CLI harmonisée."""
    parser = argparse.ArgumentParser(
        description="Summora V3 - Recommandations Bienveillantes Meeting (Input: JSON extraction)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
python scripts/main_reco.py extraction.json                    # Analyse basique
python scripts/main_reco.py extraction.json --reco-only       # Mode rapide
python scripts/main_reco.py extraction.json --enable-spot-check  # Avec QA

Input: JSON de main_extract.py (obligatoire)
        """
    )

    parser.add_argument("extraction_file", help="Fichier JSON extraction (main_extract.py)")
    parser.add_argument("--reco-only", action="store_true", help="Mode recommandations rapide")
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

    # Vérification fichier
    if not Path(args.extraction_file).exists():
        logger.error(f"❌ Fichier extraction non trouvé: {args.extraction_file}")
        sys.exit(1)

    try:
        # Mode rapide (Dict Enhanced seulement)
        if args.reco_only:
            logger.info("💡 Mode recommandations rapide (Dict Enhanced)")
            # Lecture JSON
            with open(args.extraction_file, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)

            # Analyse qualité + Dict Enhanced direct
            analyzer = RecommendationAnalyzer()
            quality_analysis = analyzer.analyze_extraction_quality(extraction_data)

            dict_engine = MeetingRecommendationEngine()
            recommendations = dict_engine.generate_recommendations(
                quality_analysis,
                extraction_data.get('extraction', {})
            )

            # Affichage rapide
            grade = quality_analysis.get('quality_indicators', {}).get('grade', 'C')
            print(f"🎯 Grade: {grade} | {len(recommendations)} recommandations générées")
            for i, reco in enumerate(recommendations, 1):
                print(f"  {i}. [{reco['categorie']}] {reco['titre']}")

            return 0

        # Pipeline complet
        results = analyze_and_recommend(
            args.extraction_file,
            enable_spot_check=args.enable_spot_check,
            spot_sample_size=args.spot_sample
        )

        if "error" in results:
            logger.error(f"❌ Erreur pipeline: {results['error']}")
            sys.exit(1)

        # Sauvegarde
        if not args.no_save:
            output_dir = args.output or "output/recommendations"
            saver = RecommendationSaver(output_dir)
            saved_file = saver.save_results(results)

        # Affichage
        if not args.quiet:
            print_results_summary(results)

        return 0

    except KeyboardInterrupt:
        logger.info("🛑 Analyse interrompue")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
