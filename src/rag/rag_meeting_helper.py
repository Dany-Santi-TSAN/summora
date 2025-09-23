"""
Simple RAG context function pour enrichir Qwen
MVP approche - keep it simple and working
"""
import logging
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Templates questions RAG alignés avec main_reco.py
BUSINESS_QUESTION_TEMPLATES = {
    'structure': [
        "Comment définir un ordre du jour clair avant le meeting?"
        ,"Quelles techniques pour mieux structurer une réunion?"
        ,"Comment organiser efficacement les phases d'un meeting?"
        ,"Quelles sont les meilleures pratiques pour animer une équipe?"
    ],
    'participation': [
        "Comment encourager chaque participant à s'exprimer?"
        ,"Quelles méthodes pour stimuler la discussion en réunion?"
        ,"Comment utiliser la technique du tour de table efficacement?"
        ,"Comment améliorer la communication en équipe?"
        ,"Quelles techniques pour favoriser l'expression de chacun?"
        ,"Comment gérer les conflits de communication?"
        ,"Comment faire participer les plus timides?"
    ],
    'efficacite': [
        "Comment limiter la durée des meetings pour plus d'efficacité?"
        ,"Quelles techniques pour maintenir le rythme en réunion?"
        ,"Comment optimiser la gestion du temps en meeting?"
        ,"Comment améliorer le leadership en réunion?"
    ],
    'decisions': [
        "Comment formuler clairement les décisions prises?"
        ,"Quelles méthodes pour assigner responsables et échéances?"
        ,"Comment s'assurer du suivi des décisions?"
        ,"Quels outils sont efficaces pour améliorer la prise de décision?"
    ],
    'technique': [
        "Comment améliorer la qualité audio des réunions?"
        ,"Quelles bonnes pratiques techniques pour les meetings?"
        ,"Comment optimiser l'environnement technique de réunion?"
    ]
}

def get_rag_context(meeting_analysis: Dict, docs_path: str = "data/rag/documents") -> Dict:
    """
    Fonction simple pour récupérer contexte RAG selon analyse meeting.

    Args:
        meeting_analysis: Analyse qualité du meeting (grade, etc.)
        docs_path: Chemin vers documents RAG

    Returns:
        Dict: Contexte RAG simple ou vide si échec
    """
    try:
        # Import RAG seulement si nécessaire
        from src.rag.rag import RAGlight

        # Quick check si docs existent
        docs_dir = Path(docs_path)
        if not docs_dir.exists() or not list(docs_dir.glob("*.pdf")):
            logger.warning(f"❌ Pas de docs RAG dans {docs_path}")
            return {"success": False, "context": "", "sources": []}

        # Init RAG simple
        rag = RAGlight(docs_path)
        rag.build_rag_pipeline()

        # Question selon grade meeting
        grade = meeting_analysis.get('quality_indicators', {}).get('grade', 'C')

        if grade == 'D':
            question = "Comment transformer une réunion inefficace en meeting productif ?"
        elif grade == 'C':
            question = "Comment optimiser une réunion moyennement efficace ?"
        else:  # A ou B
            question = "Comment maintenir l'excellence d'une réunion déjà réussie ?"

        # Query RAG
        result = rag.query_rag(question, top_k=3)

        if result.get('context'):
            # Limiter contexte (pour pas exploser les tokens LLM)
            context = result['context'][:2000] + "..." if len(result['context']) > 2000 else result['context']

            logger.info(f"✅ RAG contexte: {len(context)} chars, grade {grade}")
            return {
                "success": True,
                "context": context,
                "sources": result.get('sources', []),
                "question_used": question,
                "grade": grade
            }
        else:
            return {"success": False, "context": "", "sources": []}

    except Exception as e:
        logger.warning(f"⚠️ RAG context failed: {e}")
        return {"success": False, "context": "", "sources": [], "error": str(e)}

def enhance_qwen_with_rag(transcription_text: str, extraction_data: Dict,
                         meeting_analysis: Dict) -> Dict:
    """
    Enrichit extraction_data avec contexte RAG pour Qwen.
    Simple wrapper qui ajoute RAG context si disponible.

    Args:
        transcription_text: Transcription meeting
        extraction_data: Données extraction existantes
        meeting_analysis: Analyse qualité meeting

    Retourne:
        Dict: extraction_data enrichi avec RAG
    """
    enhanced_data = extraction_data.copy() if extraction_data else {}

    # Tentative RAG context
    rag_context = get_rag_context(meeting_analysis)

    if rag_context.get('success'):
        enhanced_data['rag_context'] = {
            'leadership_insights': rag_context['context'],
            'sources': rag_context['sources'],
            'meeting_grade': rag_context['grade']
        }
        logger.info(f"💡 Qwen enrichi avec RAG: {len(rag_context['context'])} chars")
    else:
        logger.info("💡 Qwen sans enrichissement RAG")

    return enhanced_data
