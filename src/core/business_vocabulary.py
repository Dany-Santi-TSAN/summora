"""
Vocabulaire business centralisé pour Summora Meeting Analyzer
Single source of truth pour tous les mots-clés métier
"""
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

# === Vocabulaire business meetings ===
# Core logic pour transcriber, extractor et metrics

BUSINESS_KEYWORDS = {
    'actions': [
        'action', 'tâche', 'faire', 'réaliser', 'livrer', 'livrable', 'assigner',
        'responsable', 'charge de', 'doit', 'va faire', 'prendre en charge',
        'todo', 'à faire', 'next step', 'prochaine étape'
    ],
    'decisions': [
        'décision', 'décider', 'valider', 'approuver', 'trancher',
        'choix', 'opter pour', 'retenir', 'adopter', 'accepter',
        'refuser', 'rejeter', 'arbitrage', 'conclusion'
    ],
    'planning': [
        'planning', 'délai', 'échéance', 'deadline', 'calendrier',
        'roadmap', 'timeline', 'avant', 'pour le', 'date limite',
        'livraison', 'fin', 'début', 'lancement'
    ],
    'questions': [
        'question', 'souci', 'problème', 'blocage', 'difficulté',
        'bug', 'issue', 'point bloquant', 'interrogation', 'clarification'
    ],
    'agreement': [
        'd\'accord', 'ok', 'parfait', 'exactement', 'entendu',
        'validé', 'approuvé', 'c\'est bon', 'ça marche', 'deal'
    ],
    'finance': [
        'budget', 'coût', 'euros', 'millions', 'investissement',
        'financement', 'rentabilité', 'roi', 'chiffre', 'revenus',
        'prix', 'tarif', 'facture', 'devis'
    ],
    'organisation': [
        'équipe', 'équipes', 'manager', 'chef', 'direction',
        'département', 'service', 'ressource', 'humaines',
        'collaborateur', 'collègue', 'hiérarchie'
    ],
    'objectifs': [
        'objectif', 'objectifs', 'but', 'goal', 'cible',
        'résultat', 'performance', 'kpi', 'métrique',
        'indicateur', 'mesure', 'succès'
    ],
    'meeting_type': {
        'brainstorming': ['brainstorm', 'idées', 'créativité', 'innovation', 'idéation'],
        'copil': ['copil', 'pilotage', 'stratégie', 'gouvernance', 'arbitrage'],
        'rétrospective': ['rétro', 'amélioration', 'sprint', 'feedback', "retour d'expérience", 'post-mortem'],
        'client': ['client', 'démonstration', 'livrable', 'présentation', 'business review', 'closing'],
        'conflit': ['conflit', 'désaccord', 'tension', 'médiation', 'litige'],
        'décisionnelle': ['décision', 'choix', 'validation', 'approuver', 'trancher'],
        'kickoff': ['kick-off', 'lancement', 'démarrage', 'initiation', 'présentation projet'],
        'cloture': ['clôture', 'bilan', 'fin de projet', 'conclusion'],
        'one_to_one': ['one-to-one', '1:1', 'entretien individuel', 'suivi personnel'],
        'formation': ['formation', 'apprentissage', 'transfert de connaissances', 'atelier pédagogique', 'onboarding'],
        'team_building': ['team building', 'cohésion', "activité d'équipe", 'dynamique de groupe'],
        'information': ['information', 'annonce', 'communication', 'news', 'all hands'],
        'standup': ['stand-up', 'daily', 'point rapide', 'synchronisation', '15 minutes'],
        'atelier': ['atelier collaboratif', 'workshop', 'co-création', 'travail de groupe'],
        'prospective': ['prospective', 'innovation', 'anticipation', 'futur', 'tendances','pivot'],
        'crise': ['comité de crise', 'urgence', 'situation critique', 'problème majeur', 'réunion exceptionnelle'],
        'negociation': ['négociation', 'accord', 'transaction', 'partenaires', 'deal'],
        'budget': ['budget', 'financier', 'chiffres', 'coût', 'ressources'],
        'produit': ['produit', 'roadmap', 'évolution', 'release', 'fonctionnalités','feature','pivot','mvp'],
        'marketing': ['marketing', 'campagne', 'communication', 'publicité', 'funnel de conversion', 'taux de'],
        'commerciale': ['commercial', 'ventes', 'pipeline', 'prospects', 'clients'],
        'rd': ['R&D', 'recherche', 'développement', 'prototype', 'innovation technique', 'poc','backlog'],
        'qualite': ['qualité', 'norme', 'processus', 'audit'],
        'securite': ['sécurité', 'risque', 'protection', 'protocoles']
}

}

# === Fonctions utilitaires ===

def get_all_business_keywords() -> Set[str]:
    """
    Retourne tous les mots-clés business en un seul ensemble.

    Retourne:
        Set[str]: Ensemble de tous les mots-clés business
    """
    all_keywords = set()
    for category, keywords in BUSINESS_KEYWORDS.items():
        all_keywords.update(word.lower() for word in keywords)

    logger.info(f"📋 Vocabulaire business total: {len(all_keywords)} mots-clés")
    return all_keywords

def get_business_keywords_by_category(category: str) -> List[str]:
    """
    Retourne les mots-clés d'une catégorie spécifique.

    Args:
        category: Nom de la catégorie ('actions', 'decisions', etc.)

    Retourne:
        List[str]: Liste des mots-clés de la catégorie
    """
    if category not in BUSINESS_KEYWORDS:
        available = list(BUSINESS_KEYWORDS.keys())
        raise ValueError(f"Catégorie '{category}' inconnue. Disponibles: {available}")

    return BUSINESS_KEYWORDS[category].copy()

def get_business_categories() -> List[str]:
    """
    Retourne la liste des catégories business disponibles.

    Retourne:
        List[str]: Noms des catégories
    """
    return list(BUSINESS_KEYWORDS.keys())

def is_business_keyword(word: str) -> bool:
    """
    Vérifie si un mot est un mot-clé business.

    Args:
        word: Mot à vérifier

    Retourne:
        bool: True si le mot est dans le vocabulaire business
    """
    all_keywords = get_all_business_keywords()
    return word.lower() in all_keywords

def get_keyword_category(word: str) -> str:
    """
    Trouve la catégorie d'un mot-clé business.

    Args:
        word: Mot-clé à categoriser

    Retourne:
        str: Nom de la catégorie ou 'unknown' si pas trouvé
    """
    word_lower = word.lower()

    for category, keywords in BUSINESS_KEYWORDS.items():
        if word_lower in [kw.lower() for kw in keywords]:
            return category

    return 'unknown'

# === Compatibilité avec code existant ===

def get_meeting_keywords_legacy() -> Dict[str, List[str]]:
    """
    Retourne les keywords dans le format legacy pour compatibilité.
    Permet migration douce depuis l'ancien format.

    Retourne:
        Dict: Format compatible avec ActionDecisionDetector existant
    """
    return BUSINESS_KEYWORDS.copy()

# === Métadonnées ===

BUSINESS_VOCAB_VERSION = "2.0"
BUSINESS_VOCAB_TOTAL_WORDS = len(get_all_business_keywords())

def get_business_vocab_info() -> Dict:
    """
    Retourne les métadonnées du vocabulaire business.

    Retourne:
        Dict: Informations sur le vocabulaire
    """
    return {
        'version': BUSINESS_VOCAB_VERSION,
        'total_keywords': BUSINESS_VOCAB_TOTAL_WORDS,
        'categories': get_business_categories(),
        'words_per_category': {
            cat: len(keywords) for cat, keywords in BUSINESS_KEYWORDS.items()
        }
    }

if __name__ == "__main__":
    # Test et affichage info
    info = get_business_vocab_info()
    print("📋 VOCABULAIRE BUSINESS SUMMORA")
    print(f"Version: {info['version']}")
    print(f"Total mots-clés: {info['total_keywords']}")
    print(f"Catégories: {info['categories']}")
    print("\nMots par catégorie:")
    for cat, count in info['words_per_category'].items():
        print(f"  • {cat}: {count} mots")
