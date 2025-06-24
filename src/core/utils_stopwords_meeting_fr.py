"""
Stopwords spécialisés pour l'analyse de meetings en français.
Séparation claire des données linguistiques et du code métier.

Version: 1.0
Langue: Français
Contexte: Réunions professionnelles, meetings business
Problème : Améliorer la qualité de l'extraction en rajoutant des stopwords
"""

from typing import Set, List
import logging

logger = logging.getLogger(__name__)

# === MÉTADONNÉES ===
STOPWORDS_VERSION = "1.0"
STOPWORDS_LANGUAGE = "fr"
STOPWORDS_CONTEXT = "business_meetings"
LAST_UPDATED = "2025-06-20"

# === CATEGORIES DE STOPWORDS ===

# Contractions françaises courantes (Rajout suite essai du 20/06/25)
FRENCH_CONTRACTIONS = [
    "c'est", "qu'il", "j'ai", "qu'on", "qu'à", "c'était", "j'étais",
    "qu'est-ce", "est-ce", "n'est", "s'il", "d'ailleurs", "d'accord",
    "aujourd'hui", "jusqu'à", "peut-être", "c'était", "s'était"
]

# Stopwords de base français | NLTK équivalent pour fallback
FRENCH_BASE_STOPWORDS = [
    "le", "de", "et", "à", "un", "il", "être", "en", "avoir", "que", "pour",
    "dans", "ce", "son", "une", "sur", "avec", "ne", "se", "pas", "tout", "plus",
    "par", "grand", "donc", "alors", "bien", "très", "où", "du", "quand", "mais",
    "sans", "sous", "entre", "après", "avant", "pendant", "depuis", "vers", "chez",
    "selon", "malgré", "sauf", "outre", "celui", "celle", "ceux", "celles"
]

# Mots courts non-informatifs pour meetings
MEETING_BASIC_WORDS = [
    "bon", "oui", "non", "bien", "fin", "fois", "comme", "plus",
    "moins", "encore", "déjà", "jamais", "toujours", "souvent",
    "plutôt", "assez", "trop", "beaucoup", "peu", "très", "tout"
]

# Verbes fréquents meetings
MEETING_COMMON_VERBS = [
    "être", "avoir", "faire", "aller", "venir", "voir", "savoir",
    "pouvoir", "vouloir", "falloir", "devoir", "dire", "prendre",
    "mettre", "donner", "penser", "croire", "sembler", "paraître"
]

# Connecteurs et transitions meetings
MEETING_CONNECTORS = [
    "alors", "donc", "du coup", "ensuite", "après", "puis", "en fait",
    "enfin", "voilà", "bref", "par contre", "quand même", "de toute façon",
    "avant", "pendant", "depuis", "vers", "chez", "entre", "parmi",
    "selon", "malgré", "sauf", "outre", "néanmoins", "cependant",
    "toutefois", "pourtant", "sinon", "notamment", "particulièrement"
]

# Fillers et expressions orales meetings
MEETING_FILLERS = [
    "heu", "euh", "bah", "ben", "hein", "quoi", "genre", "ouais", "ouai",
    "nan", "ok", "okay", "parfait", "exact", "voilà", "bon voilà"
]

# Expressions argotiques modernes
MEETING_SLANG = [
    "putain", "de ouf", "fréro", "grave", "en vrai", "gros",
    "c'est relou", "c'est chiant", "c'est genre", "du style",
    "genre de truc", "truc de ouf", "c'est chaud", "c'est abusé"
]

# Expressions meetings business
MEETING_EXPRESSIONS = [
    "on va", "il faut", "je pense", "je crois", "on peut", "ça va",
    "c'est bon", "ok alors", "du coup on", "donc on", "alors on", "bon alors",
    "tu vois", "je sais pas", "je veux dire", "c'est clair", "tu sais",
    "en gros", "en mode", "ça veut dire", "tu m'entends", "si tu veux",
    "je dirais", "ça marche", "j'sais pas", "tu vois ce que je veux dire",
    "tu comprends", "j'veux dire", "tu me suis", "vois-tu"
]

# Politesse et formules sociales
MEETING_POLITENESS = [
    "merci", "s'il vous plaît", "s'il te plaît", "excusez-moi", "pardon",
    "désolé", "bonjour", "bonsoir", "au revoir", "à bientôt",
    "bonne journée", "sorry", "thanks", "ciao", "bye"
]

# Pronoms et déterminants
MEETING_PRONOUNS = [
    "celui", "celle", "ceux", "celles", "certains", "certaines",
    "plusieurs", "quelques", "chaque", "aucun", "aucune",
    "lequel", "laquelle", "lesquels", "lesquelles", "tous", "toutes"
]

# Expressions temporelles meetings
MEETING_TEMPORAL = [
    "hier", "demain", "maintenant", "bientôt", "récemment",
    "prochainement", "actuellement", "désormais", "autrefois",
    "jadis", "naguère", "dorénavant", "ultérieurement"
]

# Intensificateurs
MEETING_INTENSIFIERS = [
    "vraiment", "réellement", "absolument", "complètement",
    "totalement", "entièrement", "particulièrement", "spécialement",
    "extrêmement", "incroyablement", "énormément", "considérablement"
]

# === ASSEMBLAGE FINAL ===

def get_all_meeting_stopwords_fr() -> Set[str]:
    """
    Retourne l'ensemble complet des stopwords français pour meetings.

    Retourne:
        Set[str]: Stopwords optimisés pour l'analyse de meetings business en français
    """
    all_stopwords = set()

    # Assemblage de toutes les catégories
    categories = [
        FRENCH_BASE_STOPWORDS,     # ✅ Nouveau: base NLTK pour fallback
        FRENCH_CONTRACTIONS,
        MEETING_BASIC_WORDS,
        MEETING_COMMON_VERBS,
        MEETING_CONNECTORS,
        MEETING_FILLERS,
        MEETING_SLANG,
        MEETING_EXPRESSIONS,
        MEETING_POLITENESS,
        MEETING_PRONOUNS,
        MEETING_TEMPORAL,
        MEETING_INTENSIFIERS
    ]

    for category in categories:
        all_stopwords.update(word.lower() for word in category)

    logger.info(f"📋 Stopwords meetings assemblés: {len(all_stopwords)} mots")
    return all_stopwords

def get_fallback_stopwords_fr() -> Set[str]:
    """
    Retourne les stopwords de fallback si NLTK n'est pas disponible.
    Version light mais complète pour garantir la qualité des topics.

    Retourne:
        Set[str]: Stopwords de fallback optimisés
    """
    fallback_categories = [
        FRENCH_BASE_STOPWORDS,     # Base NLTK équivalent
        FRENCH_CONTRACTIONS,       # Fix tes problèmes de topics
        MEETING_BASIC_WORDS,       # Mots courts meetings
        MEETING_CONNECTORS,        # Connecteurs essentiels
        MEETING_FILLERS            # Fillers critiques
    ]

    fallback_stopwords = set()
    for category in fallback_categories:
        fallback_stopwords.update(word.lower() for word in category)

    logger.info(f"📋 Stopwords fallback: {len(fallback_stopwords)} mots (version light)")
    return fallback_stopwords

def get_category_stopwords(category: str) -> Set[str]:
    """
    Retourne les stopwords d'une catégorie spécifique.

    Args:
        category: Nom de la catégorie
                 ('contractions', 'basic', 'verbs', 'connectors', 'fillers',
                  'slang', 'expressions', 'politeness', 'pronouns', 'temporal', 'intensifiers')

    Retourne:
        Set[str]: Stopwords de la catégorie demandée
    """
    category_map = {
        'contractions': FRENCH_CONTRACTIONS,
        'base': FRENCH_BASE_STOPWORDS,        # ✅ Nouveau
        'basic': MEETING_BASIC_WORDS,
        'verbs': MEETING_COMMON_VERBS,
        'connectors': MEETING_CONNECTORS,
        'fillers': MEETING_FILLERS,
        'slang': MEETING_SLANG,
        'expressions': MEETING_EXPRESSIONS,
        'politeness': MEETING_POLITENESS,
        'pronouns': MEETING_PRONOUNS,
        'temporal': MEETING_TEMPORAL,
        'intensifiers': MEETING_INTENSIFIERS
    }

    if category not in category_map:
        available = list(category_map.keys())
        raise ValueError(f"Catégorie '{category}' inconnue. Disponibles: {available}")

    return set(word.lower() for word in category_map[category])

def analyze_stopwords_coverage(problematic_words: List[str]) -> dict:
    """
    Analyse la couverture des stopwords sur une liste de mots problématiques

    Args:
        problematic_words: Liste des mots qui passent encore dans les topics

    Retourne:
        dict: Statistiques de couverture
    """
    all_stopwords = get_all_meeting_stopwords_fr()

    covered = []
    missing = []

    for word in problematic_words:
        if word.lower() in all_stopwords:
            covered.append(word)
        else:
            missing.append(word)

    coverage_rate = len(covered) / len(problematic_words) if problematic_words else 0

    return {
        'total_words': len(problematic_words),
        'covered': covered,
        'missing': missing,
        'coverage_rate': coverage_rate,
        'stopwords_count': len(all_stopwords)
    }

def get_stopwords_metadata() -> dict:
    """Retourne les métadonnées des stopwords."""
    return {
        'version': STOPWORDS_VERSION,
        'language': STOPWORDS_LANGUAGE,
        'context': STOPWORDS_CONTEXT,
        'last_updated': LAST_UPDATED,
        'total_stopwords': len(get_all_meeting_stopwords_fr()),
        'categories': [
            'contractions', 'basic', 'verbs', 'connectors', 'fillers',
            'slang', 'expressions', 'politeness', 'pronouns', 'temporal', 'intensifiers'
        ]
    }

# === TEST ET VALIDATION ===

def test_stopwords_quality():
    """Test de qualité des stopwords avec tes mots problématiques."""

    # Tes mots problématiques des résultats
    problematic_topics = [
        "c'est", "qu'il", "j'ai", "qu'on", "bon", "oui", "fin",
        "fois", "bien", "donc", "comme", "plus", "encore"
    ]

    print("🔍 TEST QUALITÉ STOPWORDS MEETINGS")
    print("=" * 50)

    coverage = analyze_stopwords_coverage(problematic_topics)

    print(f"📊 Mots testés: {coverage['total_words']}")
    print(f"✅ Couverts: {len(coverage['covered'])}")
    print(f"❌ Manquants: {len(coverage['missing'])}")
    print(f"🎯 Taux couverture: {coverage['coverage_rate']*100:.1f}%")

    if coverage['missing']:
        print(f"\n❌ Mots encore problématiques: {coverage['missing']}")

    print(f"\n📋 Total stopwords: {coverage['stopwords_count']}")

    return coverage['coverage_rate'] >= 0.9  # 90% minimum

if __name__ == "__main__":
    # Test automatique
    success = test_stopwords_quality()
    print(f"\n🎯 Test réussi: {'✅ OUI' if success else '❌ NON'}")

    # Métadonnées
    metadata = get_stopwords_metadata()
    print(f"\n📋 Métadonnées: {metadata}")
