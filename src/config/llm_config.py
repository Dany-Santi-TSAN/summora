"""
Configuration centralisée LLM minimaliste pour Summora V3
Modèles + prompts + timeouts en un seul endroit
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()

@dataclass
class LLMConfig:
    """Configuration du LLM."""
    # API
    openrouter_api_key: str = os.getenv('OPENROUTER_API_KEY', '')
    base_url: str = "https://openrouter.ai/api/v1"

    # Timeouts
    default_timeout: int = 60
    max_tokens: int = 50000
    max_tokens_reco: int = 4000
    judge_max_tokens: int = 1000
    temperature: float = 0.1
    temperature_reco: float = 0.2 # légère créativité pour recommandations variées

# Modèles par tâche
MODELS = {
    # Extraction
    'extraction_enhanced' : "qwen/qwen3-next-80b-a3b-instruct"
    ,'extraction_fallback': "mistralai/mistral-small-3.2-24b-instruct:free"

    # Judges
    ,'judge_primary': "mistralai/ministral-8b"
    ,'judge_fallback': "deepseek/deepseek-r1-0528-qwen3-8b:free"

    # Recommandations
    ,'recommendation': "qwen/qwen3-next-80b-a3b-instruct"
    ,'recommendation_fallback': "mistralai/mistral-small-3.2-24b-instruct:free"

    # Correction
    ,'correction': "qwen/qwen3-next-80b-a3b-instruct"
    ,'correction_fallback': "mistralai/mistral-small-3.2-24b-instruct:free"

}

# Prompts centralisés
PROMPTS = {
    # Extraction hybride (JSON + Liste fallback)
    'extraction_hybrid': lambda transcription, yake_context=None: f"""Tu es expert en analyse de réunions d'entreprise.

{f"CONTEXTE DETECTÉ : Topics business : {', '.join(yake_context.get('business_topics', [])[:5])}" if yake_context else ""}

**INSTRUCTIONS STRICTES :**
- Ne génère **AUCUN** texte en dehors du format demandé (pas de balises Markdown, pas de commentaires, pas de "Voici le JSON", etc.).
- Réponds **UNIQUEMENT** avec un JSON valide ou une liste, sans introduction ni conclusion.
- Si tu ne peux pas générer un JSON, utilise **exclusivement** le format liste.

**FORMATS AUTORISÉS :**
1. **OPTION 1 (PRIORITAIRE) - JSON uniquement** (sans balises ````jsonION 1 (PRÉFÉRÉE) - JSON uniquement :
{{
    "topics_principaux": ["topic1", "topic2", "topic3", "topic4", "topic5"],
    "points_a_retenir": ["point1", "point2", "point3", "point4", "point5", "point6", "point7", "point8", "point9", "point10"],
    "resume_abstractif": "Synthèse en 1 paragraphe des enseignements et décisions clés"
}}

**OPTION 2 (SI JSON IMPOSSIBLE) - Format liste** :
TOPICS PRINCIPAUX:
1. [topic business 1]
2. [topic business 2]
3. [topic business 3]
4. [topic business 4]
5. [topic business 5]

POINTS CLÉS:
- [point actionnable 1]
- [point actionnable 2]
- [point actionnable 3]
- [point actionnable 4]
- [point actionnable 5]
- [point actionnable 6]
- [point actionnable 7]
- [point actionnable 8]
- [point actionnable 9]
- [point actionnable 10]

RÉSUMÉ:
[Synthèse en 2-3 phrases de cette réunion]

Transcription :
{transcription[:3000]}

Choisis OPTION 1 (JSON) si possible, sinon OPTION 2 (liste) :""",

    # Judge extraction
    'judge_extraction': lambda transcription, extraction_data: f"""Tu es expert en évaluation d'extraction de réunions.

**INSTRUCTIONS STRICTES :**
- Réponds **UNIQUEMENT** avec un JSON valide, sans balises Markdown, sans commentaires, sans introduction.
- Ne génère **AUCUN** texte en dehors du JSON.

**CRITÈRES D'ÉVALUATION (score 0-100) :**
- **Pertinence** : Les topics reflètent-ils bien le contenu ?
- **Complétude** : Les points importants sont-ils couverts ?
- **Précision** : Les informations sont-elles exactes ?

TRANSCRIPTION ORIGINALE (extrait) :
{transcription[:2000]}...

EXTRACTION À ÉVALUER :
{str(extraction_data)[:1000]}...

Réponds en JSON uniquement :
{{
    "pertinence": 85,
    "completude": 90,
    "precision": 88,
    "score_global": 88,
    "justification": "Bonne extraction qui couvre les points essentiels..."
}}""",

    # Recommandations
'recommendation': lambda transcription, extraction_data=None, enhanced_context="", specialized_context="", meeting_type="Général": f"""Consultant expert optimisation meetings. {enhanced_context}

**INSTRUCTIONS STRICTES :**
- Réponds **UNIQUEMENT** avec un JSON valide, sans balises Markdown, sans commentaires, sans introduction.
- Ne génère **AUCUN** texte en dehors du JSON (pas de balises ```json, pas de <think>, pas de "Voici le JSON").
- **Commence ta réponse immédiatement par le JSON**, sans espace ni saut de ligne avant.

**CONTEXTE SPÉCIALISÉ:** {specialized_context}

**FEW-SHOTS EXEMPLES (ne pas copier) :**
Réunion dispersée → Structure: "Agenda 24h avant + Timekeeper", Animation: "Parking lot idées off-topic"
Meeting technique → Communication: "Schémas visuels + Glossaire", Participation: "Validation compréhension"
Réunion conflictuelle → Animation: "Règles débat + Médiateur neutre", Efficacité: "Points convergence d'abord"

**CATÉGORIES:** Structure, Animation, Participation, Efficacité, Communication

**TRANSCRIPTION:**
{transcription[:1500]}

**FORMAT DE RÉPONSE OBLIGATOIRE :**
{{
    "recommandations_principales": [
        {{
            "categorie": "[Structure|Animation|Participation|Efficacité|Communication]",
            "titre": "[action concrète en 10 mots max]",
            "description": "[détails précis + exemples/outils concrets]",
            "impact": "[very_high|high|medium|low]",
            "facilite_implementation": "[easy|medium|hard]"
        }}
    ],
    "resume_conseil": "[2-3 priorités adaptées au contexte {meeting_type}]",
    "score_amelioration_potentiel": [0-100]
}}

**RÈGLES ABSOLUES :**
- Génère 3-7 recommandations concrètes et actionnables
- Adapte au contexte détecté, ne copie pas les exemples
- Score basé sur: clarté objectifs + structure + participation + décisions
- Commence **immédiatement** par le JSON

- **INTERDIT ABSOLU** : Ne génère JAMAIS de balises ```json ou ``` ou <JSON> ou toute autre balise."""
,

    # Judge recommandations
    'judge_recommendation': lambda transcription, recommendations_data: f"""Tu es expert en évaluation de qualité de recommandations business.

Évalue ces recommandations d'amélioration meeting selon ces critères (score 0-100) :
- **Pertinence** : Les recommandations sont-elles adaptées aux problèmes détectés ?
- **Actionnabilité** : Peut-on facilement les mettre en œuvre ?
- **Impact** : Vont-elles vraiment améliorer les futures réunions ?
- **Spécificité** : Sont-elles concrètes et précises ?

TRANSCRIPTION ORIGINALE (extrait) :
{transcription[:2000]}...

RECOMMANDATIONS À ÉVALUER :
{str(recommendations_data)[:1000]}...

Réponds en JSON uniquement :
{{
    "pertinence": 85,
    "actionnabilite": 90,
    "impact_potentiel": 80,
    "specificite": 88,
    "score_global": 86,
    "justification": "Bonnes recommandations concrètes et applicables...",
    "qualite_conseil": "high"
}}""",

### ==== Correction ====
    'correction': lambda chunk: f"""Tu es un correcteur automatique ultra-précis.

**INSTRUCTIONS STRICTES :**
- Réponds uniquement avec le texte corrigé, brut.
- Pas de JSON, pas de Markdown, pas de "think", pas de commentaires.
- Commence immédiatement par le texte corrigé.

**RÈGLES DE CORRECTION :**
1. Corrige :
   - Les fautes d'orthographe, grammaire, syntaxe et ponctuation.
   - Les noms de villes et marques mal orthographiés (ex: "lyons" → "Lyon").
2. Préserve :
   - Les noms propres (prénoms, noms de famille) tels quels, même s'ils semblent
     incorrects.
   - Les acronymes et sigles (ex: "SNCF", "DAF", "KPI").
3. Structure :
   - 80 caractères maximum par ligne (utilise des retours à la ligne).
   - Pas d’énumérations ou de formats supplémentaires, uniquement du texte.

**EXEMPLE :**
Texte original : "marie a acheter une voiture à lyons. La marque est Toyotta."
Texte corrigé  :
"Marie a acheté une voiture à Lyon.
La marque est Toyota."

**TEXTE À CORRIGER :**
{chunk}""",

### ==== Judge correction ===
    'judge_correction': lambda original, corrected: f"""Tu es un expert en évaluation de correction de transcriptions multilingues.
**INSTRUCTIONS STRICTES :**
- Réponds **UNIQUEMENT** avec un JSON valide, sans balises Markdown, sans commentaires, sans introduction.
- **Personnalise les scores et justifications** en fonction des transcriptions fournies.
- Ne répète **JAMAIS** les mêmes scores ou justifications pour des corrections différentes.

**EXEMPLES DE RÉPONSES (few-shot) :**
1. **Exemple 1 (Correction excellente) :**
   {{
       "fidelite_contenu": 95,
       "correction_grammaire": 98,
       "amelioration_ponctuation": 96,
       "preservation_sens": 99,
       "score_global": 97,
       "justification": "La transcription corrigée est fluide, sans erreur grammaticale et préserve parfaitement le sens original. La ponctuation améliore la lisibilité sans altérer les termes techniques.",
       "note_sur_10": 9.7,
       "qualite_ground_truth": "very_high"
   }}

2. **Exemple 2 (Correction moyenne) :**
   {{
       "fidelite_contenu": 75,
       "correction_grammaire": 80,
       "amelioration_ponctuation": 70,
       "preservation_sens": 85,
       "score_global": 78,
       "justification": "La correction améliore la grammaire mais introduit des reformulations inutiles. La ponctuation est parfois incorrecte, et certains termes techniques sont mal préservés.",
       "note_sur_10": 7.8,
       "qualite_ground_truth": "medium"
   }}

3. **Exemple 3 (Correction insuffisante) :**
   {{
       "fidelite_contenu": 50,
       "correction_grammaire": 60,
       "amelioration_ponctuation": 45,
       "preservation_sens": 55,
       "score_global": 53,
       "justification": "La correction contient des erreurs grammaticales et des omissions. Le sens original est partiellement altéré, et la ponctuation est chaotique.",
       "note_sur_10": 5.3,
       "qualite_ground_truth": "low"
   }}

**CRITÈRES D'ÉVALUATION (score 0-100) :**
- **Fidélité** : Préservation du contenu original sans reformulation excessive.
- **Grammaire** : Correction des erreurs orthographiques et grammaticales.
- **Ponctuation** : Amélioration de la structure et de la lisibilité.
- **Préservation** : Maintien du sens et des termes techniques.

**DONNÉES À ÉVALUER :**
- TRANSCRIPTION ORIGINALE (Whisper) :
{original[:2000]}
- TRANSCRIPTION CORRIGÉE (LLM) :
{corrected[:2000]}

**FORMAT DE RÉPONSE OBLIGATOIRE :**
{{
    "fidelite_contenu": [score],
    "correction_grammaire": [score],
    "amelioration_ponctuation": [score],
    "preservation_sens": [score],
    "score_global": [score],
    "justification": "[Justification DÉTAILLÉE et PERSONNALISÉE en fonction des transcriptions]",
    "note_sur_10": [note],
    "qualite_ground_truth": "[very_high|high|medium|low]"
}}

**RÈGLES ABSOLUES :**
- Commence ta réponse **immédiatement** par le JSON, sans espace ni saut de ligne avant.
- **Adapte les scores et justifications** en fonction des transcriptions réelles. Ne copie jamais les exemples.
"""
}

# Instance globale
config = LLMConfig()
