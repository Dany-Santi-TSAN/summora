# 🎯 Summora V3
**Speech In, Sense Out**

## 🎯 Projet Personnel - Démonstration de Compétences

**Objectif** : Développer mes compétences en LLM Engineering et NLP appliqué à l'analyse de réunions, en abordant le projet comme un vrai produit avec une architecture scalable.

**Contexte** : Solution pour automatiser la synthèse de réunions (transcription, résumé, recommandations) en réponse au problème concret de managers perdant ~2h/semaine en rédaction de comptes rendus.

⚠️ Note importante : Projet développé avec des données publiques uniquement. Pas de déploiement en production pour des raisons de conformité RGPD (protection des données de réunions d'entreprise).

## 🏗️ Architecture Pipeline

### Vue d'ensemble
```
📥 Audio Input → 🎤 Whisper → 🧠 Extraction LLM → 💡 Recommandations → 📊 Rapport
```

### Pipeline détaillé

#### 1. **Transcription Audio** (Whisper)
- **Modèles supportés** : base, small, medium, large
- **Optimisations** : Prompts spécialisés meetings, cleanup GPU automatique
- **Fallbacks** : Dégradation automatique si mémoire insuffisante
- **Output** : JSON avec métadonnées, scores de confiance, métriques temporelles

#### 2. **Extraction d'Insights** (Cascade LLM + Fallbacks)
Pipeline en cascade avec 4 niveaux de robustesse :

```python
1. 🚀 Qwen Enhanced (YAKE + LLM Premium)     → Qualité maximale
2. 🆓 LLM Gratuit (Llama 3.2-3b)            → Backup fiable
3. 🔧 Phi3 Mini Local (SLM + YAKE)          → Mode RGPD (lent)
4. 📊 YAKE Fallback                         → Toujours fonctionne
```

**Extraction YAKE+LLM :**
- Pré-analyse YAKE pour détecter contexte business
- Enrichissement du prompt LLM avec vocabulaire spécialisé
- Guidage intelligent vers actions/décisions prioritaires

#### 3. **Génération de Recommandations**
- **LLM Judge** : Évaluation qualité des extractions
- **Recommandations contextuelles** : Adaptées au grade détecté
- **Cascade intelligente** : Premium → Gratuit → Dictionnaire pré-codé

#### 4. **Monitoring & Métriques**
- **Coûts API** : Calcul précis via tokenizers (€/1M tokens)
- **Performance** : RAM, GPU, temps d'exécution par étape
- **Qualité** : BERTScore, confidence Whisper, densité business
- **Rapports** : JSON détaillés pour analyse post-traitement

## 🚀 Installation

### Prérequis
- Python 3.10+
- 16GB RAM recommandés (pour Whisper medium+)
- GPU optionnel (accélération Whisper/Phi3)

### Setup rapide
```bash
# Clone et setup
git clone https://github.com/username/summora.git
cd summora

# Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Configuration
python main.py --create-config
# Éditer .env avec vos clés API
```

### Configuration API (optionnelle)
```bash
# .env
OPENROUTER_API_KEY=your-key-here
# Fallback automatique vers méthodes locales si non configuré
```

## 💻 Usage

### Pipeline complet
```bash
# Pipeline CŒUR (transcription + extraction)
python main.py meeting.mp3 --light

# Pipeline COMPLET (+ recommandations + analyse audio)
python main.py meeting.mp3 --all

# Pipeline avec recommandations personnalisées
python main.py meeting.mp3 --light --with-reco
```

### Modes spécialisés
```bash
# Transcription seule
python main.py meeting.mp3 --transcribe-only --model medium

# Extraction depuis transcription existante
python main.py transcription.txt --extract-only

# Recommandations depuis extraction existante
python main.py extraction.json --reco-only
```

### Usage programmatique
```python
from main import transcribe_audio_file, analyze_meeting_full_pipeline

# Transcription simple
result = transcribe_audio_file("meeting.mp3", model="base")

# Pipeline complet
insights = analyze_meeting_full_pipeline("meeting.mp3", include_all_features=True)
```

## 📊 Outputs

### Structure des résultats
```
output/
├── transcriptions/     # Fichiers .txt + JSON métadonnées
├── extractions/       # Topics, actions, décisions (JSON)
├── recommendations/   # Conseils d'amélioration (JSON)
├── audio_analysis/   # Qualité audio, spectrogrammes
└── reports/          # Métriques complètes, monitoring
```

### Exemple d'extraction
```json
{
  "topics_principaux": [
    "budget marketing 2025",
    "lancement produit Q2",
    "recrutement équipe"
  ],
  "points_a_retenir": [
    "Décision: augmenter budget marketing de 20%",
    "Action Jean: plan détaillé avant vendredi",
    "Validation KPIs par Paul"
  ],
  "insights_business": {
    "actions_cles": ["Plan marketing détaillé", "Validation KPIs"],
    "decisions_prises": ["Budget +20%", "Lancement Q2 confirmé"],
    "next_steps": ["Réunion suivi dans 2 semaines"]
  }
}
```

## 🧪 Tests et Qualité

### Tests automatisés
```bash
# Tests modules LLM
pytest tests/test_llm/ -v

# Tests métriques
pytest tests/test_metrics/ -v

# Tests complets
pytest tests/ -v
```

### Monitoring intégré
- **Cascade tracking** : Succès/échecs par niveau
- **Cost monitoring** : Coûts API en temps réel
- **Quality metrics** : Scores de confiance, densité business
- **Performance** : RAM/GPU/tokens par seconde

## ⚙️ Configuration Avancée

### Modèles Whisper
```yaml
# config.yaml
model: "base"          # tiny|base|small|medium|large
language: "fr"
temperature: 0.0
min_confidence: 0.7
```

### Pipeline LLM
```yaml
# Fonctionnalités bonus
enable_recommendation: true    # Recommandations LLM
enable_correction: false      # Correction transcription
enable_visual: true          # Analyse spectrogrammes
enable_spot_check: false     # QA aléatoire
```

## 🔒 Confidentialité & RGPD

**⚠️ Limitations connues :**
- APIs externes (OpenRouter) = données hors UE
- Mode local disponible mais performances réduites
- Recommandé uniquement pour données publiques/tests

### Données d'entraînement
- **Sources** : Podcasts publics, conférences YouTube, réunions simulées
- **Aucune donnée privée** utilisée pour développement

## 🛠️ Architecture Technique

### Stack
- **Transcription** : OpenAI Whisper
- **LLM** : Qwen, Llama (via OpenRouter)
- **Extraction locale** : YAKE, Phi3 Mini
- **Évaluation** : LLM Judges, BERTScore et métriques adaptés
- **Monitoring** : tracemalloc, CUDA metrics

### Patterns de conception
- **Cascade resilience** : Fallbacks automatiques à chaque étape
- **Configuration centralisée** : YAML + dataclasses
- **Monitoring pervasif** : Métriques à tous les niveaux
- **API abstraite** : Interface programmatique simple

### Performance
- **Whisper base** : ~30s pour 10min audio (CPU)
- **Crash test audio 1h30 (medium)** : <5min pour 1h30h d'audio (GPU 16go VRAM)
- **Extraction LLM** : ~5s via API, ~2h en local
- **Memory usage** : ~2GB baseline, 8GB+ avec modèles large

## 📈 Roadmap

### Version actuelle
- ✅ Pipeline cascade robuste
- ✅ Monitoring complet
- ✅ Interface CLI + programmatique

### Idées d'évolution
- 🔄 Interface Streamlit
- 🔄 Rajout documents dans le RAG pour recommendations enrichies
- 🔄 Implémenter un graphe de connaissance avec document interne
- 🔄 Support modèles européens (Mistral AI)
- 🔄 Anonymisation des transcriptions et suppression adresse mail
- 🔄 Déploiement Docker + FastAPI


## 🤝 Contribution

Projet personnel à des fins éducatives. Contributions/feedback bienvenus via Issues.

---

**Développé par Dany Tsan** | Projet personnel - Montée en compétences LLM/NLP
*Non destiné à un usage commercial sans audit RGPD complet*
