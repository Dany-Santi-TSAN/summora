# 🎯 Summora V3
**Speech In, Sense Out**

**Analyse intelligente de meeting et reporting automatisé**

*Summora transforme chaque réunion en un livrable clair et actionnable grâce à un pipeline LLM modulaire et robuste, optimisé pour le traitement audio et l’analyse métier.*

## 🎯 Projet Personnel - Démonstration technique

**Objectif** : Développer mes compétences en LLM Engineering et NLP appliqué à l'analyse de réunions, en abordant le projet comme un vrai produit avec une architecture scalable.

**Contexte** : Solution pour automatiser la synthèse de réunions (transcription, ground_truth automatisé, résumé, recommandations) en réponse au problème concret de managers perdant ~2h/semaine en rédaction de comptes rendus.

⚠️ **Note importante** : Projet développé avec des données publiques uniquement. Pas de déploiement en production pour des raisons de conformité RGPD (protection des données de réunions d'entreprise).


## 🚀 Architecture technique

| Étape Audio                          | NLP        | Cascade LLM                     | Insights                                                   |
| ------------------------------------ | ---------- | ------------------------------- | ---------------------------------------------------------- |
| 🎙️ Audio → Whisper + Audio Analysis | YAKE + LLM | [Premium] → [Gratuit] → [Local] | Transcription + Détection NLP + Qualité / Énergie / Rythme |


Note : la cascade garantit qu’une réponse est toujours fournie, même si un LLM tombe.

## 🔧 Personnalisation métier

**Prompts centralisés** : src/config/llm_config.py

**Vocabulaire métier** : stopwords et termes spécifiques → src/core

**RAG enrichi** : intégration de documents internes pour refléter la culture d’entreprise → src/rag/rag_documents

## 📂 Préparation des données

Créer un dossier data à la racine pour y déposer vos fichiers audio :

mkdir data

## 🔑 Configuration

## ⚙️ Setup rapide (démonstration locale)

### Création environnement virtuel
python -m venv venv

source venv/bin/activate   # Linux/Mac

venv\Scripts\activate      # Windows

### Installation dépendances
pip install -r requirements.txt

### Variables d'environnement (API keys)

touch .env

echo "OPENROUTER_API_KEY=xxx" >> .env

echo "HF_TOKEN=xxx" >> .env

cat .env

## ▶️ Usage

**Exécution complète**

python scripts/main.py --model medium --full data/audio.mp3

**Modules spécifiques**

python scripts/main_extract.py      # Extraction thèmes & actions

python scripts/main_reco.py         # Recommandations & insights

python scripts/main_corrector.py    # Correcteur LLM


## Déploiement app
### Terminal 1
python app/backend.py

### Terminal 2
streamlit run app/ui_streamlit.py

## Déploiement EC2 (AWS)

1) Créer un dossier **keys/** à la racine et y placer votre **key pair AWS.**

2) Lancer une instance (ex. g4dn.xlarge)

3) Suivre le guide complet disponible dans deploy/

📌 Ce projet est conçu comme démonstration portfolio : l’installation sert uniquement à tester le pipeline sur des données publiques.

## 📊 Outputs

Arborescence des résultats :

output/

├── transcriptions/     # Fichiers .txt + JSON métadonnées

├── extractions/        # Topics, actions, décisions (JSON)

├── recommendations/    # Conseils d'amélioration (JSON)

├── audio_analysis/     # Qualité audio, spectrogrammes

└── reports/            # Métriques complètes, monitoring


### Exemple d’extraction :

| Catégorie             | Détails                                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Topics principaux** | budget marketing 2025<br>lancement produit Q2<br>recrutement équipe                                                    |
| **Points à retenir**  | Décision : augmenter budget marketing de 20%<br>Action Jean : plan détaillé avant vendredi<br>Validation KPIs par Paul |
| **Actions clés**      | Plan marketing détaillé<br>Validation KPIs                                                                             |
| **Décisions prises**  | Budget +20%<br>Lancement Q2 confirmé                                                                                   |
| **Next steps**        | Réunion suivi dans 2 semaines                                                                                          |



## 🤝 Contribution & Déploiement

Ce projet est un portfolio technique conçu pour démontrer :

La mise en place d’une pipeline LLM robuste (audio → NLP → insights business).

**Les prompts et context engineering** ainsi que les **évaluations multi-critères** (LLM-as-a-judge, CER/WER/BERTScore).

La **scalabilité cloud** avec un guide de déploiement sur AWS EC2 GPU (dossier deploy/).

## 👤 Auteur

**Développé par Dany Tsan**
Projet personnel – Démonstration technique
⚠️ Non destiné à un usage commercial sans audit RGPD complet

*Merci pour la contribution des 3 managers pour leur précieux feedback*
