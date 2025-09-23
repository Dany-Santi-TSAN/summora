#!/usr/bin/env python3
"""
Summora V3 - Interface Streamlit refactorisée
Design Apple-like avec parsing JSON robuste
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional, Any
import time

# Configuration page
st.set_page_config(
    page_title="Summora",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS minimaliste
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-blue: #007AFF;
        --bg-primary: #FFFFFF;
        --bg-secondary: #F9F9F9;
        --text-primary: #1D1D1F;
        --text-secondary: #86868B;
        --border-light: #E5E5E7;
        --shadow-light: 0 4px 16px rgba(0,0,0,0.06);
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main, .sidebar .sidebar-content, .stMarkdown, .stText {
        font-family: var(--font-family) !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .main-title {
        font-family: var(--font-family) !important;
        font-size: 3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        text-align: center;
    }

    .main-subtitle {
        font-family: var(--font-family) !important;
        font-size: 1.2rem;
        color: var(--text-secondary);
        font-weight: 400;
        text-align: center;
        margin-bottom: 2rem;
    }

    .card-title {
        font-family: var(--font-family) !important;
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .upload-container {
        background: var(--bg-secondary);
        border: 2px dashed var(--border-light);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }

    .upload-container:hover {
        border-color: var(--primary-blue);
        background: rgba(0,122,255,0.02);
    }

    .result-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: var(--shadow-light);
        margin: 1rem 0;
        border: 1px solid var(--border-light);
    }

    .stButton > button {
        background: var(--primary-blue);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-light);
        font-family: var(--font-family) !important;
    }

    .stButton > button:hover {
        background: #0056CC;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,122,255,0.3);
    }

    .metric-card {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }

    .bullet-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border-light);
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }

    .bullet-icon {
        color: var(--primary-blue);
        font-weight: bold;
        margin-top: 0.1rem;
    }

    .reco-item {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-blue);
    }

    .reco-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }

    .reco-description {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }

    .empty-state {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        background: var(--bg-secondary);
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration API
API_BASE = "http://localhost:8080"

class SummoraUI:
    """Interface Summora refactorisée avec parsing JSON robuste"""

    def __init__(self):
        self.api_base = API_BASE

    def check_api_health(self) -> bool:
        """Vérifie la santé de l'API"""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def call_analyze_api(self, file_content: bytes, filename: str, mode: str, model: str) -> Optional[Dict]:
        """Appel API unifié vers /analyze"""
        try:
            files = {"file": (filename, file_content, "audio/mpeg")}
            data = {"mode": mode, "model": model}
            response = requests.post(
                f"{self.api_base}/analyze",
                files=files,
                data=data,
                timeout=600
            )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Erreur API: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            st.error("Timeout: L'analyse prend plus de 10 minutes")
            return None
        except Exception as e:
            st.error(f"Erreur connexion API: {str(e)}")
            return None

    def safe_get_nested(self, data: Dict, *keys) -> Any:
        """Navigation sécurisée dans les structures imbriquées"""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def extract_extraction_data(self, result_data: Dict) -> Optional[Dict]:
        """Extrait les données d'extraction de la structure JSON"""
        # Structure: results.extraction.extraction
        extraction_container = self.safe_get_nested(result_data, "results", "extraction")
        if extraction_container:
            return self.safe_get_nested(extraction_container, "extraction")
        return None


    def extract_audio_data(self, result_data: Dict) -> Optional[Dict]:
        """Extrait les données audio de la structure JSON"""
        # Structure: results.audio_analysis.analysis
        audio_container = self.safe_get_nested(result_data, "results", "audio_analysis")
        if audio_container:
            return self.safe_get_nested(audio_container, "analysis")
        return None

    def extract_recommendations_data(self, result_data: Dict) -> List[Dict]:
        """Extrait les recommandations - Triple imbrication"""
        reco_container = self.safe_get_nested(result_data, "results", "recommendations")
        if reco_container:
            # Première couche recommendations
            reco_layer1 = self.safe_get_nested(reco_container, "recommendations")
            if reco_layer1:
                # Deuxième couche recommendations
                reco_layer2 = self.safe_get_nested(reco_layer1, "recommendations")
                if reco_layer2:
                    # Troisième couche recommandations_principales + resume_conseil
                    return {
                        'recommandations':reco_layer2.get("recommandations_principales", [])
                        ,'resume_conseil':reco_layer2.get("resume_conseil","")
                    }

        return {'recommandations':[], 'resume_conseil': ""}

    def render_header(self):
        """Affiche le header principal"""
        st.markdown("""
        <div class="main-title">🎤 Summora</div>
        <div class="main-subtitle">Speech In, Sense Out | Analyse intelligente de réunion</div>
        """, unsafe_allow_html=True)

    def render_sidebar(self) -> tuple:
        """Affiche la sidebar et retourne (mode, model)"""
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")

            mode = st.selectbox(
                "Mode d'analyse",
                ["light", "optimal", "full"],
                index=1,  # optimal par défaut
                help="""
                • **Light**: Transcription + résumé simple
                • **Optimal**: + Analyse audio + métriques
                • **Full**: + Correction + recommandations
                """
            )

            model = st.selectbox(
                "Modèle Whisper",
                ["base", "small", "medium", "large"],
                index=0,  # base par défaut
                help="Medium recommandé pour rendu optimal"
            )

            st.markdown("---")

            # Status API
            if self.check_api_health():
                st.success("🟢 API connectée")
            else:
                st.error("🔴 API non disponible")

        return mode, model

    def render_upload_zone(self):
        """Affiche la zone d'upload"""
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Glissez votre fichier audio ici",
            type=['mp3', 'wav', 'm4a', 'webm'],
            help="Formats supportés: MP3, WAV, M4A, WebM"
        )

        st.markdown("</div>", unsafe_allow_html=True)
        return uploaded_file

    def render_summary_section(self, result_data: Dict):
        """Affiche la section résumé"""
        st.markdown("""
        <div class="result-card">
            <div class="card-title">📝 Résumé de la réunion</div>
        """, unsafe_allow_html=True)

        extraction_data = self.extract_extraction_data(result_data)

        # DEBUG spécifique extraction
        #st.write("DEBUG extraction_data type:", type(extraction_data))
        #st.write("DEBUG extraction_data value:", extraction_data)

        if extraction_data:
            # Correction : double imbrication extraction.extraction
            real_extraction = extraction_data.get("extraction", {})

            # Topics principaux - en markdown
            topics = real_extraction.get("topics_principaux", [])
            if topics:
                st.markdown("**🎯 Voici les sujets principaux abordés :**")
                for i, topic in enumerate(topics[:5], 1):
                    st.markdown(f"**{i}.** {topic}")
                st.markdown("")

            # Points à retenir - Style en markdown
            points = real_extraction.get("points_a_retenir", [])
            if points:
                st.markdown("**💡 Les points clés à retenir :**")
                for point in points[:10]:
                    st.markdown(f"• {point}")
                st.markdown("")

            # Résumé abstractif - Mise en valeur
            resume = real_extraction.get("resume_abstractif", "")
            if resume:
                st.markdown("**📋 En résumé :**")
                st.markdown(f"*{resume}*")
                st.markdown("")
        else:
            st.markdown("""
            <div class="empty-state">
                <p>Aucune donnée d'extraction disponible</p>
                <small>Vérifiez que le pipeline d'extraction s'est exécuté correctement</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    def render_audio_section(self, result_data: Dict):
        """Affiche la section analyse audio"""
        st.markdown("""
        <div class="result-card">
            <div class="card-title">📊 Analyse Audio</div>
        """, unsafe_allow_html=True)

        audio_data = self.extract_audio_data(result_data)

        if audio_data:
            # Métriques en colonnes
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                duration = audio_data.get("duration_formatted", "N/A")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">⏱️ {duration}</div>
                    <div class="metric-label">Durée</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                speech_ratio = audio_data.get("speech_ratio", 0) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">🗣️ {speech_ratio:.0f}%</div>
                    <div class="metric-label">Parole</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                quality_score = audio_data.get("meeting_quality_score", 0)
                quality_grade = audio_data.get("meeting_quality_grade", "N/A")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">🎯 {quality_score}/100</div>
                    <div class="metric-label">Qualité ({quality_grade})</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                clarity = audio_data.get("vocal_clarity_score", 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">🔊 {clarity:.2f}</div>
                    <div class="metric-label">Clarté</div>
                </div>
                """, unsafe_allow_html=True)

            # Graphique spider chart
            if quality_score > 0:
                categories = ['Qualité Audio', 'Ratio Parole', 'Clarté Vocale', 'Dynamique']
                values = [
                    quality_score / 100,
                    speech_ratio / 100,
                    clarity,
                    0.8  # Placeholder dynamique
                ]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # fermeture du polygone
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Score Meeting',
                    line_color='#007AFF',
                    fillcolor='rgba(0,122,255,0.2)',
                    line=dict(width=3)
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            showline=True,
                            linewidth=1,
                            gridcolor="lightgray"
                        ),
                        angularaxis=dict(
                            showline=True,
                            linewidth=1,
                            gridcolor="lightgray"
                        )
                    ),
                    showlegend=False,
                    height=350,
                    margin=dict(t=30, b=30, l=30, r=30)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <p>Analyse audio non disponible</p>
                <small>Mode light ne génère pas d'analyse audio</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    def render_recommendations_section(self, result_data: Dict):
        """Affiche la section recommandations"""
        st.markdown("""
        <div class="result-card">
            <div class="card-title">💡 Recommandations Leadership</div>
        """, unsafe_allow_html=True)

        reco_data = self.extract_recommendations_data(result_data)
        recommendations = reco_data['recommandations']
        resume_conseil = reco_data['resume_conseil']

        # Affiche les recommandations si elles existent (liste)
        if isinstance(recommendations, list) and recommendations:
            st.markdown("**💡 Mes recommandations pour améliorer vos meetings :**")
            st.markdown("")
            for i, reco in enumerate(recommendations[:6],1):
                if isinstance(reco, dict):
                    title = reco.get("titre", "Recommandation")
                    description = reco.get("description", "")
                    category = reco.get("categorie", "Général")
                    st.markdown(f"**{i}. {title}** *({category})*")
                    st.markdown(f"{description}")
                    st.markdown("")

        # Affiche le résumé du conseil s'il existe
        if resume_conseil:
            st.markdown(f"**Résumé du conseil :** {resume_conseil}")
            st.markdown("")

        else:
            st.markdown("""
            <div class="empty-state">
                <p>Recommandations non disponibles</p>
                <small>Utilisez le mode 'full' pour générer des recommandations</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    def render_results(self, result: Dict, mode: str):
        """Affiche tous les résultats selon le mode"""
        if not result or not result.get("success"):
            st.error("❌ Échec de l'analyse")
            return

        result_data = result.get("result", {})

        # Section résumé (tous modes)
        self.render_summary_section(result_data)

        # Section audio (optimal/full)
        if mode in ["optimal", "full"]:
            self.render_audio_section(result_data)

        # Section recommandations (full uniquement)
        if mode == "full":
            self.render_recommendations_section(result_data)

        # Debug (collapsible)
        with st.expander("🔧 Données complètes"):
            st.json(result)

    def run(self):
        """Point d'entrée principal de l'interface"""
        # Header
        self.render_header()

        # Sidebar
        mode, model = self.render_sidebar()

        # Zone upload
        uploaded_file = self.render_upload_zone()

        # Traitement
        if uploaded_file:
            file_size_mb = len(uploaded_file.getvalue()) / (1024*1024)

            if file_size_mb <= 100:
                st.success(f"✅ {uploaded_file.name} ({file_size_mb:.1f} MB)")

                if st.button("🚀 Analyser", type="primary"):
                    start_time = time.time()

                    with st.spinner(f"Analyse {mode} en cours..."):
                        file_content = uploaded_file.getvalue()
                        result = self.call_analyze_api(file_content, uploaded_file.name, mode, model)

                    if result:
                        analysis_time = time.time() - start_time
                        st.success(f"✅ Analyse terminée en {analysis_time:.1f}s")
                        self.render_results(result, mode)
            else:
                st.error(f"❌ Fichier trop volumineux: {file_size_mb:.1f} MB (max 100 MB)")

def main():
    """Point d'entrée principal"""
    ui = SummoraUI()
    ui.run()

if __name__ == "__main__":
    main()
