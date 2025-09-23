#!/usr/bin/env python3
"""
Backend Summora V3 - Architecture minimaliste
1 endpoint /analyze avec 3 modes : light, optimal (défaut), full
Audio uniquement - Déploiement local - Portfolio technique
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

# Import interfaces Summora V3
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.main import transcribe_audio_file, analyze_meeting_optimal, analyze_meeting_full

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

class SummoraBackend:
    """Classe principale du backend Summora V3"""

    def __init__(self):
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Crée l'application FastAPI."""
        app = FastAPI(
            title="Summora API V3"
            ,version="3.0"
            ,description="Speech In, Sense Out - Analyse intelligente de meetings"
        )

        # CORS pour Streamlit local
        app.add_middleware(
            CORSMiddleware
            ,allow_origins=["http://localhost:8501"]
            ,allow_methods=["*"]
            ,allow_headers=["*"]
        )

        # Routes
        app.post("/analyze")(self.analyze_audio)
        app.get("/health")(self.health_check)

        return app

    async def analyze_audio(
        self,
        file: UploadFile = File(...)
        ,mode: str = Form("optimal")
        ,model: str = Form("base")
    ) -> Dict[str, Any]:
        """
        Endpoint unique : analyse audio avec 3 modes.

        Args:
            file: Fichier audio (.mp3, .wav, .m4a, .webm)
            mode: Pipeline mode (light, optimal, full)
            model: Modèle Whisper (base, small, medium, large)
        """

        # Validation
        if not self._is_valid_audio_file(file.filename):
            raise HTTPException(400, f"Format audio non supporté: {file.filename}")

        if mode not in ["light", "optimal", "full"]:
            raise HTTPException(400, f"Mode invalide: {mode}. Utilisez: light, optimal, full")

        if model not in ["base", "small", "medium", "large"]:
            raise HTTPException(400, f"Modèle invalide: {model}")

        # Sauvegarde temporaire
        temp_path = self._save_temp_file(file)

        try:
            logger.info(f"Analyse {mode} - modèle {model} - fichier {file.filename}")

            # Routage selon mode
            if mode == "light":
                result = transcribe_audio_file(str(temp_path), model)
            elif mode == "optimal":
                result = analyze_meeting_optimal(str(temp_path), model)
                logger.info(f"Type de result(optimal): {type(result)}")
                logger.info(f"Result keys (optimal): {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            elif mode == "full":
                result = analyze_meeting_full(str(temp_path), model)
                logger.info(f"Type de result(full): {type(result)}")
                logger.info(f"Result keys (full): {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

            # Réponse unifiée
            return {
                "success": True,
                "mode": mode,
                "model_used": model,
                "filename": file.filename,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erreur analyse {mode}: {e}")
            raise HTTPException(500, f"Erreur pipeline {mode}: {str(e)}")

        finally:
            self._cleanup_temp_file(temp_path)

    async def health_check(self) -> Dict[str, Any]:
        """Health check simple."""
        return {
            "status": "healthy",
            "version": "3.0",
            "modes": ["light", "optimal", "full"],
            "models": ["base", "small", "medium", "large"],
            "timestamp": datetime.now().isoformat()
        }

    def _is_valid_audio_file(self, filename: str) -> bool:
        """Validation format audio."""
        if not filename:
            return False
        valid_extensions = {'.mp3', '.wav', '.mp4', '.m4a', '.webm'}
        return Path(filename).suffix.lower() in valid_extensions

    def _save_temp_file(self, file: UploadFile) -> Path:
        """Sauvegarde temporaire sécurisée."""
        safe_filename = self._sanitize_filename(file.filename)
        temp_path = TEMP_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename}"

        try:
            with open(temp_path, "wb") as f:
                content = file.file.read()
                f.write(content)
            logger.info(f"Fichier sauvé: {temp_path.name}")
            return temp_path
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            raise HTTPException(500, f"Erreur sauvegarde: {str(e)}")

    def _cleanup_temp_file(self, file_path: Path):
        """Nettoyage fichier temporaire."""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Fichier nettoyé: {file_path.name}")
        except Exception as e:
            logger.warning(f"Erreur nettoyage {file_path}: {e}")

    def _sanitize_filename(self, filename: str) -> str:
        """Sécurisation nom de fichier."""
        if not filename:
            return "audio_file"
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        sanitized = "".join(c for c in filename if c in safe_chars)
        return sanitized or "audio_file"

# Instance globale
summora_backend = SummoraBackend()
app = summora_backend.app

# Point d'entrée pour débogage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
