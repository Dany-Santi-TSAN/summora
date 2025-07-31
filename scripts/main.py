#!/usr/bin/env python3
"""
Summora V3 - Orchestrateur
Speech In, Sense Out

Architecture modulaire : Configuration + CLI + Runner
"""
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
import yaml
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv

# Setup path pour imports Summora
sys.path.append(str(Path(__file__).parent.parent))

# Import des modules spécialisés
from src.core.handle_error import PipelineResult, ScriptResult
from src.core.ressource_monitoring import cleanup_gpu_memory

# Chargement des variables d'environnement
env_path = Path(".env")

if env_path.exists():
    load_dotenv(env_path, override=True)
    logging.debug(f"✅ .env chargé depuis: {env_path.absolute()}")
else:
    logging.warning(f"⚠️ Fichier .env non trouvé: {env_path.absolute()}")
    load_dotenv()  # Fallback standard

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Configuration ===

@dataclass
class SummoraConfig:
    """Configuration simple avec dataclass."""
    model: str = 'medium'
    language: str = 'fr'
    enable_spot_check: bool = False
    enable_visual: bool = False
    enable_correction: bool = False
    enable_recommendation: bool = False
    temperature: float = 0.0
    min_confidence: float = 0.7

    # API Keys depuis .env
    openrouter_api_key: Optional[str] = None

    def __post_init__(self):
        """Charge les clés API depuis les variables d'environnement."""
        # Clés API depuis .env
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

        # Debug : afficher les variables trouvées (sans révéler les clés)
        env_vars_found = []

        if self.openrouter_api_key:
            env_vars_found.append(f"OPENROUTER_API_KEY (***{self.openrouter_api_key[-4:]})")

        # Log des clés disponibles
        api_keys_status = []
        if self.openrouter_api_key:
            api_keys_status.append("OpenRouter")

        if api_keys_status:
            logger.info(f"🔑 Clés API chargées: {', '.join(api_keys_status)}")
            if logger.level <= logging.DEBUG:
                logger.debug(f"Variables trouvées: {env_vars_found}")
        else:
            logger.warning("⚠️ Aucune clé API trouvée - fonctionnalités LLM limitées")
            logger.warning("💡 Vérifiez votre fichier .env ou les variables d'environnement")

            # Debug aide
            current_env_vars = [k for k in os.environ.keys() if 'API_KEY' in k]
            if current_env_vars:
                logger.debug(f"Variables API_KEY trouvées dans l'environnement: {current_env_vars}")
            else:
                logger.debug("Aucune variable *API_KEY* trouvée dans l'environnement")

    @classmethod
    def load_from_yaml(cls, config_file: str = "config.yaml") -> 'SummoraConfig':
        """Charge config depuis YAML."""
        config_path = Path(config_file)

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f) or {}
                logger.info(f"✅ Config chargée: {config_file}")

                # Merge avec defaults via dataclass
                return cls(**{k: v for k, v in yaml_data.items() if hasattr(cls, k)})

            except Exception as e:
                logger.warning(f"⚠️ Erreur config YAML: {e}, utilisation defaults")
        else:
            logger.info("📋 Utilisation configuration par défaut")

        return cls()  # Defaults dataclass

    def create_template(self) -> str:
        """Crée un fichier config.yaml et .env template."""
        # Template config.yaml
        config_template = f"""# Summora - Configuration Simple
model: "{self.model}"
language: "{self.language}"

# BONUS Features (False par défaut, CŒUR = transcription + extraction)
enable_spot_check: {str(self.enable_spot_check).lower()}
enable_recommendation: {str(self.enable_recommendation).lower()}
enable_correction: {str(self.enable_correction).lower()}
enable_visual: {str(self.enable_visual).lower()}

# Options avancées
temperature: {self.temperature}
min_confidence: {self.min_confidence}
"""

        # Template .env
        env_template = """# Summora V3 - Variables d'environnement
# ⚠️⚠️ Clés API (⚠️⚠️ remplacez par vos vraies clés ⚠️⚠️)

# OpenRouter (pour accès multiple modèles via une API)
OPENROUTER_API_KEY=your-openrouter-key-here

# Exemple de rajout
# OpenAI (pour GPT-4, GPT-3.5)
OPENAI_API_KEY=sk-your-openai-key-here

# Optionnel: modèles locaux
# OLLAMA_HOST=http://localhost:11434
# HF_TOKEN=your-huggingface-token
"""

        # Création config.yaml
        config_path = Path("config.yaml")
        if not config_path.exists():
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_template)
            logger.info(f"✅ Fichier config.yaml créé")
        else:
            logger.info(f"📋 Fichier config.yaml existe déjà")

        # Création .env
        env_path = Path(".env")
        if not env_path.exists():
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(env_template)
            logger.info(f"✅ Fichier .env créé")
            logger.warning("⚠️ N'oubliez pas d'ajouter vos vraies clés API dans .env!")
        else:
            logger.info(f"📋 Fichier .env existe déjà")

        return str(config_path)

    def validate_api_keys(self) -> Dict[str, bool]:
        """Valide la disponibilité des clés API."""
        return {
            "openrouter": bool(self.openrouter_api_key and len(self.openrouter_api_key) > 10)
        }

# === CLI Arguments ===

@dataclass
class SummoraCLI:
    """Arguments CLI avec dataclass."""
    # Fichier d'entrée (obligatoire si pas create_config)
    input_file: Optional[str] = None

    # Configuration de base
    model: str = "auto"
    language: str = "fr"
    config: str = "config.yaml"

    # Modes individuels
    transcribe_only: bool = False
    extract_only: bool = False
    reco_only: bool = False
    correct_only: bool = False
    visual_only: bool = False

    # Pipeline modes
    light: bool = False
    all: bool = False

    # Options bonus
    with_visual: bool = False
    with_reco: bool = False
    with_correction: bool = False
    with_spotcheck: bool = False

    # Système
    verbose: bool = False
    quiet: bool = False
    create_config: bool = False

class SummoraCLIHandler:
    """Interface CLI simple avec dataclass."""

    def get_args(self) -> SummoraCLI:
        """Arguments CLI simples."""
        parser = argparse.ArgumentParser(
            description="Summora V3 - Orchestrateur modulaire",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
        Exemples d'usage:
  python main.py audio.mp3 --transcribe-only               # Transcription seule
  python main.py transcription.txt --extract-only          # Extraction seule
  python main.py extraction.json --reco-only               # Recommandations seules
  python main.py audio.mp3 --light                         # Pipeline CŒUR (transcribe -> extract)
  python main.py audio.mp3 --light --with-reco             # CŒUR + recommandations
  python main.py audio.mp3 --all                           # Pipeline COMPLET (CŒUR + tous bonus)
  python main.py audio.mp3 --with-visual --with-correction # Options bonus personnalisées
            """
        )

        # Fichier d'entrée
        parser.add_argument("input_file", nargs='?', help="Fichier d'entrée")

        # Modes individuels
        parser.add_argument("--transcribe-only", action="store_true",
                           help="Transcription seule")
        parser.add_argument("--extract-only", action="store_true",
                           help="Extraction seule")
        parser.add_argument("--reco-only", action="store_true",
                           help="Recommandations seules")
        parser.add_argument("--correct-only", action="store_true",
                           help="Correction seule")
        parser.add_argument("--visual-only", action="store_true",
                           help="Analyse audio seule")

        # Pipeline modes
        parser.add_argument("--light", action="store_true",
                           help="Pipeline CŒUR (transcribe → extract)")
        parser.add_argument("--all", action="store_true",
                           help="Pipeline COMPLET (CŒUR + tous les bonus)")

        # Options bonus (cœur = transcription + extraction)
        parser.add_argument("--with-visual", action="store_true",
                           help="BONUS: Ajouter analyse audio")
        parser.add_argument("--with-reco", action="store_true",
                           help="BONUS: Ajouter recommandations (sinon skip)")
        parser.add_argument("--with-correction", action="store_true",
                           help="BONUS: Ajouter correction")
        parser.add_argument("--with-spotcheck", action="store_true",
                           help="BONUS: Activer spot-check QA")

        # Configuration
        parser.add_argument("--model", default="auto",
                           choices=["tiny", "small", "medium", "large", "auto"],
                           help="Modèle Whisper")
        parser.add_argument("--language", default="fr", help="Langue")
        parser.add_argument("--config", default="config.yaml", help="Fichier config")

        # Système
        parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
        parser.add_argument("--quiet", "-q", action="store_true", help="Mode silencieux")
        parser.add_argument("--create-config", action="store_true",
                           help="Créer fichier config.yaml")

        args = parser.parse_args()

        # Conversion en dataclass
        return SummoraCLI(
            input_file=args.input_file
            ,transcribe_only=args.transcribe_only
            ,extract_only=args.extract_only
            ,reco_only=args.reco_only
            ,correct_only=args.correct_only
            ,visual_only=args.visual_only
            ,light=args.light
            ,all=args.all
            ,with_visual=args.with_visual
            ,with_reco=args.with_reco
            ,with_correction=args.with_correction
            ,with_spotcheck=args.with_spotcheck
            ,model=args.model
            ,language=args.language
            ,config=args.config
            ,verbose=args.verbose
            ,quiet=args.quiet
            ,create_config=args.create_config
        )

    def setup_logging(self, args: SummoraCLI) -> None:
        """Configuration logging selon args CLI."""
        if args.quiet:
            logging.getLogger().setLevel(logging.WARNING)
        elif args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

# === Script Runner ===

class SummoraRunner:
    """Runner pour exécuter les scripts."""

    def run_script(self, script_name: str, args: List[str]) -> ScriptResult:
        """
        Exécute un script dans le dossier scripts/ avec gestion d'erreurs.

        Args:
            script_name: Nom du script (ex: 'main_transcribe')
            args: Liste des arguments

        Retourne:
            ScriptResult: Résultat d'exécution
        """
        script_path = Path("scripts") / f"{script_name}.py"

        if not script_path.exists():
            return ScriptResult(
                success=False,
                error=f"Script non trouvé: {script_path}",
                script_name=script_name
            )

        cmd = [sys.executable, str(script_path)] + args

        try:
            logger.info(f"🧩 Exécution: {script_name} {' '.join(args[:2])}")
            start_time = datetime.now()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                logger.info(f"✅ {script_name} terminé en {execution_time:.2f}s")
                return ScriptResult(
                    success=True,
                    output=result.stdout,
                    script_name=script_name,
                    execution_time=execution_time
                )
            else:
                logger.error(f"❌ {script_name} échoué en {execution_time:.2f}s")
                return ScriptResult(
                    success=False,
                    error=result.stderr,
                    output=result.stdout,
                    script_name=script_name,
                    execution_time=execution_time
                )

        except Exception as e:
            logger.error(f"❌ Erreur exécution {script_name}: {e}")
            return ScriptResult(
                success=False,
                error=str(e),
                script_name=script_name
            )

    def find_latest_output(self, pattern: str) -> Optional[str]:
        """Trouve le fichier le plus récent selon un pattern."""
        import glob

        files = glob.glob(pattern)
        if files:
            # Tri par date de modification (le plus récent en premier)
            latest = max(files, key=lambda f: Path(f).stat().st_mtime)
            logger.info(f"📁 Fichier détecté: {Path(latest).name}")
            return latest
        return None

# === Pipeline Modulaire ===

class SummoraPipeline:
    """Pipeline utilisant les modules spécialisés."""

    def __init__(self, config: SummoraConfig, runner: SummoraRunner):
        self.config = config
        self.runner = runner

    def transcribe_audio(self, input_file: str, args: SummoraCLI) -> ScriptResult:
        """Exécute la transcription audio."""
        cmd_args = [input_file]

        # Configuration du modèle
        model = args.model if args.model != 'auto' else self.config.model
        if model != 'auto':
            cmd_args.extend(['--model', model])

        # Langue
        language = args.language if args.language != 'fr' else self.config.language
        if language != 'fr':
            cmd_args.extend(['--language', language])

        # Métadonnées JSON pour chaînage
        cmd_args.append('--json-metadata')

        # Options système
        if args.verbose:
            cmd_args.append('--verbose')
        if args.quiet:
            cmd_args.append('--quiet')

        result = self.runner.run_script('main_transcribe', cmd_args)

        # Cleanup GPU après Whisper (gourmand en VRAM)
        cleanup_gpu_memory()

        return result

    def extract_content(self, input_file: str, args: SummoraCLI) -> ScriptResult:
        """Exécute l'extraction de contenu."""
        cmd_args = [input_file]

        # Options d'extraction
        cmd_args.append('--with-eval')  # Par défaut

        if args.with_spotcheck or self.config.enable_spot_check:
            cmd_args.append('--enable-spot-check')

        # Options système
        if args.verbose:
            cmd_args.append('--verbose')
        if args.quiet:
            cmd_args.append('--quiet')

        return self.runner.run_script('main_extract', cmd_args)

    def generate_recommendations(self, input_file: str, args: SummoraCLI) -> ScriptResult:
        """Exécute la génération de recommandations."""
        cmd_args = [input_file]

        # Options recommandations
        if args.reco_only:
            cmd_args.append('--reco-only')

        if args.with_spotcheck or self.config.enable_spot_check:
            cmd_args.append('--enable-spot-check')

        # Options système
        if args.verbose:
            cmd_args.append('--verbose')
        if args.quiet:
            cmd_args.append('--quiet')

        result = self.runner.run_script('main_reco', cmd_args)

        # Cleanup GPU après LLM (peut utiliser beaucoup de VRAM)
        cleanup_gpu_memory()

        return result

    def correct_transcription(self, input_file: str, args: SummoraCLI) -> ScriptResult:
        """Exécute la correction de transcription."""
        cmd_args = [input_file]

        # Options correction
        if args.correct_only:
            cmd_args.append('--correction-only')

        if args.with_spotcheck or self.config.enable_spot_check:
            cmd_args.append('--enable-spot-check')

        # Options système
        if args.verbose:
            cmd_args.append('--verbose')

        result = self.runner.run_script('main_corrector', cmd_args)

        # Cleanup GPU après correction LLM
        cleanup_gpu_memory()

        return result

    def analyze_audio(self, input_file: str, args: SummoraCLI) -> ScriptResult:
        """Exécute l'analyse audio."""
        cmd_args = [input_file]

        # Options analyse
        if self.config.enable_visual:
            cmd_args.append('--plots')

        # Options système
        if args.verbose:
            cmd_args.append('--verbose')
        if args.quiet:
            cmd_args.append('--quiet')

        return self.runner.run_script('main_visual', cmd_args)

# === Pipeline Principal ===

def run_summora_pipeline(input_file: str, args: SummoraCLI, config: SummoraConfig) -> PipelineResult:
    """
    Fonction principale du pipeline - retourne un PipelineResult.
    """
    start_time = datetime.now()
    result = PipelineResult(success=True)

    # Initialisation
    runner = SummoraRunner()
    pipeline = SummoraPipeline(config, runner)

    try:
        input_path = Path(input_file)
        if not input_path.exists():
            result.success = False
            result.add_error(f"Fichier non trouvé: {input_file}")
            return result

        current_file = str(input_path)
        file_ext = input_path.suffix.lower()

        # Détection du type de fichier
        is_audio = file_ext in {'.mp3', '.wav', '.mp4', '.m4a', '.webm'}
        is_transcription = file_ext in {'.txt'} or 'transcription' in input_path.name
        is_extraction = file_ext in {'.json'} and 'extraction' in input_path.name

        result.add_result("file_detection", {
            "input_file": current_file,
            "file_type": "audio" if is_audio else "transcription" if is_transcription else "extraction" if is_extraction else "unknown",
            "extension": file_ext
        })

        # === MODES INDIVIDUELS ===
        if args.transcribe_only:
            logger.info("🎤 Mode: Transcription seule")
            script_result = pipeline.transcribe_audio(current_file, args)
            if script_result.success:
                result.add_result("transcription", {"output": script_result.output})
                # Chercher les fichiers créés
                transcription_json = runner.find_latest_output("output/transcriptions/*.json")
                if transcription_json:
                    result.add_output_file(transcription_json)
            else:
                result.success = False
                result.add_error(script_result.error, "transcription")

        elif args.extract_only:
            logger.info("🎯 Mode: Extraction seule")
            script_result = pipeline.extract_content(current_file, args)
            if script_result.success:
                result.add_result("extraction", {"output": script_result.output})
                extraction_json = runner.find_latest_output("output/extractions/*.json")
                if extraction_json:
                    result.add_output_file(extraction_json)
            else:
                result.success = False
                result.add_error(script_result.error, "extraction")

        elif args.reco_only:
            logger.info("💡 Mode: Recommandations seules")
            script_result = pipeline.generate_recommendations(current_file, args)
            if script_result.success:
                result.add_result("recommendations", {"output": script_result.output})
                reco_json = runner.find_latest_output("output/recommendations/*.json")
                if reco_json:
                    result.add_output_file(reco_json)
            else:
                result.success = False
                result.add_error(script_result.error, "recommendations")

        elif args.correct_only:
            logger.info("🔧 Mode: Correction seule")
            script_result = pipeline.correct_transcription(current_file, args)
            if script_result.success:
                result.add_result("correction", {"output": script_result.output})
            else:
                result.success = False
                result.add_error(script_result.error, "correction")

        elif args.visual_only:
            logger.info("🎵 Mode: Analyse audio seule")
            script_result = pipeline.analyze_audio(current_file, args)
            if script_result.success:
                result.add_result("audio_analysis", {"output": script_result.output})
            else:
                result.success = False
                result.add_error(script_result.error, "audio_analysis")

        # === PIPELINE MODES ===
        else:  # Pipeline complet par défaut
            if args.all:
                logger.info("🚀 Mode: Pipeline COMPLET (CŒUR + tous bonus)")
                # Activer tous les bonus pour --all
                args.with_visual = True
                args.with_reco = True
                args.with_correction = True
            else:
                logger.info("🚀 Mode: Pipeline CŒUR (transcribe → extract)")

            # 1. Analyse audio (si demandée et fichier audio)
            if (args.with_visual or config.enable_visual) and is_audio:
                logger.info("🎵 Étape 1: Analyse audio (BONUS)")
                script_result = pipeline.analyze_audio(current_file, args)
                if script_result.success:
                    result.add_result("audio_analysis", {"output": script_result.output})
                else:
                    result.add_error(script_result.error, "audio_analysis")
                    logger.warning("⚠️ Analyse audio échouée, continuation du pipeline")

            # 2. Transcription (si fichier audio) - CŒUR
            if is_audio:
                logger.info("🎤 Étape 2: Transcription (CŒUR)")
                script_result = pipeline.transcribe_audio(current_file, args)
                if script_result.success:
                    result.add_result("transcription", {"output": script_result.output})
                    # Recherche du fichier JSON de transcription généré
                    transcription_json = runner.find_latest_output("output/transcriptions/*.json")
                    if transcription_json:
                        current_file = transcription_json
                        result.add_output_file(transcription_json)
                    else:
                        result.add_error("Fichier JSON transcription non trouvé", "transcription")
                else:
                    result.success = False
                    result.add_error(script_result.error, "transcription")
                    return result  # Arrêt du pipeline sur erreur CŒUR

            elif is_transcription:
                logger.info("📝 Input détecté: Fichier transcription")
                result.add_result("input_type", {"type": "transcription_file"})

            # 3. Correction (si demandée) - BONUS
            if (args.with_correction or config.enable_correction) and not is_extraction:
                logger.info("🔧 Étape 3: Correction (BONUS)")
                script_result = pipeline.correct_transcription(current_file, args)
                if script_result.success:
                    result.add_result("correction", {"output": script_result.output})
                else:
                    result.add_error(script_result.error, "correction")
                    logger.warning("⚠️ Correction échouée, continuation du pipeline")

            # 4. Extraction (si pas déjà un fichier d'extraction) - CŒUR
            if not is_extraction:
                logger.info("🎯 Étape 4: Extraction (CŒUR)")
                script_result = pipeline.extract_content(current_file, args)
                if script_result.success:
                    result.add_result("extraction", {"output": script_result.output})
                    # Recherche du fichier JSON d'extraction généré
                    extraction_json = runner.find_latest_output("output/extractions/*.json")
                    if extraction_json:
                        current_file = extraction_json
                        result.add_output_file(extraction_json)
                    else:
                        result.add_error("Fichier JSON extraction non trouvé", "extraction")
                else:
                    result.success = False
                    result.add_error(script_result.error, "extraction")
                    return result  # Arrêt du pipeline sur erreur CŒUR

            elif is_extraction:
                logger.info("📊 Input détecté: Fichier extraction")
                result.add_result("input_type", {"type": "extraction_file"})

            # 5. Recommandations (BONUS - si demandées via CLI ou config)
            if args.with_reco or config.enable_recommendation:
                logger.info("💡 Étape 5: Recommandations (BONUS)")
                script_result = pipeline.generate_recommendations(current_file, args)
                if script_result.success:
                    result.add_result("recommendations", {"output": script_result.output})
                    reco_json = runner.find_latest_output("output/recommendations/*.json")
                    if reco_json:
                        result.add_output_file(reco_json)
                else:
                    result.add_error(script_result.error, "recommendations")
                    logger.warning("⚠️ Recommandations échouées (bonus)")
            else:
                logger.info("⏭️ Recommandations skippées (utilisez --with-reco pour activer)")
                result.add_result("recommendations_skipped", {"reason": "not_requested"})

            logger.info("🎉 Pipeline terminé! (CŒUR + bonus activés)")

    except Exception as e:
        logger.error(f"❌ Erreur pipeline: {e}")
        result.success = False
        result.add_error(str(e), "pipeline_fatal")
        return result

    return result

# === Calcul de coûts LLM ===

def calculate_precise_llm_cost(prompt: str, response: str, tokenizer,
                               input_rate_eur_per_1m: float, output_rate_eur_per_1m: float) -> Dict:
    """
    Calcule le coût précis d'un appel LLM via tokenizer.

    Args:
        prompt: Texte d'entrée
        response: Réponse générée
        tokenizer: Tokenizer du modèle
        input_rate_eur_per_1m: Tarif input €/1M tokens
        output_rate_eur_per_1m: Tarif output €/1M tokens

    Returns:
        Dict: Métriques complètes (tokens, coûts, ratios)
    """
    try:
        # Calcul précis via tokenizer
        input_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(response))

        # Calcul coûts
        input_cost = (input_tokens / 1_000_000) * input_rate_eur_per_1m
        output_cost = (output_tokens / 1_000_000) * output_rate_eur_per_1m
        total_cost = input_cost + output_cost

        # Métriques bonus pour analyse
        total_tokens = input_tokens + output_tokens
        cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
        output_ratio = output_tokens / input_tokens if input_tokens > 0 else 0

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost_eur": round(input_cost, 6),
            "output_cost_eur": round(output_cost, 6),
            "total_cost_eur": round(total_cost, 6),
            "cost_per_token_eur": round(cost_per_token, 8),
            "output_input_ratio": round(output_ratio, 2),
            "rates_used": {
                "input_eur_per_1m": input_rate_eur_per_1m,
                "output_eur_per_1m": output_rate_eur_per_1m
            }
        }

    except Exception as e:
        logger.warning(f"⚠️ Erreur calcul tokenizer: {e}")
        # Fallback estimation
        estimated_input = len(prompt.split()) * 1.3
        estimated_output = len(response.split()) * 1.3
        estimated_cost = ((estimated_input + estimated_output) / 1_000_000) * 0.25

        return {
            "input_tokens": int(estimated_input),
            "output_tokens": int(estimated_output),
            "total_tokens": int(estimated_input + estimated_output),
            "total_cost_eur": round(estimated_cost, 6),
            "estimation_method": "fallback_word_count",
            "warning": "Calcul approximatif - tokenizer indisponible"
        }

# Tarifs OpenRouter Juillet 2025 (€/1M tokens)
OPENROUTER_RATES = {
    "qwen/qwen-72b-chat": {"input": 0.09, "output": 0.35},
    "anthropic/claude-3.5-sonnet": {"input": 0.3, "output": 1.5},
    "openai/gpt-4-turbo": {"input": 0.8, "output": 2.4},
    "google/gemini-pro": {"input": 0.035, "output": 0.105},
    "meta-llama/llama-3-70b": {"input": 0.05, "output": 0.2}
}

def get_model_rates(model_name: str) -> Dict[str, float]:
    """Retourne les tarifs pour un modèle (avec fallback)."""
    # Recherche exacte
    if model_name in OPENROUTER_RATES:
        return OPENROUTER_RATES[model_name]

    # Recherche partielle (ex: "qwen-72b" match "qwen/qwen-72b-chat")
    for rate_model, rates in OPENROUTER_RATES.items():
        if any(part in rate_model.lower() for part in model_name.lower().split("-")):
            return rates

    # Fallback tarifs moyens
    logger.warning(f"⚠️ Tarifs inconnus pour {model_name}, utilisation tarifs moyens")
    return {"input": 0.1, "output": 0.4}

# === Interfaces programmatiques ===

def transcribe_audio_file(audio_path: str, model: str = "base", language: str = "fr", **kwargs) -> Dict:
    """
    Interface simple pour transcription.

    Args:
        audio_path: Chemin vers fichier audio
        model: Modèle Whisper
        language: Langue
        **kwargs: Options supplémentaires

    Returns:
        Dict: Résultats de transcription
    """
    args = SummoraCLI(
        input_file=audio_path,
        transcribe_only=True,
        model=model,
        quiet=True,
        **kwargs
    )

    config = SummoraConfig(model=model, language=language)
    result = run_summora_pipeline(audio_path, args, config)

    return result.to_dict()

def extract_meeting_insights(input_file: str, **kwargs) -> Dict:
    """
    Interface simple pour extraction.

    Args:
        input_file: Fichier transcription ou audio
        **kwargs: Options supplémentaires

    Returns:
        Dict: Insights extraits
    """
    args = SummoraCLI(
        input_file=input_file,
        light=True,
        quiet=True,
        **kwargs
    )

    config = SummoraConfig()
    result = run_summora_pipeline(input_file, args, config)

    return result.to_dict()

def generate_meeting_recommendations(extraction_file: str, **kwargs) -> Dict:
    """
    Interface simple pour recommandations.

    Args:
        extraction_file: Fichier JSON d'extraction
        **kwargs: Options supplémentaires

    Returns:
        Dict: Recommandations générées
    """
    args = SummoraCLI(
        input_file=extraction_file,
        reco_only=True,
        quiet=True,
        **kwargs
    )

    config = SummoraConfig(enable_recommendation=True)
    result = run_summora_pipeline(extraction_file, args, config)

    return result.to_dict()

def analyze_meeting_full_pipeline(audio_path: str, include_all_features: bool = False, **kwargs) -> Dict:
    """
    Pipeline complet.

    Args:
        audio_path: Chemin vers fichier audio
        include_all_features: Activer tous les bonus
        **kwargs: Options supplémentaires

    Returns:
        Dict: Résultats complets du pipeline
    """
    args = SummoraCLI(
        input_file=audio_path,
        all=include_all_features,
        light=not include_all_features,
        with_reco=True,
        quiet=True,
        **kwargs
    )

    config = SummoraConfig(
        enable_recommendation=True,
        enable_visual=include_all_features,
        enable_correction=include_all_features
    )

    result = run_summora_pipeline(audio_path, args, config)

    return result.to_dict()

# === Utilitaires ===

def load_pipeline_results(report_file: str) -> Dict:
    """Charge les résultats d'un rapport pipeline."""
    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_latest_pipeline_report(reports_dir: str = "output/reports") -> Optional[str]:
    """Trouve le dernier rapport de pipeline généré."""
    import glob

    pattern = f"{reports_dir}/pipeline_report_*.json"
    files = glob.glob(pattern)

    if files:
        return max(files, key=lambda f: Path(f).stat().st_mtime)
    return None

def main():
    """Point d'entrée principal."""

    # 1. CLI + Configuration
    cli = SummoraCLIHandler()
    args = cli.get_args()
    cli.setup_logging(args)

    # 2. Création config si demandée
    if args.create_config:
        config = SummoraConfig()
        config_path = config.create_template()

        # Affichage du statut des clés API
        api_status = config.validate_api_keys()
        print(f"✅ Configuration créée: {config_path}")
        print(f"🔑 Statut des clés API: {api_status}")
        if not any(api_status.values()):
            print("⚠️ Ajoutez vos clés API dans le fichier .env pour activer les fonctionnalités LLM")
        return

    # 3. Validation input
    if not args.input_file:
        logger.error("❌ Fichier d'entrée requis")
        return

    # 4. Exécution du pipeline
    config = SummoraConfig.load_from_yaml(args.config)

    try:
        # Exécution du pipeline principal
        pipeline_result = run_summora_pipeline(args.input_file, args, config)

        # Affichage des résultats
        if pipeline_result.success:
            logger.info("🎉 Pipeline exécuté avec succès!")
            logger.info(f"⏱️ Temps d'exécution: {pipeline_result.execution_time:.2f}s")
            logger.info(f"📁 Fichiers créés: {len(pipeline_result.outputs_created)}")
            for output_file in pipeline_result.outputs_created:
                logger.info(f"   • {Path(output_file).name}")

            # En mode verbose, afficher tous les résultats
            if args.verbose:
                logger.info("📊 Résultats détaillés:")
                for step, data in pipeline_result.results.items():
                    logger.info(f"   • {step}: {len(str(data))} caractères de données")

        else:
            logger.error("❌ Pipeline échoué!")
            logger.error(f"⏱️ Temps d'exécution: {pipeline_result.execution_time:.2f}s")
            logger.error(f"🚨 Erreurs ({len(pipeline_result.errors)}):")
            for error in pipeline_result.errors:
                logger.error(f"   • {error}")

        # Sauvegarde du rapport pour analyse
        pipeline_result.save_report()

        # Return du résultat pour usage programmatique
        return pipeline_result

    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrompu par l'utilisateur")
        return PipelineResult(success=False, errors=["Pipeline interrompu"])

    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return PipelineResult(success=False, errors=[f"Erreur fatale: {str(e)}"])


if __name__ == "__main__":
    main()

    # Le script peut aussi être importé et utilisé programmatiquement :
    # from main import transcribe_audio_file, analyze_meeting_full_pipeline
    # results = transcribe_audio_file("meeting.mp3", model="medium")
    # insights = analyze_meeting_full_pipeline("meeting.mp3", include_all_features=True)
