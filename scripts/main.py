#!/usr/bin/env python3
"""
Summora V3 - Orchestrateur Principal Simplifié
------------------------------------------------
🎙️ Speech In, Sense Out — Architecture modulaire épurée

Orchestrateur pur avec 3 modes pipeline :
- light : transcribe → extract
- optimal : transcribe → extract + audio_analysis (défaut)
- full : transcribe → correction → extract + audio_analysis → reco

Pour debug/usage avancé : utiliser directement scripts/main_*.py
"""

import sys
import subprocess
import argparse
import logging
import json
import glob
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Setup imports
sys.path.append(str(Path(__file__).parent.parent))
from src.core.handle_error import PipelineResult, ScriptResult
from src.core.ressource_monitoring import cleanup_gpu_memory

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===

@dataclass
class SummoraConfig:
    """Configuration unifiée."""
    model: str = "base"  # RAM 2GB constraint
    language: str = "fr"

    @classmethod
    def load_env(cls) -> 'SummoraConfig':
        """Charge config depuis .env."""
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=True)
        else:
            load_dotenv()
        return cls()

# === CLI ===

@dataclass
class CLIArgs:
    """Arguments CLI."""
    input_file: str
    mode: str = "optimal"  # optimal par défaut
    model: str = "base"
    language: str = "fr"
    verbose: bool = False
    quiet: bool = False

def parse_cli() -> CLIArgs:
    """Parse CLI simplifié."""
    parser = argparse.ArgumentParser(
        description="Summora V3 - Speech to Sense Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes pipeline:
  python main.py audio.mp3                    # Optimal (défaut): transcribe → extract + audio_analysis
  python main.py audio.mp3 --light            # Light: transcribe → extract
  python main.py audio.mp3 --full             # Full: transcribe → correction → extract + audio_analysis → reco

Modèles Whisper:
  python main.py audio.mp3 --model medium     # Upgrade modèle (si RAM suffisante)
        """
    )

    parser.add_argument("input_file", help="Fichier audio (.mp3, .wav, .m4a, etc.)")

    # Modes pipeline
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--light", action="store_true", help="Pipeline minimal")
    mode_group.add_argument("--full", action="store_true", help="Pipeline complet avec recommandations")

    # Config
    parser.add_argument("--model", default="base",
                       choices=["base", "small", "medium", "large"],
                       help="Modèle Whisper (défaut: base)")
    parser.add_argument("--language", default="fr", help="Langue (défaut: fr)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()

    # Détermination du mode
    if args.light:
        mode = "light"
    elif args.full:
        mode = "full"
    else:
        mode = "optimal"  # Défaut

    return CLIArgs(
        input_file=args.input_file,
        mode=mode,
        model=args.model,
        language=args.language,
        verbose=args.verbose,
        quiet=args.quiet
    )

# === RUNNER ===

class ScriptRunner:
    """Runner unifié pour exécuter les scripts."""

    def run_script(self, script_name: str, args: List[str]) -> ScriptResult:
        """Exécute un script avec gestion d'erreurs."""
        script_path = Path("scripts") / f"{script_name}.py"

        if not script_path.exists():
            return ScriptResult(
                success=False,
                error=f"Script non trouvé: {script_path}",
                script_name=script_name
            )

        cmd = [sys.executable, str(script_path)] + args

        try:
            logger.info(f"Exécution: {script_name}")
            start_time = datetime.now()

            result = subprocess.run(cmd, capture_output=False, text=True, check=False)
            execution_time = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                logger.info(f"Succès {script_name} ({execution_time:.2f}s)")
                return ScriptResult(
                    success=True,
                    output=result.stdout,
                    script_name=script_name,
                    execution_time=execution_time
                )
            else:
                logger.error(f"Échec {script_name}")
                return ScriptResult(
                    success=False,
                    error=result.stderr,
                    output=result.stdout,
                    script_name=script_name,
                    execution_time=execution_time
                )

        except Exception as e:
            logger.error(f"Exception {script_name}: {e}")
            return ScriptResult(
                success=False,
                error=str(e),
                script_name=script_name
            )

    def find_latest_output(self, pattern: str, context_file: str = None) -> Optional[str]:
        """Trouve le fichier le plus récent avec contexte optionnel."""
        import glob

        # Si contexte fourni, injecter le nom audio dans le pattern
        if context_file:
            audio_name = Path(context_file).stem
            pattern = pattern.replace("*", f"*{audio_name}*", 1)
            logger.info(f"Pattern contextualisé: {pattern}")

        files = glob.glob(pattern)
        if files:
            latest = max(files, key=lambda f: Path(f).stat().st_mtime)
            logger.info(f"Fichier détecté: {Path(latest).name}")
            return latest
        return None

    def _build_base_args(self, cli_args: CLIArgs) -> List[str]:
        """Args de base pour tous les scripts."""
        args = []
        if cli_args.verbose:
            args.append("--verbose")
        if cli_args.quiet:
            args.append("--quiet")
        return args

# === PIPELINE RUNNER ===

class SummoraPipelineRunner:
    """Pipeline runner : light, optimal, full."""

    def __init__(self, runner: ScriptRunner):
        self.runner = runner

    def run_light(self, audio_file: str, cli_args: CLIArgs) -> PipelineResult:
        """Pipeline light : transcribe → extract."""
        start_time = datetime.now()
        result = PipelineResult(success=True)

        try:
            # Étape 1: Transcription
            logger.info("Pipeline Light - Étape 1/2: Transcription")
            args = [audio_file, "--model", cli_args.model, "--language", cli_args.language]
            args.extend(self.runner._build_base_args(cli_args))

            script_result = self.runner.run_script("main_transcribe", args)
            if not script_result.success:
                result.success = False
                result.add_error(script_result.error, "transcription")
                return result

            cleanup_gpu_memory()

            # Lecture fichier transcription
            transcription_file = self.runner.find_latest_output("output/transcriptions/*_raw.txt", audio_file)
            if not transcription_file:
                result.success = False
                result.add_error("Fichier transcription non trouvé")
                return result

            with open(transcription_file, 'r', encoding='utf-8') as f:
                transcription_text = f.read()

            result.add_result("transcription", {"text": transcription_text, "file": transcription_file})
            result.add_output_file(transcription_file)

            # Étape 2: Extraction
            logger.info("Pipeline Light - Étape 2/2: Extraction")
            args = [transcription_file, "--with-eval"]
            args.extend(self.runner._build_base_args(cli_args))

            script_result = self.runner.run_script("main_extract", args)
            if not script_result.success:
                result.success = False
                result.add_error(script_result.error, "extraction")
                return result

            # Lecture fichier extraction
            extraction_json = self.runner.find_latest_output("output/extractions/*.json", audio_file)
            logger.info(f"EXTRACTION DEBUG: extraction_json = {extraction_json}")
            if extraction_json:
                with open(extraction_json, 'r', encoding='utf-8') as f:
                    extraction_data = json.load(f)
                result.add_result("extraction", extraction_data)
                result.add_output_file(extraction_json)

            logger.info("Pipeline Light terminé")

        except Exception as e:
            logger.error(f"Erreur pipeline light: {e}")
            result.success = False
            result.add_error(str(e), "pipeline_light")

        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result

    def run_optimal(self, audio_file: str, cli_args: CLIArgs) -> PipelineResult:
        """Pipeline optimal : transcribe → extract + audio_analysis."""
        start_time = datetime.now()
        result = PipelineResult(success=True)

        try:
            # Étape 1: Analyse audio
            logger.info("Pipeline Optimal - Étape 1/3: Analyse audio")
            args = [audio_file]
            args.extend(self.runner._build_base_args(cli_args))

            script_result = self.runner.run_script("main_audio_data", args)
            if script_result.success:
                # Lecture fichier audio analysis
                audio_json = self.runner.find_latest_output("output/audio_analysis/*.json", audio_file)
                if audio_json:
                    with open(audio_json, 'r', encoding='utf-8') as f:
                        audio_data = json.load(f)
                    result.add_result("audio_analysis", audio_data)
                    result.add_output_file(audio_json)
            else:
                result.add_error(script_result.error, "audio_analysis")
                logger.warning("Analyse audio échouée, continuation sans")

            # Étape 2: Transcription
            logger.info("Pipeline Optimal - Étape 2/3: Transcription")
            args = [audio_file, "--model", cli_args.model, "--language", cli_args.language]
            args.extend(self.runner._build_base_args(cli_args))

            script_result = self.runner.run_script("main_transcribe", args)
            if not script_result.success:
                result.success = False
                result.add_error(script_result.error, "transcription")
                return result

            cleanup_gpu_memory()

            # Lecture transcription
            transcription_file = self.runner.find_latest_output("output/transcriptions/*_raw.txt", audio_file)
            if not transcription_file:
                result.success = False
                result.add_error("Fichier transcription non trouvé")
                return result

            with open(transcription_file, 'r', encoding='utf-8') as f:
                transcription_text = f.read()

            result.add_result("transcription", {"text": transcription_text, "file": transcription_file})
            result.add_output_file(transcription_file)

            # Étape 3: Extraction
            logger.info("Pipeline Optimal - Étape 3/3: Extraction")
            args = [transcription_file]
            args.extend(self.runner._build_base_args(cli_args))

            script_result = self.runner.run_script("main_extract", args)
            if not script_result.success:
                result.success = False
                result.add_error(script_result.error, "extraction")
                return result

            # Lecture extraction
            extraction_json = self.runner.find_latest_output("output/extractions/*.json", audio_file)
            logger.info(f"EXTRACTION CHECK: extraction_json = {extraction_json}")
            if extraction_json:
                with open(extraction_json, 'r', encoding='utf-8') as f:
                    extraction_data = json.load(f)
                result.add_result("extraction", extraction_data)
                result.add_output_file(extraction_json)

            logger.info("Pipeline Optimal terminé")

        except Exception as e:
            logger.error(f"Erreur pipeline optimal: {e}")
            result.success = False
            result.add_error(str(e), "pipeline_optimal")

        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result

    def run_full(self, audio_file: str, cli_args: CLIArgs) -> PipelineResult:
        """Pipeline full : transcribe → correction → extract + audio_analysis → reco."""
        start_time = datetime.now()
        result = PipelineResult(success=True)

        try:
            # Étape 1: Analyse audio
            logger.info("Pipeline Full - Étape 1/5: Analyse audio")
            args = [audio_file]
            args.extend(self.runner._build_base_args(cli_args))

            script_result = self.runner.run_script("main_audio_data", args)
            if script_result.success:
                # Lecture fichier audio analysis
                audio_json = self.runner.find_latest_output("output/audio_analysis/*.json", audio_file)
                if audio_json:
                    with open(audio_json, 'r', encoding='utf-8') as f:
                        audio_data = json.load(f)
                    result.add_result("audio_analysis", audio_data)
                    result.add_output_file(audio_json)
            else:
                result.add_error(script_result.error, "audio_analysis")
                logger.warning("Analyse audio échouée, continuation sans")

            # Étape 2: Transcription
            logger.info("Pipeline Full - Étape 2/5: Transcription")
            args = [audio_file, "--model", cli_args.model, "--language", cli_args.language]
            args.extend(self.runner._build_base_args(cli_args))

            script_result = self.runner.run_script("main_transcribe", args)
            if not script_result.success:
                result.success = False
                result.add_error(script_result.error, "transcription")
                return result

            result.add_result("transcription", {"output": script_result.output})
            cleanup_gpu_memory()

            # Récupération transcription
            transcription_file = self.runner.find_latest_output("output/transcriptions/*_raw.txt", audio_file)
            if not transcription_file:
                result.success = False
                result.add_error("Fichier transcription non trouvé")
                return result

            result.add_output_file(transcription_file)

            # Étape 3: Correction
            logger.info("Pipeline Full - Étape 3/5: Correction")
            args = [transcription_file, "--correction-only"]
            args.extend(self.runner._build_base_args(cli_args))

            script_result = self.runner.run_script("main_corrector", args)
            if script_result.success:
                result.add_result("correction", {"output": script_result.output})
            else:
                result.add_error(script_result.error, "correction")
                logger.warning("Correction échouée, continuation sans")

            cleanup_gpu_memory()

            # Étape 4: Extraction
            logger.info("Pipeline Full - Étape 4/5: Extraction")

            corrected_file = self.runner.find_latest_output("output/ground_truth/*_ground_truth_*.txt", audio_file)
            input_file = corrected_file if corrected_file else transcription_file

            args = [input_file]
            args.extend(self.runner._build_base_args(cli_args))

            logger.info(f"Extraction utilise: {'le ground truth' if corrected_file else 'la transcription brute'}")

            script_result = self.runner.run_script("main_extract", args)
            if not script_result.success:
                result.success = False
                result.add_error(script_result.error, "extraction")
                return result

            # Lecture extraction
            extraction_json = self.runner.find_latest_output("output/extractions/*.json", audio_file)
            logger.info(f"EXTRACTION CHECK: extraction_json = {extraction_json}")
            if extraction_json:
                with open(extraction_json, 'r', encoding='utf-8') as f:
                    extraction_data = json.load(f)
                result.add_result("extraction", extraction_data)
                result.add_output_file(extraction_json)

            # Étape 5: Recommandations
            logger.info("Pipeline Full - Étape 5/5: Recommandations")
            args = [input_file]

            logger.info(f"Recommandation utilise: {'le ground truth' if corrected_file else 'la transcription brute'}")

            # Enrichissement avec audio analysis
            audio_json = self.runner.find_latest_output("output/audio_analysis/*.json", audio_file)
            if audio_json:
                args.extend(["--audio-data", audio_json])

            args.extend(self.runner._build_base_args(cli_args))

            script_result = self.runner.run_script("main_reco", args)
            if script_result.success:
                reco_json = self.runner.find_latest_output("output/recommendations/*.json", audio_file)
                if reco_json:
                    with open(reco_json, 'r', encoding='utf-8') as f:
                        reco_data = json.load(f)
                        result.add_result("recommendations", reco_data)
                        result.add_output_file(reco_json)
            else:
                result.add_error(script_result.error, "recommendations")
                logger.warning("Recommandations échouées")

            cleanup_gpu_memory()
            logger.info("Pipeline Full terminé")

        except Exception as e:
            logger.error(f"Erreur pipeline full: {e}")
            result.success = False
            result.add_error(str(e), "pipeline_full")

        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result

# === ORCHESTRATEUR ===

def run_pipeline(audio_file: str, cli_args: CLIArgs) -> PipelineResult:
    """Orchestrateur principal épuré."""

    # Validation fichier audio
    audio_path = Path(audio_file)
    if not audio_path.exists():
        result = PipelineResult(success=False)
        result.add_error(f"Fichier audio non trouvé: {audio_file}")
        return result

    # Vérification format audio
    audio_formats = ['.mp3', '.wav', '.mp4', '.m4a', '.webm']
    if audio_path.suffix.lower() not in audio_formats:
        result = PipelineResult(success=False)
        result.add_error(f"Format non supporté: {audio_path.suffix}. Formats acceptés: {audio_formats}")
        return result

    # Initialisation
    runner = ScriptRunner()
    pipeline_runner = SummoraPipelineRunner(runner)

    # Routage selon le mode
    if cli_args.mode == "light":
        return pipeline_runner.run_light(audio_file, cli_args)
    elif cli_args.mode == "optimal":
        return pipeline_runner.run_optimal(audio_file, cli_args)
    elif cli_args.mode == "full":
        return pipeline_runner.run_full(audio_file, cli_args)
    else:
        result = PipelineResult(success=False)
        logger.info(f"ce que donne result {result[500:]}")
        result.add_error(f"Mode inconnu: {cli_args.mode}")
        return result

# === INTERFACES SIMPLIFIÉES ===

def transcribe_audio_file(audio_path: str, model: str = "base", **kwargs) -> Dict:
    """Interface simple transcription."""
    cli_args = CLIArgs(input_file=audio_path, mode="light", model=model, quiet=True)
    result = run_pipeline(audio_path, cli_args)
    return result.to_dict()

def analyze_meeting_optimal(audio_path: str, model: str = "base", **kwargs) -> Dict:
    """Interface pipeline optimal."""
    cli_args = CLIArgs(input_file=audio_path, mode="optimal", model=model, quiet=True)
    result = run_pipeline(audio_path, cli_args)
    logger.info(f"analyze optimal retourne : {str(result)[:500]}...")
    return result.to_dict()

def analyze_meeting_full(audio_path: str, model: str = "base", **kwargs) -> Dict:
    """Interface pipeline complet."""
    cli_args = CLIArgs(input_file=audio_path, mode="full", model=model, quiet=True)
    result = run_pipeline(audio_path, cli_args)
    logger.info(f"analyze full retourne : {str(result)[:500]}...")
    return result.to_dict()

# === POINT D'ENTRÉE ===

def main():
    """Point d'entrée principal."""
    cli_args = parse_cli()

    # Configuration logging
    if cli_args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif cli_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validation input
    if not Path(cli_args.input_file).exists():
        logger.error(f"Fichier non trouvé: {cli_args.input_file}")
        return 1

    try:
        # Exécution pipeline
        pipeline_result = run_pipeline(cli_args.input_file, cli_args)

        # Affichage résultats
        if pipeline_result.success:
            logger.info(f"Pipeline {cli_args.mode} réussi")
            logger.info(f"Temps: {pipeline_result.execution_time:.2f}s")
            logger.info(f"Fichiers créés: {len(pipeline_result.outputs_created)}")

            if cli_args.verbose:
                for output in pipeline_result.outputs_created:
                    logger.info(f"  • {Path(output).name}")
        else:
            logger.error(f"Pipeline {cli_args.mode} échoué")
            for error in pipeline_result.errors:
                logger.error(f"  • {error}")
            return 1

        # Sauvegarde rapport
        pipeline_result.save_report()
        return 0

    except KeyboardInterrupt:
        logger.info("Pipeline interrompu")
        return 1
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        if cli_args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
