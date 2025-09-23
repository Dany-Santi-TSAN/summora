#!/usr/bin/env python3
"""
Summora V3 - Orchestrateur Principal Refactorisé
------------------------------------------------

🎙️ Speech In, Sense Out — une architecture modulaire pour transformer la parole en sens.

L’orchestrateur V3 coordonne les 5 modules spécialisés  :
- main_transcribe.py  → transcription audio (Whisper)
- main_extractor.py   → résumé extractif (10 bullets, 5 best topics) + résumé abstractif
- main_corrector.py   → transcription nettoyée (manager) + génération de ground truth
- main_audio_data.py  → analyse technique et qualitative de l'audio (parole, silences, ratios)
- main_reco.py        → amélioration du leadership et dynamique de réunion

Patterns consolidés et simplifiés pour plus de cohérence et de scalabilité :
- Cascade : enchaînement maîtrisé des étapes
- Configuration : flexibilité centralisée
- Pipeline : modularité et réutilisabilité
- Runner : orchestration traçable et robuste

Objectif : offrir une architecture claire, maintenable et extensible, qui fait de Summora un outil évolutif et fiable.
"""

import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict
import os
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

# === CONFIGURATION UNIFIÉE ===

@dataclass
class SummoraConfig:
    """Configuration unifiée - DRY principle."""
    # Core settings
    model: str = "medium"
    language: str = "fr"
    temperature: float = 0.0

    # Features flags
    enable_audio_data: bool = False
    enable_correction: bool = False
    enable_recommendation: bool = False
    enable_spot_check: bool = False

    @classmethod
    def load_env(cls) -> 'SummoraConfig':
        """Charge config depuis .env."""
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.debug(f"Env chargé: {env_path}")
        else:
            load_dotenv()
            logger.warning("Pas de .env trouvé")
        return cls()

# === CLI UNIFIÉ ===

@dataclass
class CLIArgs:
    """Arguments CLI simplifiés."""
    input_file: str
    mode: str = "light"  # light|optimal|full|transcribe|audio_data|correct|extract|reco
    model: str = "auto"
    language: str = "fr"
    with_audio_data: bool = False
    with_correction: bool = False
    with_reco: bool = False
    with_spot_check: bool = False
    verbose: bool = False
    quiet: bool = False

def parse_cli() -> CLIArgs:
    """Parse CLI avec patterns consolidés des 6 mains."""
    parser = argparse.ArgumentParser(
        description="Summora V3 - Orchestrateur Refactorisé",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py audio.mp3                         # Pipeline light (transcribe → extract)
  python main.py audio.mp3 --optimal               # Pipeline optimum (transcribe -> extract + audio_data)
  python main.py audio.mp3 --full                  # Pipeline complet
  python main.py audio.mp3 --transcribe-only       # Transcription seule
  python main.py transcription.txt --extract-only  # Extraction seule
  python main.py audio.mp3 --with-reco            # Optimal + recommandations
        """
    )

    parser.add_argument("input_file", help="Fichier d'entrée")

    # Modes pipeline (mutuellement exclusifs)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--transcribe-only", action="store_true")
    mode_group.add_argument("--extract-only", action="store_true")
    mode_group.add_argument("--reco-only", action="store_true")
    mode_group.add_argument("--correct-only", action="store_true")
    mode_group.add_argument("--audio-data-only", action="store_true")
    mode_group.add_argument("--optimal", "-opt", action="store_true",help="Pipeline avec résumé et analyse audio")
    mode_group.add_argument("--full", "-f", action="store_true", help="Pipeline complet")

    # Options bonus
    parser.add_argument("--with-audio-data", action="store_true")
    parser.add_argument("--with-correction", action="store_true")
    parser.add_argument("--with-reco", action="store_true")
    parser.add_argument("--with-spot-check", action="store_true")

    # Config
    parser.add_argument("--model", default="auto", choices=["auto","base","small","medium","large"])
    parser.add_argument("--language", default="fr")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()

    # Détermination du mode
    if args.transcribe_only:
        mode = "transcribe"
    elif args.extract_only:
        mode = "extract"
    elif args.reco_only:
        mode = "reco"
    elif args.correct_only:
        mode = "correct"
    elif args.audio_data_only:
        mode = "visual"
    elif args.optimal:
        mode ='optimal'
    elif args.full:
        mode = "full"
    else:
        mode = "light"

    return CLIArgs(
        input_file=args.input_file,
        mode=mode,
        model=args.model,
        language=args.language,
        with_audio_data=args.with_audio_data,
        with_correction=args.with_correction,
        with_reco=args.with_reco,
        with_spot_check=args.with_spot_check,
        verbose=args.verbose,
        quiet=args.quiet
    )

# === RUNNER UNIFIÉ ===

class ScriptRunner:
    """Runner unifié pour exécuter les scripts - Pattern des 5 modules."""

    def run_script(self, script_name: str, args: List[str]) -> ScriptResult:
        """Exécute un script avec gestion d'erreurs unifiée."""
        script_path = Path("scripts") / f"{script_name}.py"

        if not script_path.exists():
            return ScriptResult(
                success=False
                ,error=f"Script non trouvé: {script_path}"
                ,script_name=script_name
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
                    success=True
                    ,output=result.stdout
                    ,script_name=script_name
                    ,execution_time=execution_time
                )
            else:
                logger.error(f"Échec {script_name}")
                return ScriptResult(
                    success=False
                    ,error=result.stderr
                    ,output=result.stdout
                    ,script_name=script_name
                    ,execution_time=execution_time
                )

        except Exception as e:
            logger.error(f"Exception {script_name}: {e}")
            return ScriptResult(
                success=False
                ,error=str(e)
                ,script_name=script_name
            )

    def find_latest_output(self, pattern: str) -> Optional[str]:
        """Trouve le fichier le plus récent - Pattern main_transcribe."""
        import glob
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=lambda f: Path(f).stat().st_mtime)
            logger.info(f"Fichier détecté: {Path(latest).name}")
            return latest
        return None

# === PIPELINE UNIFIÉ ===

class SummoraPipeline:
    """Pipeline unifié consolidant les patterns des 6 mains."""

    def __init__(self, config: SummoraConfig, runner: ScriptRunner):
        self.config = config
        self.runner = runner

    def _build_base_args(self, cli_args: CLIArgs) -> List[str]:
        """Args de base pour tous les scripts."""
        args = []
        if cli_args.verbose:
            args.append("--verbose")
        if cli_args.quiet:
            args.append("--quiet")
        return args

    def transcribe(self, input_file: str, cli_args: CLIArgs) -> ScriptResult:
        """Transcription avec fallback - Pattern main_transcribe."""
        args = [input_file]

        # Configuration modèle
        model = cli_args.model if cli_args.model != "auto" else self.config.model
        if model != "auto":
            args.extend(["--model", model])

        if cli_args.language != "fr":
            args.extend(["--language", cli_args.language])

        args.extend(self._build_base_args(cli_args))

        result = self.runner.run_script("main_transcribe", args)
        cleanup_gpu_memory()  # Pattern ressources
        return result

    def extract(self, input_file: str, cli_args: CLIArgs) -> ScriptResult:
        """Extraction cascade - Pattern main_extract."""
        args = [input_file, "--with-eval"]

        if cli_args.with_spot_check or self.config.enable_spot_check:
            args.append("--enable-spot-check")

        args.extend(self._build_base_args(cli_args))
        return self.runner.run_script("main_extract", args)

    def recommend(self, input_file: str, audio_analysis_file: Optional[str], cli_args: CLIArgs) -> ScriptResult:
        """Recommandations RAG - Pattern main_reco."""
        args = [input_file]

        if audio_analysis_file:
            args.extend(["--with-visual", audio_analysis_file])

        if cli_args.mode == "reco":
            args.append("--reco-only")

        if cli_args.with_spot_check or self.config.enable_spot_check:
            args.append("--enable-spot-check")

        args.extend(self._build_base_args(cli_args))

        result = self.runner.run_script("main_reco", args)
        cleanup_gpu_memory()
        return result

    def correct(self, input_file: str, cli_args: CLIArgs) -> ScriptResult:
        """Correction transcription - Pattern main_corrector."""
        args = [input_file]

        if cli_args.mode == "correct":
            args.append("--correction-only")

        if cli_args.with_spot_check or self.config.enable_spot_check:
            args.append("--enable-spot-check")

        args.extend(self._build_base_args(cli_args))

        result = self.runner.run_script("main_corrector", args)
        cleanup_gpu_memory()
        return result

    def analyze_audio(self, input_file: str, cli_args: CLIArgs) -> ScriptResult:
        """Analyse audio - Pattern main_audio_data."""
        args = [input_file]

        if self.config.enable_audio_data:
            args.append("--plots")

        args.extend(self._build_base_args(cli_args))
        return self.runner.run_script("main_audio_data", args)

# === ORCHESTRATEUR PRINCIPAL ===

def detect_file_type(file_path: Path) -> str:
    """Détection du type de fichier avant de lancer la pipeline"""
    file_name = file_path.name.lower()
    file_ext = file_path.suffix.lower()
    list_audio_suffix = ['.mp3']

    if file_ext in list_audio_suffix:
        return "audio"

    # Condition des formats spécifiques à Summora
    if file_ext == '.txt':
        if '_raw' in file_name and 'transcription' in file_name:
            return 'transcription_raw'
        elif 'transcription' in file_name:
            return 'transcription_formatted'
        elif 'ground_truth' in file_name:
            return 'ground_truth'
        return 'text_generic'

    if file_ext == '.json':
        if 'transcription' in file_name:
            return 'transcription_metadata'
        elif 'extraction' in file_name:
            return 'extraction'
        elif 'recommendation' in file_name:
            return 'recommendation'
        return 'json_generic'

    return 'unknown'

def run_pipeline(input_file: str, cli_args: CLIArgs, config: SummoraConfig) -> PipelineResult:
    """
    Orchestrateur principal - Patterns consolidés des 5 modules.
    Architecture: Validation → Détection type → Exécution mode → Chaînage
    """
    start_time = datetime.now()
    result = PipelineResult(success=True)

    # Initialisation
    runner = ScriptRunner()
    pipeline = SummoraPipeline(config, runner)

    try:
        # Validation fichier
        input_path = Path(input_file)
        if not input_path.exists():
            result.success = False
            result.add_error(f"Fichier non trouvé: {input_file}")
            return result

        file_type = detect_file_type(input_path)

        transcription_file = None
        extraction_data = None
        audio_analysis_file = None

        logger.info(f"Fichier détecté: {'audio' if file_type == 'audio' else 'transcription' if file_type == 'transcription' else 'extraction'}")

        # === MODES INDIVIDUELS ===
        if cli_args.mode == "transcribe":
            if not file_type == 'audio':
                result.success = False
                result.add_error("Mode transcribe nécessite un fichier audio")
                return result

            script_result = pipeline.transcribe(input_file, cli_args)
            if script_result.success:
                result.add_result("transcription", {"output": script_result.output})
                transcription_file = runner.find_latest_output("output/transcriptions/*_raw.txt")
                if transcription_file:
                    result.add_output_file(transcription_file)
            else:
                result.success = False
                result.add_error(script_result.error, "transcription")

        elif cli_args.mode == "extract":
            script_result = pipeline.extract(input_file, cli_args)
            if script_result.success:
                result.add_result("extraction", {"output": script_result.output})
                extraction_json = runner.find_latest_output("output/extractions/*.json")
                if extraction_json:
                    result.add_output_file(extraction_json)
            else:
                result.success = False
                result.add_error(script_result.error, "extraction")

        elif cli_args.mode == "reco":
            script_result = pipeline.recommend(input_file, cli_args)
            if script_result.success:
                result.add_result("recommendations", {"output": script_result.output})
                reco_json = runner.find_latest_output("output/recommendations/*.json")
                if reco_json:
                    result.add_output_file(reco_json)
            else:
                result.success = False
                result.add_error(script_result.error, "recommendations")

        elif cli_args.mode == "correct":
            script_result = pipeline.correct(input_file, cli_args)
            if script_result.success:
                result.add_result("correction", {"output": script_result.output})
            else:
                result.success = False
                result.add_error(script_result.error, "correction")

        elif cli_args.mode == "audio_data":
            if not file_type == 'audio':
                result.success = False
                result.add_error("Mode audio_data nécessite un fichier audio")
                return result

            script_result = pipeline.analyze_audio(input_file, cli_args)
            if script_result.success:
                result.add_result("audio_analysis", {"output": script_result.output})
                audio_json = runner.find_latest_output("output/audio_analysis/*.json")
                if audio_json:
                    result.add_output_file(audio_json)
            else:
                result.success = False
                result.add_error(script_result.error, "audio_analysis")

        # === MODES PIPELINE ===
        else:
            # Configuration features pour mode full
            if cli_args.mode == "full":
                cli_args.with_audio_data = True
                cli_args.with_reco = True
                cli_args.with_correction = True
                logger.info("Mode: Pipeline complet")
            else:
                logger.info("Mode: Pipeline light")

            # Étape 1: Analyse audio (optionnelle)
            if (cli_args.with_audio_data or config.enable_audio_data) and file_type == 'audio':
                logger.info("Étape: Analyse audio")
                script_result = pipeline.analyze_audio(input_file, cli_args)
                if script_result.success:
                    result.add_result("audio_analysis", {"output": script_result.output})
                else:
                    result.add_error(script_result.error, "audio_analysis")
                    logger.warning("Analyse audio échouée, continuation")

            # Étape 2: Transcription (si fichier audio)
            if file_type == 'audio':
                logger.info("Étape: Transcription")
                script_result = pipeline.transcribe(input_file, cli_args)
                if script_result.success:
                    result.add_result("transcription", {"output": script_result.output})
                    transcription_file = runner.find_latest_output("output/transcriptions/*_raw.txt")
                    if transcription_file:
                        result.add_output_file(transcription_file)
                    else:
                        result.add_error("Fichier brut 'transcription_raw' non trouvé")
                else:
                    result.success = False
                    result.add_error(script_result.error, "transcription")
                    return result

            # Étape 3: Correction (optionnelle)
            if (cli_args.with_correction or config.enable_correction) and not extraction_data:
                logger.info("Étape: Correction")
                script_result = pipeline.correct(transcription_file, cli_args)
                if script_result.success:
                    result.add_result("correction", {"output": script_result.output})
                else:
                    result.add_error(script_result.error, "correction")
                    logger.warning("Correction échouée, continuation")

            # Étape 4: Extraction
            if not extraction_data:
                logger.info("Étape: Extraction")
                script_result = pipeline.extract(transcription_file, cli_args)
                if script_result.success:
                    result.add_result("extraction", {"output": script_result.output})
                    extraction_json = runner.find_latest_output("output/extractions/*.json")
                    if extraction_json:
                        extraction_data = extraction_json
                        result.add_output_file(extraction_json)
                    else:
                        result.add_error("JSON extraction non trouvé")
                else:
                    result.success = False
                    result.add_error(script_result.error, "extraction")
                    return result

            # Étape 5: Recommandations (optionnelles)
            if cli_args.with_reco or config.enable_recommendation:
                logger.info("Étape: Recommandations")

                # Enrichissement avec audio analysis
                if result.results.get('audio_analysis'):
                    audio_json = runner.find_latest_output("output/audio_analysis/*.json")
                    audio_analysis_file = audio_json

                script_result = pipeline.recommend(extraction_data, audio_analysis_file, cli_args) # extraction_data = None ou extraction_json
                if script_result.success:
                    result.add_result("recommendations", {"output": script_result.output})
                    reco_json = runner.find_latest_output("output/recommendations/*.json")
                    if reco_json:
                        result.add_output_file(reco_json)
                else:
                    result.add_error(script_result.error, "recommendations")
                    logger.warning("Recommandations échouées")
            else:
                logger.info("Recommandations skippées (utilisez --with-reco)")
                result.add_result("recommendations_skipped", {"reason": "not_requested"})

        logger.info("Pipeline terminé")

    except Exception as e:
        logger.error(f"Erreur pipeline: {e}")
        result.success = False
        result.add_error(str(e), "pipeline_fatal")

    result.execution_time = (datetime.now() - start_time).total_seconds()
    return result

# === INTERFACES SIMPLIFIÉES ===

def transcribe_audio_file(audio_path: str, model: str = "medium", **kwargs) -> Dict:
    """Interface simple transcription."""
    cli_args = CLIArgs(input_file=audio_path, mode="transcribe", model=model, quiet=True)
    config = SummoraConfig(model=model)
    result = run_pipeline(audio_path, cli_args, config)
    return result.to_dict()

def analyze_meeting_optimal(audio_path: str, **kwargs) -> Dict:
    """Interface pipeline optimal"""

    mode = "optimal"

    cli_args = CLIArgs(
        input_file=audio_path
        ,mode=mode
        ,with_audio_data=True
        ,quiet=True
    )
    config = SummoraConfig(
        enable_audio_data=True
    )
    result = run_pipeline(audio_path, cli_args, config)
    return result.to_dict()

def analyze_meeting_full(audio_path: str, include_all: bool = False, **kwargs) -> Dict:
    """Interface pipeline complet."""

    mode = "full" if include_all else "extract"

    cli_args = CLIArgs(
        input_file=audio_path
        ,mode=mode
        ,with_reco=True
        ,quiet=True
    )
    config = SummoraConfig(
        enable_recommendation=True
        ,enable_audio_data=include_all
        ,enable_correction=include_all
        ,enable_spot_check=include_all
    )
    result = run_pipeline(audio_path, cli_args, config)
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
        # Chargement config
        config = SummoraConfig.load_env()

        # Exécution pipeline
        pipeline_result = run_pipeline(cli_args.input_file, cli_args, config)

        # Affichage résultats
        if pipeline_result.success:
            logger.info("Pipeline réussi")
            logger.info(f"Temps: {pipeline_result.execution_time:.2f}s")
            logger.info(f"Fichiers créés: {len(pipeline_result.outputs_created)}")

            if cli_args.verbose:
                for output in pipeline_result.outputs_created:
                    logger.info(f"  • {Path(output).name}")
        else:
            logger.error("Pipeline échoué")
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
