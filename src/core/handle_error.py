"""
Gestion d'erreurs pour Summora - Module spécialisé
Responsabilité unique : PipelineResult, ScriptResult, gestion erreurs
Localisation : src/utils/handle_error.py
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# Setup path pour imports Summora
sys.path.append(str(Path(__file__).parent.parent))

from src.core.ressource_monitoring import RessourceMetrics

logger = logging.getLogger(__name__)

@dataclass
class ScriptResult:
    """Résultat d'exécution de script"""
    success: bool
    output: str = ""
    error: str = ""
    script_name: str = ""
    execution_time: float = 0.0
    output_files: List[str] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []

@dataclass
class PipelineResult:
    """Résultat d'exécution du pipeline"""
    success: bool
    execution_time: float = 0.0
    results: Dict = None
    errors: List[str] = None
    outputs_created: List[str] = None
    steps_completed: List[str] = None

    # Monitoring des ressources
    resource_metrics: RessourceMetrics = None

    def __post_init__(self):
        if self.results is None:
            self.results = {}
        if self.errors is None:
            self.errors = []
        if self.outputs_created is None:
            self.outputs_created = []
        if self.steps_completed is None:
            self.steps_completed = []
        if self.resource_metrics is None:
            self.resource_metrics = RessourceMetrics()

    def add_error(self, error: str, step: str = ""):
        """Ajoute une erreur avec contexte."""
        error_msg = f"[{step}] {error}" if step else error
        self.errors.append(error_msg)
        logger.error(f"❌ {error_msg}")

    def add_result(self, step: str, data: Dict):
        """Ajoute un résultat d'étape."""
        self.results[step] = data
        self.steps_completed.append(step)
        logger.info(f"✅ {step} terminé")

    def add_output_file(self, filepath: str):
        """Ajoute un fichier créé."""
        if Path(filepath).exists():
            self.outputs_created.append(filepath)
            logger.info(f"📁 Fichier créé: {Path(filepath).name}")

    def record_api_usage(self, model: str, input_tokens: int, output_tokens: int, cost: float = 0.0):
        """Enregistre l'usage d'une API LLM."""
        self.resource_metrics.add_api_usage(model, input_tokens, output_tokens, cost)
        logger.debug(f"📊 API usage: {model} - {input_tokens}+{output_tokens} tokens, €{cost:.4f}")

    def record_api_usage_with_tokenizer(self, model: str, prompt: str, response: str,
                                      tokenizer, input_rate_eur_per_1m: float,
                                      output_rate_eur_per_1m: float):
        """
        Enregistre l'usage API avec calcul précis par tokenizer (méthode data scientist).

        Args:
            model: Nom du modèle (ex: "openrouter/qwen-72b")
            prompt: Texte d'entrée
            response: Réponse générée
            tokenizer: Tokenizer du modèle utilisé
            input_rate_eur_per_1m: Tarif input en €/1M tokens
            output_rate_eur_per_1m: Tarif output en €/1M tokens
        """
        from .ressource_monitoring import calculate_precise_llm_cost

        try:
            # Calcul précis via tokenizer (méthode notebook)
            cost_data = calculate_precise_llm_cost(
                prompt, response, tokenizer,
                input_rate_eur_per_1m, output_rate_eur_per_1m
            )

            # Enregistrement
            self.record_api_usage(
                model,
                cost_data["input_tokens"],
                cost_data["output_tokens"],
                cost_data["total_cost_eur"]
            )

            logger.debug(f"💰 Coût détaillé {model}: "
                        f"€{cost_data['input_cost_eur']:.6f} input + "
                        f"€{cost_data['output_cost_eur']:.6f} output = "
                        f"€{cost_data['total_cost_eur']:.6f}")

        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul tokenizer pour {model}: {e}")
            # Fallback estimation approximative
            estimated_input = len(prompt.split()) * 1.3  # ~1.3 tokens/mot
            estimated_output = len(response.split()) * 1.3
            estimated_cost = ((estimated_input + estimated_output) / 1_000_000) * 0.25  # Tarif moyen
            self.record_api_usage(model, int(estimated_input), int(estimated_output), estimated_cost)

    def record_failed_api_call(self):
        """Enregistre un échec d'API call."""
        self.resource_metrics.failed_api_calls += 1

    def finalize_metrics(self):
        """Finalise les métriques (à appeler à la fin du pipeline)."""
        self.resource_metrics.calculate_performance_metrics(self.execution_time)
        self.resource_metrics.stop_memory_tracking()

        # Cleanup GPU final (bonne pratique notebooks)
        self._cleanup_gpu_memory()

    def _cleanup_gpu_memory(self):
        """Nettoyage mémoire GPU - pattern notebook."""
        from .ressource_monitoring import cleanup_gpu_memory
        cleanup_gpu_memory()

    def to_dict(self) -> Dict:
        """Export pour analyse/debug avec métriques."""
        base_dict = asdict(self)
        # Ajout du résumé des métriques pour lisibilité
        base_dict["resource_summary"] = self.resource_metrics.get_summary()
        return base_dict

    def save_report(self, output_dir: str = "output/reports"):
        """Sauvegarde le rapport d'exécution avec métriques."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"pipeline_report_{timestamp}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Rapport sauvé: {report_file}")

    def print_resource_summary(self):
        """Affiche un résumé des ressources utilisées."""
        metrics = self.resource_metrics.get_summary()

        logger.info("📊 === RÉSUMÉ DES RESSOURCES ===")

        # Memory (RAM)
        memory = metrics["memory"]
        logger.info(f"🧠 RAM: {memory['peak_mb']:.1f}MB peak ({memory['delta_mb']:+.1f}MB delta)")
        logger.info(f"     Start: {memory['start_mb']:.1f}MB → Current: {memory['current_mb']:.1f}MB")

        # GPU
        if metrics["gpu"]["available"]:
            gpu = metrics["gpu"]
            logger.info(f"🔥 GPU: {gpu['memory_peak_mb']:.1f}MB peak, {gpu['utilization_percent']:.1f}% util")
        else:
            logger.info("🔥 GPU: Non disponible")

        # Tokens
        total_tokens = metrics["tokens"]["total_input"] + metrics["tokens"]["total_output"]
        if total_tokens > 0:
            logger.info(f"🪙 Tokens: {total_tokens:,} total "
                       f"({metrics['tokens']['total_input']:,} in + {metrics['tokens']['total_output']:,} out)")
            for model, usage in metrics["tokens"]["by_model"].items():
                logger.info(f"   • {model}: {usage['input']:,}+{usage['output']:,} tokens")

        # Coût
        if metrics["cost"]["total_eur"] > 0:
            logger.info(f"💰 Coût estimé: €{metrics['cost']['total_eur']:.4f}")
            for api, cost in metrics["cost"]["by_api"].items():
                logger.info(f"   • {api}: €{cost:.4f}")

        # Performance
        if metrics["performance"]["avg_tokens_per_sec"] > 0:
            logger.info(f"⚡ Performance: {metrics['performance']['avg_tokens_per_sec']:.1f} tokens/sec")

        # API calls
        if metrics["performance"]["total_api_calls"] > 0:
            success_rate = ((metrics["performance"]["total_api_calls"] - metrics["performance"]["failed_calls"]) /
                          metrics["performance"]["total_api_calls"] * 100)
            logger.info(f"📡 API Calls: {metrics['performance']['total_api_calls']} total, {success_rate:.1f}% success")

# === Utilitaires pour gestion d'erreurs ===

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
