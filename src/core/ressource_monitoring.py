"""
Monitoring des ressources pour Summora - Module spécialisé
Responsabilité unique : tracking RAM, GPU, tokens, coûts
Localisation : src/utils/ressource_monitoring.py
"""
import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

@dataclass
class RessourceMetrics:
    """Métriques de ressources avec tracemalloc."""
    # Memory metrics (RAM + GPU)
    memory_start_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_available: bool = False

    # Token usage (LLM APIs)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tokens_by_model: Dict[str, Dict[str, int]] = None

    # Cost tracking
    estimated_cost_eur: float = 0.0
    cost_by_api: Dict[str, float] = None

    # Performance metrics
    avg_tokens_per_second: float = 0.0
    total_api_calls: int = 0
    failed_api_calls: int = 0

    # Tracemalloc internal
    _tracemalloc_started: bool = False

    def __post_init__(self):
        if self.tokens_by_model is None:
            self.tokens_by_model = {}
        if self.cost_by_api is None:
            self.cost_by_api = {}

        # Démarrage tracemalloc si pas déjà fait
        self.start_memory_tracking()

    def start_memory_tracking(self):
        """Démarre le tracking mémoire avec tracemalloc"""
        import tracemalloc
        import psutil

        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started = True

        # Baseline mémoire initiale
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_start_mb = current_memory
        self.memory_current_mb = current_memory

        logger.debug(f"📊 Memory tracking démarré: {current_memory:.1f}MB baseline")

    def update_memory_usage(self):
        """Met à jour les métriques mémoire (RAM + GPU)"""
        import tracemalloc
        import psutil

        # RAM via psutil (plus fiable que tracemalloc pour le total)
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024
            self.memory_current_mb = current_memory

            # Peak via tracemalloc si disponible
            if tracemalloc.is_tracing():
                current_trace, peak_trace = tracemalloc.get_traced_memory()
                self.memory_peak_mb = max(self.memory_peak_mb, peak_trace / 1024 / 1024)
            else:
                self.memory_peak_mb = max(self.memory_peak_mb, current_memory)

        except Exception as e:
            logger.debug(f"Erreur memory tracking: {e}")

        # GPU si disponible
        self._update_gpu_usage()

    def _update_gpu_usage(self):
        """Met à jour l'usage GPU (PyTorch + nvidia-ml-py)"""
        # PyTorch GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_memory_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
                self.gpu_memory_peak_mb = max(
                    self.gpu_memory_peak_mb,
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                )
        except ImportError:
            pass

        # GPU utilization via nvidia-ml-py
        try:
            import pynvml
            if not hasattr(self, '_pynvml_initialized'):
                pynvml.nvmlInit()
                self._pynvml_initialized = True

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_utilization_percent = info.gpu

        except Exception:
            self.gpu_utilization_percent = 0.0

    def add_api_usage(self, model: str, input_tokens: int, output_tokens: int, cost: float = 0.0):
        """Ajoute l'usage d'un modèle/API + update memory."""
        if model not in self.tokens_by_model:
            self.tokens_by_model[model] = {"input": 0, "output": 0}

        self.tokens_by_model[model]["input"] += input_tokens
        self.tokens_by_model[model]["output"] += output_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_api_calls += 1

        if cost > 0:
            api_name = model.split("/")[0]  # ex: "openai/gpt-4" -> "openai"
            if api_name not in self.cost_by_api:
                self.cost_by_api[api_name] = 0.0
            self.cost_by_api[api_name] += cost
            self.estimated_cost_eur += cost

        # Update memory après chaque API call
        self.update_memory_usage()

    def calculate_performance_metrics(self, execution_time: float):
        """Calcule les métriques de performance finales."""
        if execution_time > 0 and self.total_output_tokens > 0:
            self.avg_tokens_per_second = self.total_output_tokens / execution_time

        # Final memory snapshot
        self.update_memory_usage()

    def stop_memory_tracking(self):
        """Arrête le tracking mémoire"""
        import tracemalloc

        if self._tracemalloc_started and tracemalloc.is_tracing():
            # Dernière mesure
            current_trace, peak_trace = tracemalloc.get_traced_memory()
            self.memory_peak_mb = peak_trace / 1024 / 1024

            tracemalloc.stop()
            logger.debug(f"📊 Memory tracking arrêté: peak {self.memory_peak_mb:.1f}MB")

    def get_summary(self) -> Dict:
        """Résumé des métriques pour affichage"""
        return {
            "memory": {
                "start_mb": round(self.memory_start_mb, 2),
                "current_mb": round(self.memory_current_mb, 2),
                "peak_mb": round(self.memory_peak_mb, 2),
                "delta_mb": round(self.memory_current_mb - self.memory_start_mb, 2)
            },
            "gpu": {
                "available": self.gpu_available,
                "memory_used_mb": round(self.gpu_memory_used_mb, 2),
                "memory_peak_mb": round(self.gpu_memory_peak_mb, 2),
                "utilization_percent": round(self.gpu_utilization_percent, 1)
            },
            "tokens": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "by_model": self.tokens_by_model
            },
            "cost": {
                "total_eur": round(self.estimated_cost_eur, 4),
                "by_api": {k: round(v, 4) for k, v in self.cost_by_api.items()}
            },
            "performance": {
                "avg_tokens_per_sec": round(self.avg_tokens_per_second, 2),
                "total_api_calls": self.total_api_calls,
                "failed_calls": self.failed_api_calls
            }
        }

# === Utilitaires de calcul de coût ===

def calculate_precise_llm_cost(prompt: str, response: str, tokenizer,
                               input_rate_eur_per_1m: float, output_rate_eur_per_1m: float) -> Dict:
    """
    Calcule le coût précis d'un appel LLM via tokenizer (méthode notebook).

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

# Tarifs OpenRouter courants (à ajuster selon tes observations)
OPENROUTER_RATES = {
    "qwen/qwen-72b-chat": {"input": 0.09, "output": 0.35},      # €/1M tokens
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
    return {"input": 0.1, "output": 0.4}  # Tarifs de référence Juillet 2025

def quick_cost_calculation(model: str, prompt: str, response: str, tokenizer) -> Dict:
    """Interface rapide pour calculer le coût (usage dans tes scripts LLM)."""
    rates = get_model_rates(model)
    return calculate_precise_llm_cost(prompt, response, tokenizer,
                                    rates["input"], rates["output"])

def cleanup_gpu_memory():
    """Nettoyage mémoire GPU - pattern notebook."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("🔥 GPU cache vidé")
    except ImportError:
        pass
