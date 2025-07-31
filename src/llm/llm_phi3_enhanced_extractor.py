"""
Extracteur LLM Phi3-mini-128k enrichi par YAKE
Fallback local intelligent pour le pipeline cascade
"""
import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Path setup pour imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Imports Summora
from src.meeting.extractor import MeetingContentExtractor, MeetingExtractionConfig
from src.core.business_vocabulary import BUSINESS_KEYWORDS

# Imports ML (avec gestion d'erreur)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers/torch non disponibles - Phi3 désactivé")

logger = logging.getLogger(__name__)

class Phi3EnhancedExtractor:
    """
    Extracteur Phi3-mini-128k enrichi par YAKE.
    Fallback local avec quantification 4-bit.
    """

    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers et torch requis pour Phi3Enhanced")

        self.model = None
        self.tokenizer = None
        self.model_name = "microsoft/Phi-3-mini-128k-instruct"  # Version 128k !
        self.max_context_tokens = 16000  # Limite sécurisée (vs 128k théorique)

        # Extracteur YAKE pour preprocessing
        yake_config = MeetingExtractionConfig(
            use_business_vocabulary=True,
            use_enhanced_stopwords=True,
            extract_actions=True,
            extract_decisions=True
        )
        self.yake_extractor = MeetingContentExtractor(yake_config)

        self._load_model()
        logger.info("🧠 Phi3 Enhanced by YAKE initialisé")

    def _load_model(self):
        """Charge Phi3-mini-128k avec quantification 4-bit."""
        try:
            logger.info(f"🔄 Chargement {self.model_name} quantifié...")

            # Configuration quantification
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Chargement modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=False,
                attn_implementation='eager'
            )

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=False
            )

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"✅ Phi3-128k quantifié 4-bit chargé - GPU: {gpu_memory:.2f}GB")
            else:
                logger.info("✅ Phi3-128k quantifié 4-bit chargé - CPU")

        except Exception as e:
            logger.error(f"❌ Erreur chargement Phi3: {e}")
            raise

    def _extract_yake_context(self, transcription: str) -> Dict:
        """Extrait le contexte YAKE pour guider Phi3 (même que Qwen Enhanced)."""
        try:
            yake_results = self.yake_extractor.extract_meeting_content(transcription)

            context = {
                "business_topics": [],
                "action_indicators": [],
                "decision_indicators": [],
                "business_density": 0
            }

            # Topics business détectés par YAKE
            if 'topics' in yake_results and yake_results['topics'].get('topics'):
                business_topics = [
                    topic['keyword'] for topic in yake_results['topics']['topics']
                    if topic.get('is_business', False)
                ]
                context["business_topics"] = business_topics[:5]  # Top 5 pour Phi3

            # Actions détectées
            if 'actions' in yake_results and yake_results['actions'].get('actions'):
                context["action_indicators"] = [
                    action['action'][:80] for action in yake_results['actions']['actions'][:2]
                ]

            # Décisions détectées
            if 'decisions' in yake_results and yake_results['decisions'].get('decisions'):
                context["decision_indicators"] = [
                    decision['decision'][:80] for decision in yake_results['decisions']['decisions'][:2]
                ]

            # Densité business
            if 'topics' in yake_results:
                total_topics = len(yake_results['topics'].get('topics', []))
                business_topics_count = yake_results['topics'].get('business_topics', 0)
                context["business_density"] = (business_topics_count / total_topics * 100) if total_topics > 0 else 0

            logger.info(f"📊 Contexte YAKE: {len(context['business_topics'])} topics business")
            return context

        except Exception as e:
            logger.warning(f"⚠️ Erreur YAKE preprocessing: {e}")
            return {"business_topics": [], "action_indicators": [], "decision_indicators": [], "business_density": 0}

    def _build_phi3_prompt(self, transcription: str, yake_context: Dict) -> str:
        """
        Construit un prompt optimisé Phi3 avec contexte YAKE.
        Version courte pour éviter les hallucinations.
        """
        # Vocabulaire business réduit
        business_categories = {
            'actions': BUSINESS_KEYWORDS.get('actions', [])[:8],
            'decisions': BUSINESS_KEYWORDS.get('decisions', [])[:8],
        }

        # Transcription tronquée si trop longue
        max_chars = 8000  # Limite pour Phi3
        if len(transcription) > max_chars:
            transcription_short = transcription[:max_chars] + "..."
            logger.info(f"📝 Transcription tronquée: {len(transcription)} → {max_chars} chars")
        else:
            transcription_short = transcription

        prompt = f"""Tu es expert en analyse de réunions d'entreprise.

CONTEXTE BUSINESS détecté par système YAKE:
- Topics: {', '.join(yake_context['business_topics'][:3])}
- Actions trouvées: {len(yake_context['action_indicators'])}
- Décisions trouvées: {len(yake_context['decision_indicators'])}

VOCABULAIRE PRIORITAIRE:
- Actions: {', '.join(business_categories['actions'][:5])}
- Décisions: {', '.join(business_categories['decisions'][:5])}

Analyse cette réunion et extrais:
- 2-3 topics principaux SEULEMENT
- 5 points clés MAXIMUM

Transcription: {transcription_short}

Réponds UNIQUEMENT en JSON valide:
{{"topics_principaux": ["topic1", "topic2"], "points_a_retenir": ["point1", "point2", "point3", "point4", "point5"]}}"""

        return prompt

    def extract_meeting_insights_phi3(self, transcription: str) -> Dict:
        """
        Extraction avec Phi3-128k enrichi par YAKE.
        Optimisé pour éviter les hallucinations.
        """
        logger.info(f"🧠 Extraction Phi3 Enhanced - Input: {len(transcription)} chars")
        start_time = time.time()

        try:
            # 1. Preprocessing YAKE
            yake_context = self._extract_yake_context(transcription)

            # 2. Prompt optimisé
            prompt = self._build_phi3_prompt(transcription, yake_context)

            # 3. Tokenisation avec gestion des limites
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_tokens
            )

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # 4. Génération
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Limité pour 2-3 topics + 5 points
                    temperature=0.1,    # Température basse = moins d'hallucinations
                    do_sample=False,    # Déterministe
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False     # Économie mémoire
                )

            # 5. Décodage
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()

            duration = time.time() - start_time
            logger.info(f"✅ Phi3 Enhanced terminé en {duration:.2f}s")

            # 6. Parse JSON
            try:
                # Nettoyage réponse (peut contenir du texte après JSON)
                if response.startswith('{'):
                    # Trouve la fin du JSON
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(response):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break

                    if json_end > 0:
                        response = response[:json_end]

                parsed_result = json.loads(response)

                return {
                    "success": True,
                    "data": parsed_result,
                    "yake_context": yake_context,
                    "enhancement_used": True,
                    "metrics": {
                        "duration": duration,
                        "model": self.model_name,
                        "input_chars": len(transcription),
                        "context_tokens": inputs['input_ids'].shape[1]
                    }
                }

            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur JSON Phi3: {e}")
                logger.error(f"📋 Réponse brute: {response}")
                return {
                    "success": False,
                    "error": "json_parsing_failed",
                    "raw_response": response,
                    "yake_context": yake_context
                }

        except Exception as e:
            logger.error(f"❌ Erreur Phi3 Enhanced: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"duration": time.time() - start_time}
            }

# Fonction utilitaire
def extract_with_phi3_enhanced(transcription: str) -> Dict:
    """
    Fonction utilitaire pour extraction avec Phi3 Enhanced.
    Compatible avec le pipeline cascade.
    """
    try:
        extractor = Phi3EnhancedExtractor()

        # Extraction
        extraction_result = extractor.extract_meeting_insights_phi3(transcription)

        if not extraction_result["success"]:
            return {
                "method": "phi3_enhanced_by_yake",
                "success": False,
                "error": extraction_result["error"],
                "yake_context": extraction_result.get("yake_context", {})
            }

        # Format compatible pipeline
        return {
            "method": "phi3_enhanced_by_yake",
            "success": True,
            "extraction": extraction_result["data"],
            "yake_context": extraction_result["yake_context"],
            "quality_scores": {"score_global": 75},  # Score conservateur pour SLM
            "metrics": extraction_result["metrics"],
            "enhancement_used": True
        }

    except ImportError:
        return {
            "method": "phi3_enhanced_by_yake",
            "success": False,
            "error": "transformers_not_available"
        }
    except Exception as e:
        return {
            "method": "phi3_enhanced_by_yake",
            "success": False,
            "error": str(e)
        }

# Test
if __name__ == "__main__":
    test_text = """
    Réunion budget 2025. Décision d'augmenter le budget marketing de 20%.
    Action pour Jean : préparer le plan détaillé d'ici vendredi.
    Objectif : lancement campagne en mars 2025.
    """

    if TRANSFORMERS_AVAILABLE:
        result = extract_with_phi3_enhanced(test_text)
        print("🧪 Test Phi3 Enhanced:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("❌ Transformers non disponible - test impossible")
