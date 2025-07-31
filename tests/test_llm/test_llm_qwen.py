"""
Tests unitaires pour les modules LLM Qwen
"""
import pytest
import json
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Ajout du path pour imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm.llm_fallback_extractor import QwenExtractor, extract_with_qwen
from src.llm.llm_qwen_corrector import QwenCorrector, correct_transcription_for_download

class TestQwenExtractor:
    """Tests pour l'extracteur Qwen."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key pour tests."""
        return "test_api_key"

    @pytest.fixture
    def sample_transcription(self):
        """Transcription de test."""
        return """
        Réunion budget 2025. Présents : Marie, Jean, Paul.
        Décision d'augmenter le budget marketing de 20%.
        Action pour Jean : préparer le plan détaillé d'ici vendredi.
        Objectif : lancement campagne en mars 2025.
        """

    @pytest.fixture
    def mock_extraction_response(self):
        """Réponse simulée de l'API."""
        return {
            "topics_principaux": ["budget 2025", "marketing", "campagne", "planning", "objectifs"],
            "points_a_retenir": [
                "Augmentation budget marketing de 20%",
                "Jean responsable du plan détaillé",
                "Échéance vendredi pour le plan",
                "Lancement campagne prévu mars 2025",
                "Présents: Marie, Jean, Paul"
            ],
            "resume_abstractif": "Réunion de planification budget 2025 avec décision d'augmenter le marketing de 20% et attribution des responsabilités."
        }

    def test_extractor_initialization(self, mock_api_key):
        """Test d'initialisation de l'extracteur."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            extractor = QwenExtractor()
            assert extractor.extractor_model == "qwen/qwen3-235b-a22b-07-25:free"
            assert extractor.judge_model == "tngtech/deepseek-r1t-chimera:free"

    @patch('src.llm.llm_qwen_extractor.OpenAI')
    def test_extraction_success(self, mock_openai, mock_api_key, sample_transcription, mock_extraction_response):
        """Test d'extraction réussie."""
        # Mock de la réponse API
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_extraction_response)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            extractor = QwenExtractor()
            result = extractor.extract_meeting_insights(sample_transcription)

        assert result["success"] is True
        assert "data" in result
        assert result["data"]["topics_principaux"] == mock_extraction_response["topics_principaux"]
        assert len(result["data"]["points_a_retenir"]) == 5

    @patch('src.llm.llm_qwen_extractor.OpenAI')
    def test_extraction_json_parse_error(self, mock_openai, mock_api_key, sample_transcription):
        """Test d'erreur de parsing JSON."""
        # Mock réponse JSON invalide
        mock_choice = Mock()
        mock_choice.message.content = "Réponse non-JSON invalide"

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            extractor = QwenExtractor()
            result = extractor.extract_meeting_insights(sample_transcription)

        assert result["success"] is False
        assert result["error"] == "json_parsing_failed"

    @patch('src.llm.llm_qwen_extractor.OpenAI')
    def test_judge_evaluation(self, mock_openai, mock_api_key, sample_transcription, mock_extraction_response):
        """Test d'évaluation par le judge."""
        mock_judge_response = {
            "pertinence": 90,
            "completude": 85,
            "precision": 95,
            "score_global": 90,
            "justification": "Excellente extraction"
        }

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_judge_response)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            extractor = QwenExtractor()
            result = extractor.judge_extraction_quality(sample_transcription, mock_extraction_response)

        assert result["success"] is True
        assert result["scores"]["score_global"] == 90
        assert result["metrics"]["fallback_used"] is False

    def test_extract_with_qwen_function(self, mock_api_key, sample_transcription):
        """Test de la fonction utilitaire extract_with_qwen."""
        with patch('src.llm.llm_qwen_extractor.QwenExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor

            # Mock extraction
            mock_extractor.extract_meeting_insights.return_value = {
                "success": True,
                "data": {"topics_principaux": ["test"]},
                "metrics": {"duration": 1.5}
            }

            # Mock judge
            mock_extractor.judge_extraction_quality.return_value = {
                "success": True,
                "scores": {"score_global": 85},
                "metrics": {"duration": 0.5}
            }

            with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
                result = extract_with_qwen(sample_transcription)

            assert result["success"] is True
            assert result["method"] == "qwen_llm_extractor"
            assert result["metrics"]["total_duration"] == 2.0

class TestQwenCorrector:
    """Tests pour le correcteur Qwen."""

    @pytest.fixture
    def mock_api_key(self):
        return "test_api_key"

    @pytest.fixture
    def sample_transcription_brute(self):
        """Transcription brute avec erreurs."""
        return """
        euh bon alors on commence la réunion euh marie tu es là oui oui je suis là parfait
        bon alors donc on va parler du budget 2025 euh jean tu as préparé les chiffres oui j'ai
        tout préparé alors euh on a une augmentation de 20% sur le marketing
        """

    @pytest.fixture
    def sample_transcription_corrigee(self):
        """Transcription corrigée attendue."""
        return """
        Bon, alors on commence la réunion. Marie, tu es là ? Oui, je suis là, parfait.
        Alors, donc on va parler du budget 2025. Jean, tu as préparé les chiffres ? Oui, j'ai
        tout préparé. Alors, on a une augmentation de 20% sur le marketing.
        """

    def test_corrector_initialization(self, mock_api_key):
        """Test d'initialisation du correcteur."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            corrector = QwenCorrector()
            assert corrector.corrector_model == "qwen/qwen3-235b-a22b-07-25:free"
            assert corrector.max_tokens_per_chunk == 45000

    def test_token_counting(self, mock_api_key):
        """Test du comptage de tokens."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            corrector = QwenCorrector()
            text = "Bonjour, ceci est un test."
            token_count = corrector.count_tokens(text)
            assert isinstance(token_count, int)
            assert token_count > 0

    def test_chunking_small_text(self, mock_api_key, sample_transcription_brute):
        """Test découpage texte court (pas de chunking)."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            corrector = QwenCorrector()
            chunks = corrector.chunk_transcription(sample_transcription_brute)
            assert len(chunks) == 1
            assert chunks[0] == sample_transcription_brute

    @patch('src.llm.llm_qwen_corrector.OpenAI')
    def test_correction_success(self, mock_openai, mock_api_key, sample_transcription_brute, sample_transcription_corrigee):
        """Test de correction réussie."""
        mock_choice = Mock()
        mock_choice.message.content = sample_transcription_corrigee

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            corrector = QwenCorrector()
            result = corrector.correct_full_transcription(sample_transcription_brute)

        assert result["success"] is True
        assert result["corrected_transcription"] == sample_transcription_corrigee
        assert result["metrics"]["chunks_processed"] == 1
        assert result["metrics"]["chunks_success"] == 1

    @patch('src.llm.llm_qwen_corrector.OpenAI')
    def test_correction_api_error(self, mock_openai, mock_api_key, sample_transcription_brute):
        """Test d'erreur API lors de la correction."""
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            corrector = QwenCorrector()
            result = corrector.correct_full_transcription(sample_transcription_brute)

        # Doit utiliser le texte original en fallback
        assert result["success"] is False
        assert result["corrected_transcription"] == sample_transcription_brute
        assert len(result["errors"]) == 1

    def test_correct_transcription_for_download_function(self, mock_api_key, sample_transcription_brute):
        """Test de la fonction utilitaire de correction."""
        with patch('src.llm.llm_qwen_corrector.QwenCorrector') as mock_corrector_class:
            mock_corrector = Mock()
            mock_corrector_class.return_value = mock_corrector

            mock_corrector.correct_full_transcription.return_value = {
                "success": True,
                "corrected_transcription": "Texte corrigé",
                "original_transcription": sample_transcription_brute,
                "errors": None,
                "metrics": {"total_duration": 2.5}
            }

            with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
                result = correct_transcription_for_download(sample_transcription_brute)

            assert result["method"] == "qwen_llm_corrector"
            assert result["success"] is True
            assert result["ready_for_download"] is True

class TestIntegration:
    """Tests d'intégration entre extracteur et correcteur."""

    @pytest.fixture
    def mock_api_key(self):
        return "test_api_key"

    def test_missing_api_key(self):
        """Test d'erreur si clé API manquante."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Clé API OpenRouter requise"):
                QwenExtractor()

            with pytest.raises(ValueError, match="Clé API OpenRouter requise"):
                QwenCorrector()

    @patch('src.llm.llm_qwen_extractor.QwenExtractor')
    @patch('src.llm.llm_qwen_corrector.QwenCorrector')
    def test_pipeline_complet_mock(self, mock_corrector_class, mock_extractor_class, mock_api_key):
        """Test du pipeline complet avec mocks."""
        transcription = "Réunion test avec décision importante."

        # Mock extracteur
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        mock_extractor.extract_meeting_insights.return_value = {
            "success": True,
            "data": {"topics_principaux": ["test"]},
            "metrics": {"duration": 1.5}
        }

        mock_extractor.judge_extraction_quality.return_value = {
            "success": True,
            "scores": {"score_global": 85},
            "metrics": {"duration": 0.5}
        }

        # Mock correcteur
        mock_corrector = Mock()
        mock_corrector_class.return_value = mock_corrector

        mock_corrector.correct_full_transcription.return_value = {
            "success": True,
            "corrected_transcription": "Réunion test avec décision importante (corrigé).",
            "original_transcription": transcription,
            "errors": None,
            "metrics": {"total_duration": 2.5}
        }

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            # Test extraction
            extract_result = extract_with_qwen(transcription)
            assert extract_result["success"] is True

            # Test correction
            correct_result = correct_transcription_for_download(transcription)
            assert correct_result["success"] is True
            assert correct_result["ready_for_download"] is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
