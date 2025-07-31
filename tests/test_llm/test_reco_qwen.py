"""
Tests unitaires pour le module LLM Qwen Recommendation
Pattern identique à test_qwen_llm.py
"""
import pytest
import json
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Ajout du path pour imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm.llm_qwen_recommendation import (
    QwenRecommendator,
    generate_meeting_recommendations_simple,
    recommend_and_evaluate_meeting
)

class TestQwenRecommendator:
    """Tests pour le recommandateur Qwen."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key pour tests."""
        return "test_api_key"

    @pytest.fixture
    def sample_transcription(self):
        """Transcription de test avec problèmes identifiables."""
        return """
        Réunion équipe marketing. On a discuté du lancement produit.
        Beaucoup de confusion sur les délais. Pas de décisions claires prises.
        Jean a dit qu'il fallait revoir le budget. Marie n'était pas d'accord.
        La réunion a duré 2h sans conclusion. Personne ne savait qui fait quoi.
        On se revoit la semaine prochaine pour re-discuter de tout ça.
        """

    @pytest.fixture
    def sample_extraction_data(self):
        """Données d'extraction pour enrichir le contexte."""
        return {
            "success": True,
            "extraction": {
                "topics_principaux": ["lancement produit", "budget marketing", "délais", "équipe", "planning"],
                "points_a_retenir": [
                    "Confusion sur les délais du lancement",
                    "Désaccord entre Jean et Marie sur le budget",
                    "Aucune décision claire prise",
                    "Réunion de 2h sans conclusion",
                    "Prochaine réunion programmée"
                ]
            }
        }

    @pytest.fixture
    def mock_recommendations_response(self):
        """Réponse simulée des recommandations."""
        return {
            "recommandations_principales": [
                {
                    "categorie": "Structure",
                    "titre": "Définir un ordre du jour précis",
                    "description": "Préparer et partager l'agenda 24h avant le meeting avec objectifs clairs",
                    "impact": "high",
                    "facilite_implementation": "easy"
                },
                {
                    "categorie": "Animation",
                    "titre": "Améliorer la prise de décision",
                    "description": "Utiliser des techniques de vote ou consensus pour trancher les désaccords",
                    "impact": "high",
                    "facilite_implementation": "medium"
                },
                {
                    "categorie": "Efficacité",
                    "titre": "Limiter la durée des réunions",
                    "description": "Fixer un timer et respecter les créneaux alloués à chaque sujet",
                    "impact": "medium",
                    "facilite_implementation": "easy"
                }
            ],
            "resume_conseil": "Meeting nécessitant une structure claire et des processus de décision définis pour éviter les discussions sans fin.",
            "score_amelioration_potentiel": 85
        }

    @pytest.fixture
    def mock_judge_response(self):
        """Réponse simulée du juge d'évaluation."""
        return {
            "pertinence": 90,
            "actionnabilite": 85,
            "impact_potentiel": 88,
            "specificite": 82,
            "score_global": 86,
            "justification": "Recommandations bien adaptées aux problèmes identifiés dans la transcription",
            "qualite_conseil": "high"
        }

    def test_recommender_initialization(self, mock_api_key):
        """Test d'initialisation du recommandateur."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            recommender = QwenRecommendator()
            assert recommender.recommender_model == "qwen/qwen3-235b-a22b-07-25:free"
            assert recommender.judge_model == "tngtech/deepseek-r1t-chimera:free"
            assert recommender.fallback_judge == "nousresearch/deephermes-3-llama-3-8b-preview:free"

    @patch('src.llm.llm_qwen_recommendation.OpenAI')
    def test_recommendations_generation_success(self, mock_openai, mock_api_key,
                                              sample_transcription, mock_recommendations_response):
        """Test de génération de recommandations réussie."""
        # Mock de la réponse API
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_recommendations_response)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            recommender = QwenRecommendator()
            result = recommender.generate_meeting_recommendations(sample_transcription)

        assert result["success"] is True
        assert "data" in result
        assert len(result["data"]["recommandations_principales"]) == 3
        assert result["data"]["score_amelioration_potentiel"] == 85
        assert result["metrics"]["context_enriched"] is False

    @patch('src.llm.llm_qwen_recommendation.OpenAI')
    def test_recommendations_with_extraction_context(self, mock_openai, mock_api_key,
                                                   sample_transcription, sample_extraction_data,
                                                   mock_recommendations_response):
        """Test de génération avec contexte d'extraction enrichi."""
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_recommendations_response)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            recommender = QwenRecommendator()
            result = recommender.generate_meeting_recommendations(sample_transcription, sample_extraction_data)

        assert result["success"] is True
        assert result["metrics"]["context_enriched"] is True

    @patch('src.llm.llm_qwen_recommendation.OpenAI')
    def test_recommendations_json_parse_error(self, mock_openai, mock_api_key, sample_transcription):
        """Test d'erreur de parsing JSON."""
        # Mock réponse JSON invalide
        mock_choice = Mock()
        mock_choice.message.content = "Réponse non-JSON invalide avec recommandations..."

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            recommender = QwenRecommendator()
            result = recommender.generate_meeting_recommendations(sample_transcription)

        assert result["success"] is False
        assert result["error"] == "json_parsing_failed"
        assert "raw_content" in result

    @patch('src.llm.llm_qwen_recommendation.OpenAI')
    def test_judge_evaluation_success(self, mock_openai, mock_api_key,
                                    sample_transcription, mock_recommendations_response, mock_judge_response):
        """Test d'évaluation par le judge réussie."""
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_judge_response)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            recommender = QwenRecommendator()
            result = recommender.judge_recommendations_quality(sample_transcription, mock_recommendations_response)

        assert result["success"] is True
        assert result["scores"]["score_global"] == 86
        assert result["qualite_conseil"] == "high"
        assert result["metrics"]["fallback_used"] is False

    @patch('src.llm.llm_qwen_recommendation.OpenAI')
    def test_judge_fallback_success(self, mock_openai, mock_api_key,
                                  sample_transcription, mock_recommendations_response, mock_judge_response):
        """Test de fallback du judge en cas d'échec du principal."""
        # Premier appel (judge principal) échoue
        mock_openai.return_value.chat.completions.create.side_effect = [
            Exception("Judge principal failed"),
            # Deuxième appel (fallback) réussit
            Mock(choices=[Mock(message=Mock(content=json.dumps(mock_judge_response)))])
        ]

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            recommender = QwenRecommendator()
            result = recommender.judge_recommendations_quality(sample_transcription, mock_recommendations_response)

        assert result["success"] is True
        assert result["metrics"]["fallback_used"] is True

    @patch('src.llm.llm_qwen_recommendation.OpenAI')
    def test_judge_json_extraction_robust(self, mock_openai, mock_api_key,
                                        sample_transcription, mock_recommendations_response, mock_judge_response):
        """Test d'extraction JSON robuste avec texte additionnel."""
        # Réponse avec du texte avant le JSON (comme dans le corrector)
        response_with_text = f"""Voici mon évaluation des recommandations:

Les recommandations sont bien adaptées au contexte.

{json.dumps(mock_judge_response)}

Fin de l'évaluation."""

        mock_choice = Mock()
        mock_choice.message.content = response_with_text

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            recommender = QwenRecommendator()
            result = recommender.judge_recommendations_quality(sample_transcription, mock_recommendations_response)

        assert result["success"] is True
        assert result["scores"]["score_global"] == 86

    def test_generate_meeting_recommendations_simple_function(self, mock_api_key, sample_transcription):
        """Test de la fonction utilitaire simple."""
        with patch('src.llm.llm_qwen_recommendation.QwenRecommendator') as mock_recommender_class:
            mock_recommender = Mock()
            mock_recommender_class.return_value = mock_recommender

            mock_recommender.generate_meeting_recommendations.return_value = {
                "success": True,
                "data": {"recommandations_principales": ["test"]},
                "metrics": {"duration": 1.5, "context_enriched": False}
            }

            with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
                result = generate_meeting_recommendations_simple(sample_transcription)

            assert result["success"] is True
            assert result["method"] == "qwen_llm_recommender_simple"

    def test_recommend_and_evaluate_meeting_function(self, mock_api_key, sample_transcription):
        """Test de la fonction pipeline complète."""
        with patch('src.llm.llm_qwen_recommendation.QwenRecommendator') as mock_recommender_class:
            mock_recommender = Mock()
            mock_recommender_class.return_value = mock_recommender

            # Mock du pipeline complet
            mock_recommender.recommend_and_evaluate.return_value = {
                "success": True,
                "method": "qwen_recommender_with_judge",
                "recommendations": {
                    "recommandations_principales": ["test"],
                    "nb_recommandations": 3
                },
                "quality_evaluation": {"success": True, "score_global": 85},
                "metrics": {"total_duration": 2.0}
            }

            with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
                result = recommend_and_evaluate_meeting(sample_transcription)

            assert result["success"] is True
            assert result["method"] == "qwen_recommender_with_judge"
            assert result["recommendations"]["nb_recommandations"] == 3

class TestRecommendationPipeline:
    """Tests du pipeline complet recommandations."""

    @pytest.fixture
    def mock_api_key(self):
        return "test_api_key"

    @patch('src.llm.llm_qwen_recommendation.OpenAI')
    def test_full_pipeline_success(self, mock_openai, mock_api_key,
                                 sample_transcription, mock_recommendations_response, mock_judge_response):
        """Test du pipeline complet recommandations + évaluation."""
        # Mock des deux appels API (génération + judge)
        mock_openai.return_value.chat.completions.create.side_effect = [
            # Premier appel: génération recommandations
            Mock(choices=[Mock(message=Mock(content=json.dumps(mock_recommendations_response)))]),
            # Deuxième appel: évaluation judge
            Mock(choices=[Mock(message=Mock(content=json.dumps(mock_judge_response)))])
        ]

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            recommender = QwenRecommendator()
            result = recommender.recommend_and_evaluate(sample_transcription)

        assert result["success"] is True
        assert result["method"] == "qwen_recommender_with_judge"
        assert result["ready_for_implementation"] is True
        assert result["conseil_quality"] == "high"
        assert len(result["recommendations"]["recommandations_principales"]) >= 1  # Au moins 1 recommandation

    @patch('src.llm.llm_qwen_recommendation.OpenAI')
    def test_pipeline_generation_failure(self, mock_openai, mock_api_key, sample_transcription):
        """Test d'échec de génération des recommandations."""
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
            recommender = QwenRecommendator()
            result = recommender.recommend_and_evaluate(sample_transcription)

        assert result["success"] is False
        assert result["error"] == "recommendation_generation_failed"

    def test_missing_api_key(self):
        """Test d'erreur si clé API manquante."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Clé API OpenRouter requise"):
                QwenRecommendator()

class TestRecommendationIntegration:
    """Tests d'intégration avec les autres modules."""

    @pytest.fixture
    def mock_api_key(self):
        return "test_api_key"

    def test_integration_with_extraction_data(self, mock_api_key, sample_transcription, sample_extraction_data):
        """Test d'intégration avec des données d'extraction."""
        with patch('src.llm.llm_qwen_recommendation.QwenRecommendator') as mock_recommender_class:
            mock_recommender = Mock()
            mock_recommender_class.return_value = mock_recommender

            mock_recommender.recommend_and_evaluate.return_value = {
                "success": True,
                "recommendations": {"nb_recommandations": 4},
                "metrics": {"context_enriched": True}
            }

            with patch.dict('os.environ', {'OPENROUTER_API_KEY': mock_api_key}):
                result = recommend_and_evaluate_meeting(sample_transcription, sample_extraction_data)

            # Vérifier que l'extraction_data a été passée
            mock_recommender.recommend_and_evaluate.assert_called_with(sample_transcription, sample_extraction_data)
            assert result["success"] is True

# Fixtures globales réutilisées
@pytest.fixture
def sample_transcription():
    return """
    Réunion équipe marketing. On a discuté du lancement produit.
    Beaucoup de confusion sur les délais. Pas de décisions claires prises.
    Jean a dit qu'il fallait revoir le budget. Marie n'était pas d'accord.
    La réunion a duré 2h sans conclusion.
    """

@pytest.fixture
def sample_extraction_data():
    return {
        "success": True,
        "extraction": {
            "topics_principaux": ["lancement produit", "budget marketing"],
            "points_a_retenir": ["Confusion sur les délais", "Désaccord sur budget"]
        }
    }

@pytest.fixture
def mock_recommendations_response():
    return {
        "recommandations_principales": [
            {
                "categorie": "Structure",
                "titre": "Définir un ordre du jour précis",
                "description": "Préparer et partager l'agenda",
                "impact": "high",
                "facilite_implementation": "easy"
            },
            {
                "categorie": "Animation",
                "titre": "Améliorer la prise de décision",
                "description": "Utiliser des techniques de vote",
                "impact": "high",
                "facilite_implementation": "medium"
            },
            {
                "categorie": "Efficacité",
                "titre": "Limiter la durée",
                "description": "Fixer un timer",
                "impact": "medium",
                "facilite_implementation": "easy"
            }
        ],
        "resume_conseil": "Meeting nécessitant une structure claire",
        "score_amelioration_potentiel": 85
    }

@pytest.fixture
def mock_judge_response():
    return {
        "pertinence": 90,
        "actionnabilite": 85,
        "impact_potentiel": 88,
        "specificite": 82,
        "score_global": 86,
        "justification": "Recommandations bien adaptées",
        "qualite_conseil": "high"
    }

if __name__ == "__main__":
    # Test simple pour validation
    print("🧪 Test simple validation...")

    # Test d'import
    try:
        from src.llm.llm_qwen_recommendation import generate_meeting_recommendations_simple
        print("✅ Import réussi")
    except Exception as e:
        print(f"❌ Erreur import: {e}")

    # Test avec mock API key
    import os
    os.environ['OPENROUTER_API_KEY'] = 'test_key'

    try:
        test_transcription = "Réunion test avec problèmes d'organisation"
        # result = generate_meeting_recommendations_simple(test_transcription)
        print("✅ Structure module OK")
    except Exception as e:
        print(f"❌ Erreur structure: {e}")

    # Lancer les tests complets avec pytest
    pytest.main([__file__, "-v"])
