"""
Integration test for the experimental ground truth generation pipeline.

This module contains a single, comprehensive test for the main orchestration
function in `src.ground_truth.main`. The test uses `pytest` and `monkeypatch`
to completely isolate the main function from its dependencies, such as the
file system, MLflow, the context engine, and the (mocked) Gemini API.

This approach allows for validation of the pipeline's "plumbing"—ensuring it
calls all its components in the correct order and handles data correctly—without
relying on the actual implementation of those components.
"""

from unittest.mock import MagicMock

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from src.ground_truth import main as ground_truth_main


@pytest.fixture
def mock_mlflow(monkeypatch):
    """A pytest fixture that mocks all MLflow API calls."""
    mock_client = MagicMock()
    monkeypatch.setattr("mlflow.start_run", MagicMock(return_value=mock_client))
    monkeypatch.setattr("mlflow.set_experiment", MagicMock())
    monkeypatch.setattr("mlflow.log_param", MagicMock())
    monkeypatch.setattr("mlflow.log_metric", MagicMock())
    monkeypatch.setattr("mlflow.log_artifact", MagicMock())
    return mock_client


def test_ground_truth_pipeline(monkeypatch, mock_mlflow, tmp_path):
    """Performs an end-to-end test of the experimental ground truth pipeline.

    This integration test verifies the main orchestration logic by mocking all
    external dependencies. It ensures that the pipeline correctly handles data
    flow and calls its various components as expected.

    The test uses `monkeypatch` to:
    1.  Replace file system reads with a mock DataFrame.
    2.  Redirect file system writes to a temporary directory.
    3.  Replace the Cython context engine call with a mock that returns
        predictable data.
    4.  Replace the Gemini API call with a mock that returns a predictable
        response.
    5.  Run the main function and assert that MLflow was called with the correct
        parameters and that the final output CSV was created with the expected content.
    """
    """
    Tests the full ground truth generation pipeline by mocking all
    external dependencies (filesystem, APIs, mlflow).
    """
    # 1. Mock file system inputs
    mock_recipe_df = pl.DataFrame(
        {
            "id": ["recipe_1"],
            "title": ["Test Recipe"],
            "ingredients": ["['1 cup flour', '1 large egg']"],
        }
    )
    monkeypatch.setattr("polars.read_csv", MagicMock(return_value=mock_recipe_df))

    # Create a dummy prompt template file
    prompt_path = tmp_path / "prompt_template.txt"
    prompt_path.write_text("Recipe: {recipe_id}, Context: {contexts}")
    monkeypatch.setattr(ground_truth_main, "PROMPT_TEMPLATE_PATH", prompt_path)

    # 2. Mock function calls
    mock_context_result = [{"raw_ingredient": "1 cup flour", "data": "mock_data"}]
    monkeypatch.setattr(
        "src.ground_truth.main.process_recipe_batch",
        MagicMock(return_value=mock_context_result),
    )

    mock_api_response = (
        '{"persona": "not_keto_or_vegan", "reasoning": "Test reason.", "citations": []}'
    )
    monkeypatch.setattr(
        "src.ground_truth.main.mock_gemini_api_call",
        MagicMock(return_value=mock_api_response),
    )

    # Mock the output directory
    monkeypatch.setattr(ground_truth_main, "OUTPUT_DIR", tmp_path)

    # 3. Run the main function
    ground_truth_main.main()

    # 4. Assertions
    # Verify MLflow was used correctly
    ground_truth_main.mlflow.set_experiment.assert_called_with(
        "Ground Truth Generation"
    )
    ground_truth_main.mlflow.log_param.assert_any_call("num_recipes", 1)
    ground_truth_main.mlflow.log_metric.assert_called_with(
        "successfully_processed_recipes", 1
    )

    # Verify the output file was created and has correct data
    output_file = tmp_path / "persona_ground_truth.csv"
    assert output_file.exists()
    result_df = pl.read_csv(output_file)
    assert result_df["persona"][0] == "not_keto_or_vegan"
    assert result_df["reasoning"][0] == "Test reason."
