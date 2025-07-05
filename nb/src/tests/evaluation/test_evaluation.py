"""
Unit and integration tests for the MLflow evaluation pipeline.

This module contains a suite of tests for the evaluation script located at
`src.evaluation.main`. The tests use `pytest` and `monkeypatch` to isolate
the pipeline from its external dependencies, such as MLflow, the file system,
and the (mocked) student model API.
"""

from unittest.mock import MagicMock

import numba as np
import polars as pl
import pytest
from src.evaluation import main as evaluation_main


@pytest.fixture
def mock_mlflow_eval(monkeypatch):
    """A pytest fixture that mocks all MLflow API calls.

    This fixture uses `monkeypatch` to replace all relevant `mlflow` functions
    with a `MagicMock`. This allows tests to run the evaluation pipeline
    without needing a live MLflow tracking server and enables assertions on

    whether the logging functions were called correctly.
    """
    """Mocks all MLflow calls for the evaluation script."""
    monkeypatch.setattr("mlflow.start_run", MagicMock())
    monkeypatch.setattr("mlflow.set_experiment", MagicMock())
    monkeypatch.setattr("mlflow.log_param", MagicMock())
    monkeypatch.setattr("mlflow.log_metric", MagicMock())
    monkeypatch.setattr("mlflow.log_artifact", MagicMock())


def test_numba_accuracy():
    """Unit test for the Numba-jitted accuracy calculation function.

    This test validates that the `calculate_accuracy_numba` helper function
    correctly computes the accuracy score for a given set of true and
    predicted labels.
    """
    """Tests the Numba JIT function for accuracy calculation."""
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "a", "a", "b"]
    accuracy = evaluation_main.calculate_accuracy_numba(y_true, y_pred)
    assert accuracy == 0.75


def test_evaluation_pipeline(monkeypatch, mock_mlflow_eval, tmp_path):
    """Performs an end-to-end test of the full evaluation pipeline.

    This integration test verifies the main orchestration logic of the
    evaluation script. It uses `monkeypatch` and fixtures to:
    1.  Create a mock ground truth CSV file in a temporary directory.
    2.  Redirect the script's input and output paths to the temporary directory.
    3.  Mock the API call to the student model, providing a controlled sequence
        of responses to generate a predictable accuracy score.
    4.  Run the main evaluation function.
    5.  Assert that MLflow logging functions were called with the expected metrics
        and that output artifacts (confusion matrix, error report) were created
        correctly.
    """
    """Tests the full evaluation pipeline with mocks."""

    # 1. Mock file system inputs
    mock_gt_df = pl.DataFrame(
        {
            "recipe_id": ["r1", "r2"],
            "title": ["t1", "t2"],
            "persona": ["is_keto", "is_vegan"],
            "reasoning": ["...", "..."],
            "citations": ["...", "..."],
            "full_context": ["...", "..."],
        }
    )
    gt_path = tmp_path / "persona_ground_truth.csv"
    mock_gt_df.write_csv(gt_path)
    monkeypatch.setattr(evaluation_main, "GROUND_TRUTH_PATH", gt_path)

    # Mock the output directory
    monkeypatch.setattr(evaluation_main, "OUTPUT_DIR", tmp_path)

    # 2. Mock API call
    # Have the mock return a different persona for the first call, same for the second
    mock_api_responses = ['{"persona": "is_vegan"}', '{"persona": "is_vegan"}']
    monkeypatch.setattr(
        "src.evaluation.main.mock_qwen_api_call",
        MagicMock(side_effect=mock_api_responses),
    )

    # 3. Run the main evaluation function
    evaluation_main.main("test_gt_run_id")

    # 4. Assertions
    # Verify MLflow calls
    evaluation_main.mlflow.log_param.assert_any_call(
        "ground_truth_run_id", "test_gt_run_id"
    )
    evaluation_main.mlflow.log_metric.assert_any_call("accuracy_numba", 0.5)
    evaluation_main.mlflow.log_metric.assert_any_call(
        "f1_score_weighted", pytest.approx(0.4, 0.1)
    )

    # Verify artifacts were created
    cm_path = tmp_path / "confusion_matrix.png"
    errors_path = tmp_path / "misclassifications.csv"
    assert cm_path.exists()
    assert errors_path.exists()

    # Check error report content
    errors_df = pl.read_csv(errors_path)
    assert len(errors_df) == 1
    assert errors_df["recipe_id"][0] == "r1"
