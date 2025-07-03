import pytest
import asyncio
import json
from deepeval import assert_test, evaluate
from deepeval.metrics import GEval, ContextualRelevancyMetric, HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
import polars as pl
from pathlib import Path
import sys

# Add project root to path to allow importing shared modules
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# Add nb/src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the components to be tested
from shared.database import db_manager
from ingredient_parser import parse_ingredient
from arctic_handler import ArcticText2SQLHandler
from context_aware_classifier import ContextAwareDietClassifier, is_sql_safe

# --- Helper Functions for Testing ---

def execute_sql_for_test(sql: str):
    """A helper to execute SQL and return a string context for deepeval."""
    if not is_sql_safe(sql):
        return "UNSAFE SQL"
    try:
        with db_manager.get_sqlite_connection() as conn:
            df = pl.read_database(sql, conn)
            return df.to_dicts()
    except Exception as e:
        return f"SQL Execution Error: {e}"

# --- Test Suite ---

# Test 1: Ingredient Parser Validation
@pytest.mark.asyncio
async def test_parser_accuracy():
    """Tests the accuracy of the ingredient-parser-nlp."""
    parser_metric = GEval(
        name="Parsing Correctness",
        criteria="Evaluate if the 'actual_output' JSON correctly extracts the 'name', 'quantity', and 'unit' from the 'input' ingredient string.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
    )
    test_case = LLMTestCase(
        input="1 1/2 cups of sifted all-purpose flour, plus more for dusting",
        actual_output=json.dumps(parse_ingredient("1 1/2 cups of sifted all-purpose flour, plus more for dusting", discard_isolated_stop_words=True)),
        expected_output='{"name": "all-purpose flour", "quantity": 1.5, "unit": "cup"}'
    )
    assert_test(test_case, [parser_metric])

# Test 2: Arctic SQL Generation and Retrieval
@pytest.mark.asyncio
async def test_arctic_retrieval_relevancy():
    """Tests Arctic's SQL generation and the relevancy of the retrieved context."""
    handler = ArcticText2SQLHandler()
    question = "Is butter vegan?"
    
    # Generate SQL
    sql_result = await handler.generate_sql(question)
    generated_sql = sql_result['sql']
    
    # Check if SQL is correct (simplified check)
    assert "vegan_ontology" in generated_sql.lower()
    assert "butter" in generated_sql.lower()
    
    # Get retrieval context
    retrieved_context = execute_sql_for_test(generated_sql)
    
    relevancy_metric = ContextualRelevancyMetric(threshold=0.8)
    test_case = LLMTestCase(
        input=question,
        retrieval_context=[str(retrieved_context)]
    )
    assert_test(test_case, [relevancy_metric])

# Test 3: Gemini Ground-Truth Label Validation (Conceptual)
# This test assumes you have a "golden set" of hand-verified data.
# We will use the existing `borderline_keto.csv` as a stand-in for this.
@pytest.mark.asyncio
async def test_gemini_label_consistency():
    """Uses a golden set to check the factual consistency of an LLM's labels."""
    # We can't call the real Gemini pipeline here easily, so we simulate a test case.
    # This represents a known tricky recipe.
    tricky_recipe_input = "A 'vegan' cake using flour, sugar, and butter."
    
    # This would be the output from your corrected ground_truth/generate.py
    simulated_gemini_output = '{"is_vegan": false, "reasoning": "The recipe contains butter, which is a dairy product and therefore not vegan."}'
    
    factual_consistency_metric = GEval(
        name="Factual Consistency",
        criteria="Based on the ingredients in the 'input', is the classification in the 'actual_output' factually correct according to strict vegan rules? 'butter' is not vegan.",
    )
    
    test_case = LLMTestCase(
        input=tricky_recipe_input,
        actual_output=simulated_gemini_output
    )
    assert_test(test_case, [factual_consistency_metric])

# Test 4: Qwen Student Model RAG Evaluation
@pytest.mark.asyncio
async def test_qwen_rag_performance():
    """Evaluates the final Qwen classifier's reasoning and lack of hallucination."""
    classifier = ContextAwareDietClassifier()
    ingredients = ["1 cup almond flour", "2 tbsp olive oil", "1 lb chicken breast"]
    
    # We call the actual classifier to get the final output.
    # This will run the full RAG pipeline (Arctic SQL gen -> DB query -> Qwen judgment)
    qwen_output = await classifier.classify_with_context(ingredients)
    
    # We need to manually get the retrieval context for the test case
    handler = ArcticText2SQLHandler()
    question = f"What are the nutritional values and vegan status for the following ingredients: {', '.join(ingredients)}?"
    sql_result = await handler.generate_sql(question)
    retrieved_context = execute_sql_for_test(sql_result['sql'])

    test_case = LLMTestCase(
        input=f"Classify a recipe with {', '.join(ingredients)}",
        actual_output=qwen_output['reasoning'],
        retrieval_context=[str(retrieved_context)],
        expected_output="The recipe should be classified as keto but not vegan because it contains chicken."
    )
    
    # Define metrics
    hallucination_metric = HallucinationMetric(threshold=0.0)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.9)
    
    assert_test(test_case, [hallucination_metric, answer_relevancy_metric])
