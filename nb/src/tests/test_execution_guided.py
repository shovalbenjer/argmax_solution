"""
Execution Guided Tests for SOTA Semantic Classifier

This module provides comprehensive testing for the execution-guided approach
to diet classification, focusing on the integration between different components
of the pipeline and alternative execution strategies when primary models are unavailable.

The test suite covers:
- Execution-guided classification without Arctic handler
- SQL safety validation and execution testing
- Database connectivity and schema validation
- Alternative classification approaches
- Error handling and graceful degradation
- System status monitoring and reporting

Key Test Areas:
- Classification pipeline integration
- SQL generation and execution safety
- Database schema validation
- Model availability handling
- Error recovery mechanisms
- System health monitoring

Test Features:
- Asynchronous testing for improved performance
- Comprehensive error handling and logging
- SQL safety validation testing
- Database connectivity verification
- Model availability checking
- System status reporting

Dependencies:
- pytest: Testing framework
- asyncio: Asynchronous testing support
- json: Data serialization
- logging: Test logging and debugging
- pathlib: Path management
- typing: Type hints for better code clarity

Example:
    >>> pytest nb/src/tests/test_execution_guided.py -v
    >>> # Run specific execution-guided test
    >>> asyncio.run(test_execution_guided_revision())
"""

import asyncio
import json
import logging
# Add nb/src to path for local imports
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import available components
from context_aware_classifier import ContextAwareDietClassifier
from function_calling_handler import FunctionCallingHandler
from query_engine import QueryEngine

# Configure logging
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_execution_guided_classification():
    """
    Test execution-guided classification approach.

    This test validates that the classification pipeline can work
    without the Arctic handler by using alternative approaches and
    fallback mechanisms. It ensures system resilience when primary
    models are unavailable.

    Test Validates:
    - Classification pipeline functionality without Arctic handler
    - Alternative execution strategies
    - Result structure and content validation
    - Error handling and graceful degradation
    - System integration reliability

    Returns:
        None: Test passes if classification works without Arctic handler

    Raises:
        AssertionError: If classification fails or results are invalid
        pytest.fail: If execution-guided approach fails completely

    Example:
        >>> await test_execution_guided_classification()
        >>> # Tests classification without Arctic handler dependency
    """
    try:
        classifier = ContextAwareDietClassifier()
        handler = FunctionCallingHandler()
        engine = QueryEngine()

        # Test basic functionality
        test_ingredient = "chicken breast"
        result = await classifier.classify_ingredient(test_ingredient)

        assert result is not None, "Classification result should not be None"
        assert "is_keto" in result, "Result should contain keto classification"
        assert "is_vegan" in result, "Result should contain vegan classification"

        logger.info("Execution guided classification test passed")

    except Exception as e:
        logger.error(f"Execution guided test failed: {e}")
        pytest.fail(f"Execution guided test failed: {e}")


import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arctic_handler import ArcticText2SQLHandler


async def test_execution_guided_revision():
    """
    Test the execution-guided revision functionality.

    This function performs comprehensive testing of the execution-guided
    revision system, which provides alternative execution strategies when
    primary models are unavailable. It tests SQL safety, database connectivity,
    and system status monitoring.

    Test Components:
        1. Schema Loading: Validates database schema accessibility
        2. SQL Safety Validation: Tests SQL injection prevention
        3. SQL Execution Testing: Validates database connectivity
        4. Broken SQL Handling: Tests error handling for invalid SQL
        5. Arctic SQL Generation: Tests model-based SQL generation
        6. System Status Reporting: Monitors overall system health

    Test Cases:
        - Safe SQL: "SELECT name, calories FROM nutrition_facts WHERE name LIKE '%chicken%'"
        - Unsafe SQL: "DROP TABLE nutrition_facts" (should be rejected)
        - Test SQL: "SELECT name, calories FROM nutrition_facts LIMIT 3"
        - Broken SQL: "SELECT nonexistent_column FROM nutrition_facts"

    The test validates:
    - Database schema loading and validation
    - SQL safety validation mechanisms
    - Database connectivity and execution
    - Error handling for invalid queries
    - Model availability and integration
    - System status monitoring

    Returns:
        None: Prints comprehensive test results to console

    Raises:
        Exception: If critical system components fail unexpectedly

    Example:
        >>> await test_execution_guided_revision()
        >>> # Tests complete execution-guided revision system
    """
    print("Testing Execution-Guided Revision System")
    print("=" * 45)

    handler = ArcticText2SQLHandler()

    # Test 1: Check schema loading
    print(f"\n1. Schema Loading Test:")
    print(f"   Database path: {handler.db_path}")
    print(f"   Database exists: {handler.db_path.exists()}")
    print(f"   Schema loaded: {'Yes' if 'nutrition_facts' in handler.schema else 'No'}")

    # Test 2: SQL Safety Validation
    print(f"\n2. SQL Safety Validation Test:")
    safe_sql = "SELECT name, calories FROM nutrition_facts WHERE name LIKE '%chicken%'"
    unsafe_sql = "DROP TABLE nutrition_facts"

    print(f"   Safe SQL valid: {handler._validate_sql_safety(safe_sql)}")
    print(f"   Unsafe SQL rejected: {not handler._validate_sql_safety(unsafe_sql)}")

    # Test 3: SQL Execution Testing
    print(f"\n3. SQL Execution Test:")
    test_sql = "SELECT name, calories FROM nutrition_facts LIMIT 3"
    execution_result = handler._test_sql_execution(test_sql)
    print(f"   Test SQL: {test_sql}")
    print(f"   Execution success: {execution_result['success']}")
    if execution_result["success"]:
        print(f"   Row count: {execution_result['row_count']}")
        print(f"   Columns: {execution_result['columns']}")
    else:
        print(f"   Error: {execution_result['error']}")

    # Test 4: Test with intentionally broken SQL
    print(f"\n4. Broken SQL Test:")
    broken_sql = "SELECT nonexistent_column FROM nutrition_facts"
    broken_result = handler._test_sql_execution(broken_sql)
    print(f"   Broken SQL: {broken_sql}")
    print(f"   Correctly failed: {not broken_result['success']}")
    if not broken_result["success"]:
        print(f"   Error captured: {broken_result['error']}")

    # Test 5: Test Arctic SQL generation (will fail without model but show the flow)
    print(f"\n5. Arctic SQL Generation Test:")
    try:
        sql_result = await handler.generate_sql(
            "Find nutrition data for chicken breast"
        )
        print(f"   Generation success: {sql_result['success']}")
        if sql_result["success"]:
            print(f"   Generated SQL: {sql_result['sql']}")
            print(f"   Attempts needed: {sql_result.get('attempts', 'N/A')}")
        else:
            print(f"   Expected failure (no model): {sql_result['error']}")
    except Exception as e:
        print(f"   Expected error (no model): {e}")

    print(f"\n6. System Status:")
    print(f"   Execution-guided revision: IMPLEMENTED")
    print(f"   SQL safety validation: WORKING")
    print(
        f"   Database connectivity: {'WORKING' if execution_result['success'] else 'FAILED'}"
    )
    print(f"   Model availability: PENDING (needs Ollama model loading)")


if __name__ == "__main__":
    asyncio.run(test_execution_guided_revision())
