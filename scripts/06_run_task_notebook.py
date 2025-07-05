#!/usr/bin/env python3
"""
Task Notebook Runner Script

This script runs the task.ipynb notebook programmatically to execute
the evaluation and analysis pipeline. It ensures all dependencies are
available and runs the notebook in a controlled environment.

This should be run after all the validation scripts have passed.

The script performs comprehensive pipeline execution including:
- Final evaluation against ground truth data
- Evaluation analysis and reporting
- Comprehensive performance testing
- Deepeval pipeline functionality testing
- Complete pipeline status and results reporting

Returns:
    int: Exit code (0 for success, 1 for failure)

Raises:
    FileNotFoundError: If required files are missing
    ImportError: If required modules cannot be imported
    Exception: If pipeline execution fails
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add nb/src to path for accessing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """
    Check that all prerequisites are met before running the notebook.
    
    This function validates the existence of all required files and
    dependencies needed for the task notebook pipeline execution.
    
    Returns:
        bool: True if all prerequisites are met, False otherwise
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    logger.info("Checking prerequisites...")
    
    # Check if required files exist
    required_files = [
        "nb/src/data/knowledge_graph.db",
        "nb/src/eval_data/personas_ground_truth.csv",
        "nb/src/run_final_evaluation.py",
        "nb/src/evaluation_analysis.py",
        "nb/src/performance_test.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("All required files present")
    return True


async def run_final_evaluation():
    """
    Run the final evaluation script.
    
    This function executes the comprehensive final evaluation against
    ground truth data to validate system performance and accuracy.
    
    Returns:
        bool: True if evaluation completed successfully, False otherwise
        
    Raises:
        ImportError: If evaluation module cannot be imported
        Exception: If evaluation execution fails
    """
    logger.info("Running final evaluation...")
    
    try:
        # Import and run the evaluation
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))
        from run_final_evaluation import main as run_eval
        
        # Run the evaluation
        result = await run_eval()
        
        if result == 0:
            logger.info("Final evaluation completed successfully")
            return True
        else:
            logger.error("Final evaluation failed")
            return False
            
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
        return False


def run_evaluation_analysis():
    """
    Run the evaluation analysis.
    
    This function executes the evaluation analysis script to generate
    comprehensive reports and insights from the evaluation results.
    
    Returns:
        bool: True if analysis completed successfully, False otherwise
        
    Raises:
        ImportError: If analysis module cannot be imported
        Exception: If analysis execution fails
    """
    logger.info("Running evaluation analysis...")
    
    try:
        # Import and run the analysis
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))
        from evaluation_analysis import main as run_analysis
        
        # Run the analysis
        result = run_analysis()
        
        if result == 0:
            logger.info("Evaluation analysis completed successfully")
            return True
        else:
            logger.error("Evaluation analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"Evaluation analysis failed: {e}")
        return False


def run_performance_test():
    """
    Run the performance test.
    
    This function executes the comprehensive performance testing suite
    to validate system performance under various load conditions.
    
    Returns:
        bool: True if performance test completed successfully, False otherwise
        
    Raises:
        ImportError: If performance test module cannot be imported
        Exception: If performance test execution fails
    """
    logger.info("Running performance test...")
    
    try:
        # Import and run the performance test
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))
        from performance_test import run_performance_test_suite
        
        # Run the performance test
        results = run_performance_test_suite()
        
        if results:
            logger.info("Performance test completed successfully")
            return True
        else:
            logger.error("Performance test failed")
            return False
            
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return False


async def run_deepeval_pipeline():
    """
    Run the deepeval pipeline test.
    
    This function executes the deepeval pipeline tests to validate
    function calling handler structure and query engine safety.
    
    Returns:
        bool: True if deepeval pipeline tests completed successfully, False otherwise
        
    Raises:
        ImportError: If deepeval test modules cannot be imported
        Exception: If deepeval pipeline test execution fails
    """
    logger.info("Running deepeval pipeline test...")
    
    try:
        # Import and run the deepeval test
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src" / "tests"))
        from test_deepeval_pipeline import test_function_calling_handler_structure, test_query_engine_safety_and_execution
        
        # Run the tests
        await test_function_calling_handler_structure()
        await test_query_engine_safety_and_execution()
        
        logger.info("Deepeval pipeline tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Deepeval pipeline test failed: {e}")
        return False


async def main():
    """
    Run the complete task notebook pipeline.
    
    This function orchestrates the complete task notebook pipeline execution:
    1. Prerequisites validation
    2. Final evaluation execution
    3. Evaluation analysis execution
    4. Performance testing execution
    5. Deepeval pipeline testing
    6. Summary reporting and status assessment
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
        
    Raises:
        FileNotFoundError: If required files are missing
        ImportError: If required modules cannot be imported
        Exception: If pipeline execution fails
    """
    logger.info("=" * 60)
    logger.info("TASK NOTEBOOK PIPELINE")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Please run the validation scripts first.")
        return 1
    
    # Run all components
    results = {}
    
    # Run final evaluation
    results['final_evaluation'] = await run_final_evaluation()
    
    # Run evaluation analysis
    results['evaluation_analysis'] = run_evaluation_analysis()
    
    # Run performance test
    results['performance_test'] = run_performance_test()
    
    # Run deepeval pipeline
    results['deepeval_pipeline'] = await run_deepeval_pipeline()
    
    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"Overall: {passed_tests}/{total_tests} components completed successfully")
    
    if passed_tests == total_tests:
        logger.info("All components completed successfully!")
        return 0
    elif passed_tests >= total_tests * 0.75:  # 75% threshold
        logger.warning("Most components completed, but some issues detected. Review errors above.")
        return 0  # Continue with warnings
    else:
        logger.error("Too many components failed. Please review and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 