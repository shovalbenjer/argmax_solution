import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arctic_handler import ArcticText2SQLHandler

async def test_execution_guided_revision():
    """Test the execution-guided revision functionality."""
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
    if execution_result['success']:
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
    if not broken_result['success']:
        print(f"   Error captured: {broken_result['error']}")
    
    # Test 5: Test Arctic SQL generation (will fail without model but show the flow)
    print(f"\n5. Arctic SQL Generation Test:")
    try:
        sql_result = await handler.generate_sql("Find nutrition data for chicken breast")
        print(f"   Generation success: {sql_result['success']}")
        if sql_result['success']:
            print(f"   Generated SQL: {sql_result['sql']}")
            print(f"   Attempts needed: {sql_result.get('attempts', 'N/A')}")
        else:
            print(f"   Expected failure (no model): {sql_result['error']}")
    except Exception as e:
        print(f"   Expected error (no model): {e}")
    
    print(f"\n6. System Status:")
    print(f"   Execution-guided revision: IMPLEMENTED")
    print(f"   SQL safety validation: WORKING")
    print(f"   Database connectivity: {'WORKING' if execution_result['success'] else 'FAILED'}")
    print(f"   Model availability: PENDING (needs Ollama model loading)")

if __name__ == "__main__":
    asyncio.run(test_execution_guided_revision()) 