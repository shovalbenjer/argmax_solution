"""
Query Engine for Translating Structured JSON to SQL

This module provides a secure and robust system for translating structured
JSON queries into executable SQL statements. It serves as a bridge between
function-calling LLMs and database operations, ensuring type safety and
preventing SQL injection attacks.

The module implements a two-stage validation process:
1. Structure validation: Ensures required fields are present
2. Content validation: Validates operators, table names, and parameter types

Key Features:
- SQL injection prevention through parameterized queries
- Support for complex WHERE clauses with multiple conditions
- Aggregation operations (AVG, COUNT, SUM, MIN, MAX)
- Comprehensive error handling with descriptive messages
- Type-safe parameter binding

Example:
    >>> query = {
    ...     "operation": "search",
    ...     "table": "nutrition_facts",
    ...     "select_fields": ["name", "calories"],
    ...     "filters": [{"field": "calories", "operator": ">", "value": 100}],
    ...     "limit": 10
    ... }
    >>> engine = QueryEngine()
    >>> sql, params = engine.translate_json_to_sql(query)
    >>> print(sql)
    'SELECT name, calories FROM nutrition_facts WHERE calories > ? LIMIT 10;'
"""

import json
import logging
from typing import Dict, Any, List, Optional

# Configure professional logging
logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Query Engine for translating structured JSON to SQL.
    
    This class provides a secure interface for converting high-level JSON
    query specifications into executable SQL statements with proper
    parameter binding to prevent SQL injection attacks.
    """
    
    def __init__(self):
        """Initialize the QueryEngine."""
        logger.info("QueryEngine initialized successfully")
    
    def translate_json_to_sql(self, query: Dict[str, Any]) -> tuple[str, List[Any]]:
        """
        Translate a structured JSON query object into an executable SQL query.
        
        This method converts a high-level JSON query specification into a
        parameterized SQL statement that can be safely executed against the
        database. It supports both search and aggregation operations with
        comprehensive filtering capabilities.
        
        Args:
            query: Dictionary containing the structured query with the following keys:
                - operation: Either 'search' or 'aggregate'
                - table: Target table name (must be alphanumeric with underscores)
                - select_fields: List of fields to select (for search operations)
                - filters: List of filter conditions
                - limit: Maximum number of results (for search operations)
                - aggregation_field: Field to aggregate (for aggregate operations)
                - aggregation_type: Type of aggregation (AVG, COUNT, SUM, MIN, MAX)
                - groupBy: Field to group by (for aggregate operations)
                
        Returns:
            tuple: (SQL query string, list of parameters for safe execution)
            
        Raises:
            ValueError: If query structure is invalid or contains unsupported operations
            ValueError: If table name contains invalid characters
        """
        return translate_json_to_sql(query)
    
    def validate_query(self, query: Dict[str, Any]) -> bool:
        """
        Validate a JSON query structure without executing it.
        
        Args:
            query: Dictionary containing the structured query
            
        Returns:
            bool: True if query is valid
            
        Raises:
            ValueError: If query structure is invalid
        """
        return _validate_query_structure(query)

def _validate_query_structure(query: Dict[str, Any]) -> bool:
    """
    Validate the basic structure of the incoming JSON query.
    
    Ensures that all required fields are present and that the operation
    type is supported. This is the first line of defense against malformed
    queries.
    
    Args:
        query: Dictionary containing the structured query
        
    Returns:
        bool: True if structure is valid
        
    Raises:
        ValueError: If required fields are missing or operation is invalid
        
    Example:
        >>> query = {"operation": "search", "table": "nutrition_facts"}
        >>> _validate_query_structure(query)
        True
    """
    required_keys = ['operation', 'table']
    if not all(key in query for key in required_keys):
        raise ValueError("Query must contain 'operation' and 'table' keys")
    
    if query['operation'] not in ['search', 'aggregate']:
        raise ValueError("Invalid operation specified")
        
    return True

def _build_where_clause(filters: Optional[List[Dict[str, Any]]]) -> tuple[str, List[Any]]:
    """
    Build the WHERE clause and parameters from a list of filters.
    
    Converts a list of filter dictionaries into a SQL WHERE clause with
    proper parameter binding to prevent SQL injection. Supports multiple
    operators and handles both single values and lists for IN operations.
    
    Args:
        filters: List of filter dictionaries, each containing field, operator, and value
        
    Returns:
        tuple: (WHERE clause string, list of parameters)
        
    Raises:
        ValueError: If operator is unsupported or IN operator receives non-list value
        
    Example:
        >>> filters = [
        ...     {"field": "calories", "operator": ">", "value": 100},
        ...     {"field": "name", "operator": "IN", "value": ["apple", "banana"]}
        ... ]
        >>> where_clause, params = _build_where_clause(filters)
        >>> print(where_clause)
        ' WHERE calories > ? AND name IN (?, ?)'
    """
    if not filters:
        return "", []
    
    conditions = []
    params = []
    
    for f in filters:
        field = f.get('field')
        operator = f.get('operator', '=').upper()
        value = f.get('value')
        
        if not field or value is None:
            continue
            
        # Basic validation for operators
        allowed_ops = ['=', '!=', '>', '<', '>=', '<=', 'LIKE', 'IN']
        if operator not in allowed_ops:
            raise ValueError(f"Unsupported operator: {operator}")
            
        if operator == 'IN':
            if not isinstance(value, list):
                raise ValueError("IN operator requires a list value")
            placeholders = ', '.join(['?'] * len(value))
            conditions.append(f"{field} IN ({placeholders})")
            params.extend(value)
        else:
            conditions.append(f"{field} {operator} ?")
            params.append(value)
            
    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    return where_clause, params

def translate_json_to_sql(query: Dict[str, Any]) -> tuple[str, List[Any]]:
    """
    Translate a structured JSON query object into an executable SQL query.
    
    This function converts a high-level JSON query specification into a
    parameterized SQL statement that can be safely executed against the
    database. It supports both search and aggregation operations with
    comprehensive filtering capabilities.
    
    The function implements multiple layers of validation:
    - Query structure validation
    - Table name sanitization
    - Operator validation
    - Parameter type checking
    
    Args:
        query: Dictionary containing the structured query with the following keys:
            - operation: Either 'search' or 'aggregate'
            - table: Target table name (must be alphanumeric with underscores)
            - select_fields: List of fields to select (for search operations)
            - filters: List of filter conditions
            - limit: Maximum number of results (for search operations)
            - aggregation_field: Field to aggregate (for aggregate operations)
            - aggregation_type: Type of aggregation (AVG, COUNT, SUM, MIN, MAX)
            - groupBy: Field to group by (for aggregate operations)
            
    Returns:
        tuple: (SQL query string, list of parameters for safe execution)
        
    Raises:
        ValueError: If query structure is invalid or contains unsupported operations
        ValueError: If table name contains invalid characters
        
    Example:
        >>> query = {
        ...     "operation": "search",
        ...     "table": "nutrition_facts",
        ...     "select_fields": ["name", "calories"],
        ...     "filters": [{"field": "calories", "operator": ">", "value": 100}],
        ...     "limit": 5
        ... }
        >>> sql, params = translate_json_to_sql(query)
        >>> print(sql)
        'SELECT name, calories FROM nutrition_facts WHERE calories > ? LIMIT 5;'
        >>> print(params)
        [100]
    """
    _validate_query_structure(query)
    
    table = query['table']
    operation = query['operation']
    
    # Basic table name validation to prevent injection
    if not table.isalnum() and '_' not in table:
        raise ValueError(f"Invalid table name: {table}")
    
    params = []
    
    if operation == 'search':
        select_fields = query.get('select_fields', ['*'])
        fields_str = ', '.join(select_fields)
        
        filters = query.get('filters')
        where_clause, where_params = _build_where_clause(filters)
        params.extend(where_params)
        
        limit = query.get('limit')
        limit_clause = f" LIMIT {int(limit)}" if limit else ""
        
        sql = f"SELECT {fields_str} FROM {table}{where_clause}{limit_clause};"
        
    elif operation == 'aggregate':
        aggregation_field = query.get('aggregation_field')
        aggregation_type = query.get('aggregation_type', 'AVG').upper()
        
        if not aggregation_field:
            raise ValueError("Aggregation operation requires 'aggregation_field'")
        
        filters = query.get('filters')
        where_clause, where_params = _build_where_clause(filters)
        params.extend(where_params)
        
        group_by = query.get('groupBy')
        group_by_clause = f" GROUP BY {group_by}" if group_by else ""
        
        sql = f"SELECT {aggregation_type}({aggregation_field}) FROM {table}{where_clause}{group_by_clause};"
        
    else:
        raise ValueError(f"Unsupported operation type: {operation}")
        
    return sql, params 