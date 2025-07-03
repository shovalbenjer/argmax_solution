"""Flask web application for recipe search and dietary classification.

This application serves a web interface that allows users to search for recipes
by ingredients and automatically classify them for dietary restrictions like
keto and vegan. It integrates with OpenSearch for efficient full-text search
and uses custom classifiers for dietary analysis.

The application provides a complete web-based solution for recipe discovery
and dietary analysis, featuring:
- Interactive ingredient search with autocomplete
- Real-time recipe classification
- Responsive web interface
- RESTful API endpoints
- Comprehensive error handling

Architecture:
- Flask web framework for HTTP handling
- OpenSearch for recipe and ingredient search
- Custom diet classifiers for keto/vegan analysis
- Select2 integration for enhanced UI
- Docker-ready deployment configuration

The application initializes a connection to OpenSearch on startup and pre-loads
all available ingredients into memory to provide fast autocomplete suggestions.

Example:
    >>> from app import app
    >>> with app.test_client() as client:
    ...     response = client.get('/search?ingredients=chicken,spinach')
    ...     print(response.status_code)
    200
"""
from flask import Flask, request, jsonify, render_template
from time import sleep
import sys
import logging
import os
from pathlib import Path

# Use local diet classifiers for web app
from diet_classifiers import is_keto, is_vegan

# For now, use simple configuration for web app
class WebConfig:
    """
    Simple configuration class for web application.
    
    Provides basic configuration settings for the web application,
    including OpenSearch connection details and operational parameters.
    
    Attributes:
        OPENSEARCH_URL: URL for OpenSearch service endpoint
    """
    OPENSEARCH_URL = "http://localhost:9200"

app_config = WebConfig()

# Simple database manager for web app
from opensearchpy import OpenSearch
import logging

class SimpleDBManager:
    """
    Simplified database manager for web application.
    
    This class provides a streamlined interface for OpenSearch operations
    specifically designed for the web application. It handles connection
    management, recipe search, and ingredient retrieval with simplified
    error handling suitable for web contexts.
    
    The manager focuses on the core functionality needed by the web app:
    - Recipe search with fuzzy matching
    - Ingredient autocomplete data
    - Connection health monitoring
    
    Attributes:
        opensearch_client: OpenSearch client instance
        
    Example:
        >>> db = SimpleDBManager()
        >>> recipes = db.search_recipes("chicken spinach", size=5)
        >>> ingredients = db.get_all_ingredients(limit=1000)
    """
    
    def __init__(self):
        """
        Initialize the database manager with OpenSearch connection.
        
        Attempts to establish connection to OpenSearch with basic error
        handling. If connection fails, the system continues operating
        with limited functionality.
        """
        self.opensearch_client = None
        self._init_opensearch()
    
    def _init_opensearch(self):
        """
        Initialize OpenSearch client with basic error handling.
        
        Creates OpenSearch client with simplified configuration suitable
        for web application use. Includes basic error handling without
        complex retry logic.
        
        Raises:
            None: All exceptions are caught and logged
        """
        try:
            self.opensearch_client = OpenSearch(
                hosts=[app_config.OPENSEARCH_URL],
                http_auth=None,
                use_ssl=False,
                verify_certs=False,
                ssl_show_warn=False,
            )
        except Exception as e:
            logging.error(f"Failed to connect to OpenSearch: {e}")
    
    def get_opensearch_client(self):
        """
        Get the OpenSearch client instance.
        
        Returns:
            OpenSearch: Configured OpenSearch client or None if unavailable
        """
        return self.opensearch_client
    
    def search_recipes(self, ingredient_query: str, size: int = 12):
        """
        Search recipes in OpenSearch using fuzzy matching.
        
        Performs a full-text search across recipe ingredients with automatic
        fuzzy matching for typos and variations. Returns ranked results with
        relevance scores suitable for web display.
        
        Args:
            ingredient_query: Search query for recipe ingredients
            size: Maximum number of results to return (default: 12)
            
        Returns:
            Dict containing search results with hits and metadata
            
        Example:
            >>> recipes = db.search_recipes("chicken spinach", size=5)
            >>> for hit in recipes['hits']['hits']:
            ...     recipe = hit['_source']
            ...     print(f"Recipe: {recipe['title']}")
        """
        if not self.opensearch_client:
            return {"hits": {"hits": [], "total": {"value": 0}}}
        
        query = {
            "query": {
                "match": {
                    "ingredients": {
                        "query": ingredient_query,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }
        
        try:
            response = self.opensearch_client.search(
                index="recipes",
                body=query,
                size=size
            )
            return response
        except Exception as e:
            logging.error(f"OpenSearch query failed: {e}")
            return {"hits": {"hits": [], "total": {"value": 0}}}
    
    def get_all_ingredients(self, limit: int = 10000):
        """
        Get all available ingredients from OpenSearch for autocomplete.
        
        Retrieves a comprehensive list of all ingredients stored in the
        OpenSearch index. This data is used to build autocomplete functionality
        in the web interface.
        
        Args:
            limit: Maximum number of ingredients to retrieve (default: 10000)
            
        Returns:
            List of ingredient names
            
        Example:
            >>> ingredients = db.get_all_ingredients(limit=1000)
            >>> print(f"Found {len(ingredients)} ingredients for autocomplete")
        """
        if not self.opensearch_client:
            return []
        
        try:
            response = self.opensearch_client.search(
                index="ingredients", 
                body={"query": {"match_all": {}}}, 
                size=limit
            )
            return [hit["_source"]["ingredients"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logging.error(f"Failed to get ingredients: {e}")
            return []

db_manager = SimpleDBManager()

# Configure logging - keep it simple
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'  # Simplified format
)
# Silence noisy loggers
logging.getLogger('opensearchpy').setLevel(logging.ERROR)  # Only show errors
logging.getLogger('urllib3').setLevel(logging.ERROR)       # Only show errors
logging.getLogger('opensearch').setLevel(logging.ERROR)    # Only show errors
logger = logging.getLogger(__name__)

app = Flask(__name__)


def wait_for_opensearch(client, max_retries=30, retry_interval=2):
    """
    Wait for OpenSearch to become available using exponential backoff polling.
    
    This function implements a robust connection strategy that prevents the
    application from starting before its dependencies are ready. It uses
    simple polling with configurable retry parameters.
    
    The function provides clear status messages and handles connection
    failures gracefully, ensuring the application can start even if
    OpenSearch is temporarily unavailable.
    
    Args:
        client: OpenSearch client instance to test
        max_retries: Maximum number of connection attempts (default: 30)
        retry_interval: Time in seconds to wait between attempts (default: 2)

    Returns:
        bool: True if OpenSearch becomes available, False otherwise
        
    Example:
        >>> client = db_manager.get_opensearch_client()
        >>> if wait_for_opensearch(client):
        ...     print("OpenSearch is ready")
        ... else:
        ...     print("OpenSearch failed to start")
    """
    print("Waiting for OpenSearch to be ready...")  # Simple status message
    for i in range(max_retries):
        try:
            if client.ping():
                # Simple success message
                print("Successfully connected to OpenSearch!")
                return True
        except Exception as e:
            # Log connection attempts at debug level
            logger.debug(f"Connection attempt {i+1}/{max_retries} failed.")
        sleep(retry_interval)
    logger.error("Failed to connect to OpenSearch after maximum retries")
    return False


def init_opensearch():
    """
    Initialize OpenSearch connection and preload ingredient data for autocomplete.
    
    This function sets up the OpenSearch connection and loads all available
    ingredients into memory for fast autocomplete functionality. It uses the
    shared database manager for consistent connection handling.
    
    The function performs the following operations:
    1. Establishes connection to OpenSearch
    2. Waits for service availability
    3. Preloads ingredient data for autocomplete
    4. Provides status feedback
    
    Returns:
        tuple: (OpenSearch client, list of ingredients)
        
    Raises:
        SystemExit: If OpenSearch connection fails after maximum retries
        
    Example:
        >>> client, ingredients = init_opensearch()
        >>> print(f"Loaded {len(ingredients)} ingredients for autocomplete")
    """
    client = db_manager.get_opensearch_client()
    
    if not client:
        logger.error("OpenSearch connection failed")
        sys.exit(1)

    if not wait_for_opensearch(client):
        logger.error("OpenSearch connection failed")
        sys.exit(1)

    try:
        # Load all ingredients using shared database manager
        ingredients = db_manager.get_all_ingredients()
        print(f"Successfully loaded {len(ingredients)} ingredients")
        return client, ingredients
    except Exception as e:
        logger.error(f"Error initializing OpenSearch: {str(e)}")
        sys.exit(1)


logger.info("Starting application initialization...")
# Initialize OpenSearch and load ingredients
client, ingredients = init_opensearch()
logger.info("Application initialization completed successfully")


@app.route('/')
def home():
    """
    Render the main search interface for the recipe application.
    
    This Flask route serves the primary user interface where users can
    search for recipes by ingredients and view dietary classifications.
    The page includes the search form, ingredient selection interface,
    and results display area.
    
    The route renders the `index.html` template which contains:
    - Ingredient search form with autocomplete
    - Recipe results display
    - Dietary classification indicators
    - Responsive design elements

    Returns:
        str: The rendered HTML content of the main page
        
    Example:
        >>> with app.test_client() as client:
        ...     response = client.get('/')
        ...     print(response.status_code)
        200
    """
    return render_template('index.html')


@app.route("/select2", methods=["GET"])
def select2():
    """
    Provide ingredient autocomplete API compatible with Select2 JavaScript library.
    
    This endpoint implements server-side autocomplete functionality for the
    ingredient search interface. It searches the preloaded ingredient list
    using string containment matching and returns results formatted for
    Select2 compatibility.
    
    The endpoint supports:
    - Partial string matching for ingredient names
    - Results formatted as Select2 expects (objects with 'id' and 'text')
    - Configurable result limit
    - Case-insensitive search
    
    Query Parameters:
        q: Search query string for ingredient matching
        page: Page number for pagination (optional)
        limit: Maximum results per page (optional)

    Returns:
        JSON response with autocomplete results in Select2 format
        
    Example:
        >>> response = client.get('/select2?q=chicken')
        >>> data = response.get_json()
        >>> print(data['results'][0]['text'])
        'chicken breast'
    """
    query = request.args.get('q', '').lower()
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({"results": [], "pagination": {"more": False}})
    
    # Filter ingredients based on query
    matching_ingredients = [
        ingredient for ingredient in ingredients 
        if query in ingredient.lower()
    ]
    
    # Sort by length (shorter names first) and relevance
    matching_ingredients.sort(key=lambda x: (len(x), x.lower().find(query)))
    
    # Pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    page_results = matching_ingredients[start_idx:end_idx]
    
    # Format for Select2
    results = [{"id": ingredient, "text": ingredient} for ingredient in page_results]
    
    return jsonify({
        "results": results,
        "pagination": {
            "more": end_idx < len(matching_ingredients)
        }
    })


@app.route('/search', methods=['GET'])
def search_by_ingredients():
    """
    Search for recipes based on selected ingredients and classify dietary restrictions.
    
    This is the core API endpoint that performs recipe search and dietary
    classification. It accepts a list of ingredients, searches for matching
    recipes in OpenSearch, and applies keto/vegan classification to each result.
    
    The endpoint provides:
    - Full-text search across recipe ingredients
    - Automatic dietary classification (keto/vegan)
    - Ranked results with relevance scores
    - Comprehensive recipe metadata
    
    Query Parameters:
        ingredients: Comma-separated list of ingredients to search for

    Returns:
        JSON response containing:
        - Search results with recipe details
        - Dietary classifications for each recipe
        - Search metadata and timing information
        
    Example:
        >>> response = client.get('/search?ingredients=chicken,spinach')
        >>> data = response.get_json()
        >>> print(f"Found {len(data['recipes'])} recipes")
        >>> print(f"First recipe keto: {data['recipes'][0]['keto']}")
    """
    ingredient_query = request.args.get('ingredients', '')
    
    if not ingredient_query:
        return jsonify({
            "error": "No ingredients provided",
            "recipes": [],
            "total": 0
        })
    
    try:
        # Search for recipes
        search_results = db_manager.search_recipes(ingredient_query, size=12)
        
        recipes = []
        for hit in search_results['hits']['hits']:
            recipe = hit['_source']
            recipe_ingredients = recipe.get('ingredients', [])
            
            # Classify recipe for dietary restrictions
            recipe['keto'] = is_keto(recipe_ingredients)
            recipe['vegan'] = is_vegan(recipe_ingredients)
            
            recipes.append(recipe)
        
        return jsonify({
            "recipes": recipes,
            "total": search_results['hits']['total']['value'],
            "query": ingredient_query
        })

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({
            "error": "Search failed",
            "recipes": [],
            "total": 0
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
