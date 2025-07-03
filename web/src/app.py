"""Flask web application for recipe search and dietary classification.

This application serves a web interface that allows users to search for recipes
by ingredients. It uses an OpenSearch backend for efficient full-text search
and integrates custom classifiers to determine if recipes meet dietary
requirements like 'keto' or 'vegan'.

The application is structured with the following endpoints:
    - `/`: Renders the main search page (`index.html`).
    - `/select2`: An API endpoint for ingredient autocomplete, compatible with
      the Select2 JavaScript library.
    - `/search`: The core API endpoint that performs recipe searches based on
      user-selected ingredients and returns classified results.

The application initializes a connection to OpenSearch on startup and pre-loads
all available ingredients into memory to provide fast autocomplete suggestions.
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
    OPENSEARCH_URL = "http://localhost:9200"

app_config = WebConfig()

# Simple database manager for web app
from opensearchpy import OpenSearch
import logging

class SimpleDBManager:
    def __init__(self):
        self.opensearch_client = None
        self._init_opensearch()
    
    def _init_opensearch(self):
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
        return self.opensearch_client
    
    def search_recipes(self, ingredient_query: str, size: int = 12):
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
    """Waits for OpenSearch to become available using exponential backoff polling.

    Detailed Description:
        - This function polls OpenSearch for availability using `ping()`.
        - It prevents the application from starting before its dependencies are ready.

    Parameters:
        - client (opensearchpy.OpenSearch): The configured OpenSearch client.
        - max_retries (int): Maximum connection attempts.
        - retry_interval (int): Time in seconds to wait between attempts.

    Returns:
        - bool: True if OpenSearch is available, False otherwise.
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
    """Initializes OpenSearch connection and preloads ingredient data for autocomplete.

    Uses the shared database manager for consistent connection handling.
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
    """Renders the main search interface for the recipe application.

    Detailed Description:
        - This Flask route serves the primary user interface.
        - It renders the `index.html` template, which contains the search form,
          ingredient selection, and results display area.

    Returns:
        - str: The rendered HTML content of the main page.
    """
    return render_template('index.html')


@app.route("/select2", methods=["GET"])
def select2():
    """Provides ingredient autocomplete API compatible with Select2 JavaScript library.

    Detailed Description:
        - This endpoint implements the server-side autocomplete.
        - It searches the preloaded ingredient list using string containment matching.
        - Results are formatted for Select2 (objects with 'id' and 'text').
        - Results are sorted by ingredient name length.

    Parameters (via query string):
        - q (str): The search query string.

    Returns:
        - flask.Response: JSON response with 'results' array.
    """
    q = request.args.get("q", "").strip()
    results = [{"id": id_, "text": txt_}
               for id_, txt_ in enumerate(ingredients) if q in txt_]
    results = sorted(results, key=lambda x: len(x["text"]))
    return jsonify({"results": results})


@app.route('/search', methods=['GET'])
def search_by_ingredients():
    """Executes recipe search with real-time dietary classification.

    Detailed Description:
        - This endpoint orchestrates the recipe discovery process.
        - It parses ingredient IDs and maps them to names.
        - It constructs a fuzzy `match` query for OpenSearch.
        - It applies `is_keto` and `is_vegan` classifiers to each result.
        - It formats results with recipe metadata and classification flags.

    Parameters (via query string):
        - q (str): Space-separated ingredient IDs.

    Returns:
        - flask.Response: JSON response with search results and classifications,
          or error message with appropriate HTTP status code.

    Raises:
        - 400: If no ingredient query is provided.
        - 500: If OpenSearch query execution fails.

    Examples:
        >>> # GET /search?q=123 456 789
        >>> {
        ...   "total": 25,
        ...   "results": [
        ...     {
        ...       "title": "Keto Chicken Salad",
        ...       "keto": true,
        ...       "vegan": false,
        ...       "score": 8.5
        ...     }
        ...   ]
        ... }
    """
    ingredient = request.args.get('q', '')
    if not ingredient:
        return jsonify({'error': 'Please provide an ingredient name'}), 400

    ingredient_ids = [int(id_) for id_ in ingredient.split() if id_.isdigit()]
    ingredient_ids = [ingredients[id_] for id_ in ingredient_ids]
    ingredient = " ".join(ingredient_ids)

    try:
        # Execute the search using shared database manager
        response = db_manager.search_recipes(ingredient, size=12)

        # Format the results
        hits = response['hits']['hits']
        results = [{
            'title': hit['_source']['title'],
            'description': hit['_source'].get('description', ''),
            'ingredients': hit['_source']['ingredients'],
            'instructions': hit['_source'].get('instructions', ''),
            'photo_url': hit['_source'].get('photo_url', ''),
            'keto': is_keto(hit['_source']['ingredients']),
            'vegan': is_vegan(hit['_source']['ingredients']),
            'score': hit['_score']
        } for hit in hits]
        return jsonify({
            'total': response['hits']['total']['value'],
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
