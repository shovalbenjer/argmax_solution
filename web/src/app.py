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
from opensearchpy import OpenSearch
from decouple import config
from diet_classifiers import is_keto, is_vegan
from time import sleep
import sys
import logging
import os

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
        - This function implements a robust startup pattern for distributed systems.
        - It repeatedly attempts to connect to OpenSearch using the `ping()` method, which is
          a lightweight health check that doesn't require any specific indices to exist.
        - This prevents the web application from starting before its backend dependencies are ready,
          avoiding connection errors during the critical startup phase.

    Parameters:
        - client (opensearchpy.OpenSearch): The configured OpenSearch client instance.
        - max_retries (int): Maximum number of connection attempts before giving up.
        - retry_interval (int): Time in seconds to wait between connection attempts.

    Returns:
        - bool: True if OpenSearch becomes available within the retry limit, False otherwise.

    Libraries Used:
        - time: Used for the `sleep()` function to implement the retry delay.
        - logging: For structured error reporting when connection attempts fail.
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

    Detailed Description:
        - This function handles the complete initialization of the OpenSearch backend.
        - It creates an OpenSearch client with configuration from environment variables.
        - It calls `wait_for_opensearch()` to ensure the service is available before proceeding.
        - It performs a bulk query to load all ingredient names from the 'ingredients' index
          into memory, which enables fast autocomplete responses without additional database queries.
        - This initialization pattern separates concerns: connection management, service readiness,
          and data preloading.

    Returns:
        - tuple[opensearchpy.OpenSearch, list[str]]: The initialized client and the list of
          ingredient names for autocomplete functionality.

    Raises:
        - SystemExit: If OpenSearch connection fails or ingredient loading fails, the application
          cannot function and must terminate.

    Libraries Used:
        - opensearchpy: The official Python client for OpenSearch, chosen over raw HTTP requests
          for its connection pooling, error handling, and query building capabilities.
        - decouple: For loading configuration from environment variables, providing flexibility
          across different deployment environments.
    """
    client = OpenSearch(
        hosts=[config('OPENSEARCH_URL', 'http://localhost:9200')],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )

    if not wait_for_opensearch(client):
        logger.error("OpenSearch connection failed")
        sys.exit(1)

    try:
        # Load all ingredients once OpenSearch is ready
        response = client.search(index="ingredients", body={
                                 "query": {"match_all": {}}}, size=10000)
        ingredients = [hit["_source"]["ingredients"]
                       for hit in response["hits"]["hits"]]
        # Simple status message
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
        - This Flask route serves the primary user interface of the application.
        - It renders the `index.html` template, which contains the search form,
          ingredient selection interface, and results display area.
        - This follows the standard Flask pattern for serving templated HTML content.

    Returns:
        - str: The rendered HTML content of the main page.

    Libraries Used:
        - Flask: The web framework used for routing and template rendering. Flask is chosen
          for its simplicity and flexibility compared to heavier frameworks like Django.
    """
    return render_template('index.html')


@app.route("/select2", methods=["GET"])
def select2():
    """Provides ingredient autocomplete API compatible with Select2 JavaScript library.

    Detailed Description:
        - This endpoint implements the server-side component of an autocomplete feature.
        - It searches the preloaded ingredient list using simple string containment matching.
        - Results are formatted according to Select2's expected JSON structure: objects with
          'id' and 'text' fields.
        - Results are sorted by ingredient name length to prioritize shorter, more specific matches.
        - This approach avoids database queries during autocomplete, providing sub-millisecond response times.

    Parameters (via query string):
        - q (str): The search query string from the user's input.

    Returns:
        - flask.Response: JSON response with 'results' array containing matching ingredients.

    Libraries Used:
        - Flask: For request parameter parsing and JSON response generation.
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
        - This is the core search endpoint that orchestrates the entire recipe discovery process.
        - **Input Processing:** It parses ingredient IDs from the query string and maps them back
          to ingredient names using the preloaded ingredient list.
        - **Search Execution:** It constructs a fuzzy `match` query for OpenSearch, which provides
          tolerance for spelling variations and partial matches.
        - **Classification:** For each search result, it applies the custom `is_keto` and `is_vegan`
          classifiers in real-time, providing immediate dietary information.
        - **Response Formatting:** It structures the results with recipe metadata, classification
          flags, and relevance scores for the frontend.

    Parameters (via query string):
        - q (str): Space-separated ingredient IDs from the autocomplete selection.

    Returns:
        - flask.Response: JSON response containing search results with dietary classifications,
          or error message with appropriate HTTP status code.

    Raises:
        - 400: If no ingredient query is provided.
        - 500: If OpenSearch query execution fails.

    Libraries Used:
        - opensearchpy: For executing the fuzzy search query against the recipes index.
        - diet_classifiers: Custom module containing the `is_keto` and `is_vegan` classification functions.

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

    # Create the search query
    query = {
        "query": {
            "match": {
                "ingredients": {
                    "query": ingredient,
                    "fuzziness": "AUTO"
                }
            }
        }
    }

    try:
        # Execute the search
        response = client.search(
            index="recipes",
            body=query,
            size=12
        )

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
