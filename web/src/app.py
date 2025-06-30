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
    """Polls the OpenSearch client to wait for the service to be ready.

    This function attempts to connect to OpenSearch by repeatedly calling the
    client's `ping()` method. It is used to ensure that the web application
    does not start before its database backend is available.

    Args:
        client (opensearchpy.OpenSearch): The OpenSearch client instance.
        max_retries (int): The maximum number of connection attempts.
        retry_interval (int): The delay in seconds between retries.

    Returns:
        bool: True if the connection succeeds, False otherwise.
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
    """Initializes the OpenSearch connection and pre-loads ingredient data.

    Called on application startup, this function configures the OpenSearch
    client, waits for the service to be available, and then loads all unique
    ingredient names from the 'ingredients' index into memory. This in-memory
    list is used to power the fast autocomplete search feature.

    Returns:
        tuple[opensearchpy.OpenSearch, list[str]]: A tuple containing the
            initialized OpenSearch client and the list of ingredient names.

    Raises:
        SystemExit: If the application fails to connect to OpenSearch after
            multiple retries.
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
    """Renders the main home page.

    This route serves the `index.html` template, which is the main user
    interface for the recipe search application.
    """
    return render_template('index.html')


@app.route("/select2", methods=["GET"])
def select2():
    """Provides a JSON API for ingredient autocomplete.

    This endpoint is designed to work with the Select2 JavaScript library. It
    searches the in-memory list of ingredients based on the query parameter 'q'
    and returns a JSON response in the format expected by Select2.

    The results are sorted by the length of the ingredient name to prioritize
    shorter, more specific matches.

    Returns:
        flask.Response: A JSON response containing the list of matching
            ingredients.
    """
    q = request.args.get("q", "").strip()
    results = [{"id": id_, "text": txt_}
               for id_, txt_ in enumerate(ingredients) if q in txt_]
    results = sorted(results, key=lambda x: len(x["text"]))
    return jsonify({"results": results})


@app.route('/search', methods=['GET'])
def search_by_ingredients():
    """Searches for recipes and returns classified results.

    This is the primary search endpoint. It receives a list of ingredient IDs,
    constructs a fuzzy `match` query for OpenSearch, and executes it against
    the 'recipes' index.

    For each search hit, it performs on-the-fly dietary classification by
    calling the `is_keto` and `is_vegan` functions. The final results, including
    recipe details and classification flags, are returned as a JSON object.

    Returns:
        flask.Response: A JSON response containing the search results, or an
            error message if the request is invalid or the search fails.
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
