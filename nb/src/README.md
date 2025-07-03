# Core System Modules - nb/src/

This directory contains the core application logic for the dietary classification system. The modules are organized into distinct layers following modern software architecture principles.

## Architecture Overview

The system follows a **function-calling RAG pipeline** architecture with cache-first performance optimization:

```
User Input → Function Calling → Query Engine → Database → Context Assembly → LLM Judge → Classification
     ↑                                                                                        ↓
Cache Check ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← Cache Store
```

## Core Modules

### Configuration & Infrastructure

#### `config.py` - Central Configuration Management
**Purpose:** Single source of truth for all system configurations.

**Key Features:**
- Database paths and connection settings
- Model names and parameters (Arctic, Qwen)
- Performance thresholds (keto carb limits, timeouts)
- Environment-based configuration support

**Usage:**
```python
from config import app_config
print(app_config.DB_PATH)  # Path to knowledge_graph.db
print(app_config.KETO_CARBS_THRESHOLD)  # 10g default
```

---

#### `database.py` - Unified Database Manager
**Purpose:** Provides clean, unified interface for all database interactions.

**Key Features:**
- SQLite connection management for knowledge_graph.db
- OpenSearch client for recipe data (~220k recipes)
- Connection pooling and error handling
- Context managers for safe resource cleanup

**Usage:**
```python
from database import db_manager

# SQLite access
with db_manager.get_sqlite_connection() as conn:
    # Your queries here
    
# OpenSearch access  
client = db_manager.get_opensearch_client()
```

---

### Core Classification Pipeline

#### `diet_classifiers.py` - Main Entry Points
**Purpose:** The submission interface - contains the required `is_keto()` and `is_vegan()` functions.

**Key Features:**
- Handles multiple input formats (JSON string, CSV string, Python list)
- Integrates with cache-aware classifier for performance
- Provides simple boolean outputs as required
- Professional error handling and logging

**Usage:**
```python
from diet_classifiers import is_keto, is_vegan

# All these formats work:
is_keto('["chicken", "spinach", "oil"]')  # JSON string
is_keto("chicken, spinach, oil")          # CSV string  
is_keto(["chicken", "spinach", "oil"])    # Python list

result = is_vegan(["quinoa", "black beans", "avocado"])  # True
```

---

#### `context_aware_classifier.py` - Pipeline Orchestrator
**Purpose:** The brain of the operation - orchestrates the complete classification pipeline with caching.

**Key Features:**
- **Cache-first approach**: Checks Redis before expensive LLM operations
- **Ingredient-level caching**: Reuses results for common ingredients
- **Recipe-level caching**: Stores complete recipe classifications
- **Performance tracking**: Monitors cache hit rates and processing times
- **Graceful fallback**: Works without cache (slower but functional)

**Architecture:**
1. Check cache for existing results
2. If cache miss: gather factual context via function calling
3. Use LLM judge for final classification
4. Cache results for future use

**Usage:**
```python
from context_aware_classifier import ContextAwareDietClassifier

classifier = ContextAwareDietClassifier()

# Single ingredient (with caching)
result = await classifier.classify_single_ingredient("chicken breast")

# Full recipe (with recipe-level caching)
result = await classifier.classify_recipe(["chicken", "spinach", "oil"])

# Performance stats
stats = classifier.get_performance_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

---

#### `function_calling_handler.py` - The Fact Retriever
**Purpose:** Converts natural language queries to structured JSON queries using the Arctic model.

**Key Features:**
- **Safe query generation**: Produces JSON instead of risky SQL
- **Structured output**: Well-defined query types and parameters
- **Error handling**: Graceful fallback for invalid queries
- **Model abstraction**: Clean interface to Arctic Text2SQL model

**Query Types:**
- `nutrition`: Get nutritional data for ingredients
- `vegan_status`: Check vegan/animal product status
- `unit_conversion`: Convert between measurement units

**Usage:**
```python
from function_calling_handler import FunctionCallingHandler

handler = FunctionCallingHandler()
query = await handler.generate_structured_query("Is butter vegan?")
# Returns: {"query_type": "vegan_status", "ingredient_name": "butter"}
```

---

#### `query_engine.py` - Safe Query Execution
**Purpose:** Translates structured JSON queries to safe, parameterized SQL and executes them.

**Key Features:**
- **SQL injection prevention**: Uses parameterized queries only
- **Query validation**: Rejects unsafe operations
- **Error recovery**: Handles database errors gracefully
- **Performance optimization**: Efficient query construction

**Safety Features:**
- No dynamic SQL construction
- Parameter binding for all values
- Whitelist-based query validation
- Comprehensive logging

**Usage:**
```python
from query_engine import QueryEngine

engine = QueryEngine()
json_query = {"query_type": "nutrition", "ingredient_name": "chicken"}
result = await engine.execute_structured_query(json_query)
```

---

### Communication & Caching

#### `llm_client.py` - LLM Communication Manager
**Purpose:** Unified client for interacting with Ollama-served models (Arctic, Qwen).

**Key Features:**
- **Multi-model support**: Handles both Arctic and Qwen models
- **Async operations**: Non-blocking model calls
- **Model availability checking**: Validates models before use
- **JSON parsing**: Automatic response parsing for structured outputs
- **Error handling**: Comprehensive failure recovery

**Supported Models:**
- `snowflake/arctic-text2sql-r1-7b:latest` - Function calling
- `qwen/qwen3-0.6b-gguf:q8_0` - Final classification judgment

**Usage:**
```python
from llm_client import llm_client

# Check model availability
if llm_client.is_model_available("qwen/qwen3-0.6b-gguf:q8_0"):
    result = await llm_client.query_async(model_name, prompt, as_json=True)
```

---

#### `utils/cache_manager.py` - Redis Cache Management
**Purpose:** Thread-safe Redis cache manager with automatic fallback capabilities.

**Key Features:**
- **Singleton pattern**: Single cache instance across application
- **Graceful degradation**: Works without Redis (logs warnings)
- **Automatic serialization**: JSON handling for complex objects
- **TTL management**: Configurable expiration times
- **Performance monitoring**: Cache statistics and health checks

**Cache Types:**
- **Ingredient context**: TTL 7 days (ingredients don't change often)
- **Recipe classifications**: TTL 1 day (recipes may have variations)
- **Performance metadata**: Timestamps, confidence scores, cache versions

**Usage:**
```python
from utils.cache_manager import get_cache_manager

cache = get_cache_manager()

if cache.is_available():
    # Store classification result
    cache.set_ingredient_context("chicken", result, ttl=604800)
    
    # Retrieve cached result
    cached = cache.get_ingredient_context("chicken")
    
    # Get statistics
    stats = cache.get_stats()
```

---

### Enhanced Processing

#### `ingredient_processor/` - Enhanced Ingredient Processing
**Purpose:** Provides comprehensive ingredient processing with normalization, fuzzy matching, and database access.

**Key Features:**
- **Normalization**: Consistent ingredient name formatting
- **Fuzzy matching**: Handles variations in ingredient names
- **Database integration**: Loads 8,789 ingredient names from knowledge DB
- **Cache-ready output**: Generates consistent cache keys
- **Confidence scoring**: Quality metrics for matches

**Components:**
- `processor.py`: Main processing logic with `EnhancedIngredientProcessor`
- `__init__.py`: Clean exports for easy importing

---

### Testing & Validation

#### `tests/` - Comprehensive Test Suite
**Purpose:** Production-ready testing using pytest and deepeval frameworks.

**Test Categories:**

**Integration Tests** (`test_integration.py`):
- Module import validation
- Database connectivity
- LLM model availability
- Cache manager functionality
- End-to-end pipeline validation

**DeepEval Pipeline Tests** (`test_deepeval_pipeline.py`):
- Function calling handler accuracy
- Query engine safety validation
- Context retrieval relevancy
- Classification factual consistency
- Hallucination detection
- End-to-end pipeline coherence

**Component Tests** (subdirectories):
- `context_engine/`: Context assembly validation
- `data_ingestion/`: Data quality checks
- `evaluation/`: Accuracy measurement
- `ground_truth/`: Ground truth generation validation

**Usage:**
```bash
# Run all tests
pytest nb/src/tests/ -v

# Run specific test categories
pytest nb/src/tests/test_integration.py -v
pytest nb/src/tests/test_deepeval_pipeline.py -v

# Run with coverage
pytest nb/src/tests/ --cov=nb/src --cov-report=html
```

---

### Data & Evaluation

#### `data/` - Processed Data Storage
**Purpose:** Houses the knowledge database and derived datasets.

**Contents:**
- `knowledge_graph.db`: Main SQLite database (15+ MB)
- `performance_results/`: Benchmark outputs and visualizations
- `ground_truth/`: Validation datasets
- `cache_snapshots/`: Redis backup files (if applicable)

#### `eval_data/` - Evaluation Datasets
**Purpose:** Curated test cases for accuracy validation.

**Datasets:**
- `ground_truth_sample.csv`: General test cases
- `strict_keto.csv`: Clear keto-compliant recipes
- `strict_vegan.csv`: Clear vegan recipes  
- `borderline_keto.csv`: Edge cases for keto classification

#### `raw_data/` - Source Data Files
**Purpose:** Original CSV files used to build knowledge database.

**Files:**
- `nutrition.csv`: USDA nutritional data (8,789 records)
- `vegan_ontology.csv`: Vegan/animal product classifications
- `unit_conversion.csv`: Measurement unit conversions

---

## Performance Characteristics

### Latency Targets
- **Cold classification**: ~2-5 seconds (first-time ingredient)
- **Warm classification**: ~50-200ms (cached ingredient)
- **Recipe classification**: ~1-10 seconds (depending on cache hits)

### Throughput Targets
- **Minimum**: 2 recipes/second sustained
- **With cache**: 10+ recipes/second for known ingredients
- **Memory usage**: <2GB peak during processing

### Cache Performance
- **Target hit rate**: >70% in production
- **Storage efficiency**: ~1KB per cached classification
- **TTL strategy**: 7 days (ingredients), 1 day (recipes)

## Development Workflow

### Setup
1. Ensure Redis is running (optional but recommended)
2. Run knowledge base creation: `python scripts/01_ingest_data.py`
3. Validate data quality: `python scripts/02_validate_data_quality.py`
4. Run tests: `pytest nb/src/tests/ -v`

### Making Changes
1. Update relevant module in `nb/src/`
2. Run unit tests: `pytest nb/src/tests/test_<module>.py`
3. Run integration tests: `pytest nb/src/tests/test_integration.py`
4. Validate performance: `python scripts/04_run_performance_benchmarks.py`

### Debugging
1. Check logs: All modules use `logging` with professional formatting
2. Monitor cache: Use `cache_manager.get_stats()` for performance insights
3. Trace pipeline: Each component logs key decisions and timings
4. DeepEval metrics: Use `pytest` with deepeval for detailed failure analysis

## Production Deployment

### Requirements
- Python 3.8+ with dependencies from `setup.py`
- Redis server for caching (6379/tcp)
- SQLite database with write permissions
- Ollama with required models loaded

### Configuration
- Set environment variables for Redis connection
- Configure database paths in `config.py`
- Ensure model availability in Ollama
- Pre-populate cache with `scripts/05_precompute_classifications.py`

### Monitoring
- Monitor cache hit rates via `cache_manager.get_stats()`
- Track latency and error rates through application logs
- Use performance benchmarks to validate SLA compliance
- Regular validation with accuracy test scripts

This architecture provides a robust, scalable foundation for dietary classification with modern AI techniques, comprehensive testing, and production-ready performance characteristics. 