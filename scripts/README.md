# Data Processing & Validation Scripts

This directory contains the sequential scripts required to build the knowledge base and validate the system's performance and accuracy. They should be run in numerical order for a complete system setup and evaluation.

## Execution Order

### 1. `01_ingest_data.py` - Knowledge Base Construction
**Purpose:** The master script for offline data pipeline. Creates the complete `knowledge_graph.db` from raw data sources.

**What it does:**
- Initializes SQLite database schema (nutrition_facts, vegan_ontology, unit_conversions)
- Ingests core data from CSVs in `nb/src/raw_data/` 
- Connects to OpenSearch and fetches ~220k recipes
- Parses recipe ingredients using `ingredient-parser-nlp`
- Stores structured, enriched recipe data for function-calling pipeline

**Usage:**
```bash
cd nb
python ../scripts/01_ingest_data.py
```

**Output:** Creates `nb/src/data/knowledge_graph.db` (~15MB with 8,789+ nutrition records)

---

### 2. `02_validate_data_quality.py` - Data Quality Assurance
**Purpose:** Validates that data ingestion completed successfully with required benchmarks.

**What it does:**
- Checks database file existence and size (>5MB expected)
- Validates table structure and record counts
- Ensures data quality (nutrition values, no null names, etc.)
- Reports statistics and identifies potential issues

**Benchmarks:**
- nutrition_facts: ≥8,000 records
- vegan_ontology: ≥200 records  
- unit_conversions: ≥15 records
- Valid calories: ≥80% of records

**Usage:**
```bash
python scripts/02_validate_data_quality.py
```

---

### 3. `03_validate_classification_accuracy.py` - Accuracy Validation
**Purpose:** Tests the diet classification pipeline against ground truth data for accuracy and performance.

**What it does:**
- Tests individual ingredient classification (15 test cases)
- Tests full recipe classification (7 test recipes)
- Validates input format compatibility (JSON, CSV, List)
- Measures processing times and calculates F1 scores

**Benchmarks:**
- Keto accuracy: ≥75%
- Vegan accuracy: ≥80%
- Processing time: ≤5s per ingredient, ≤30s per recipe
- Format compatibility: Required

**Usage:**
```bash
python scripts/03_validate_classification_accuracy.py
```

---

### 4. `04_run_performance_benchmarks.py` - Performance Testing
**Purpose:** Comprehensive performance testing under load conditions.

**What it does:**
- Measures cold vs warm performance (cache impact)
- Tests throughput (recipes/second) and latency (ms/recipe)
- Monitors memory usage during processing
- Validates cache hit rates and Redis performance
- Tests different input formats

**Benchmarks:**
- Throughput: ≥2.0 recipes/second
- Average latency: ≤500ms
- Memory usage: ≤2GB peak
- Error rate: ≤5%
- P95 latency: ≤1000ms

**Usage:**
```bash
python scripts/04_run_performance_benchmarks.py
```

---

### 5. `05_precompute_classifications.py` - Offline Optimization
**Purpose:** Pre-computes classifications for all known ingredients and sample recipes, storing results in Redis cache for instant lookup.

**What it does:**
- Processes all ingredients from nutrition database
- Classifies sample recipes from OpenSearch
- Caches results in Redis with appropriate TTL
- Provides dramatic performance improvement for known items

**Options:**
```bash
# Full pre-computation (ingredients + recipes)
python scripts/05_precompute_classifications.py

# Only ingredients
python scripts/05_precompute_classifications.py --ingredients-only

# Custom batch size
python scripts/05_precompute_classifications.py --batch-size 25

# Dry run (show what would be processed)
python scripts/05_precompute_classifications.py --dry-run
```

**Performance Impact:** Reduces classification time from ~2-5s to ~50ms for cached items.

---

## Prerequisites

### System Requirements
- Python 3.8+
- Redis server (for caching, optional but recommended)
- SQLite3
- ~16GB RAM recommended for full processing
- ~2GB disk space for knowledge database

### Python Dependencies
All dependencies are managed in `nb/setup.py`. Key requirements:
- `polars` - High-performance data processing
- `ingredient-parser-nlp` - Ingredient parsing
- `redis` - Caching layer
- `opensearch-py` - Recipe data access
- `sklearn` - Accuracy metrics (optional)

### OpenSearch Configuration
Scripts expect OpenSearch to be available with a `recipes` index. Configure connection details in `nb/src/config.py`.

## Troubleshooting

### Common Issues

**Database Not Found:**
```
FileNotFoundError: Database not found at nb/src/data/knowledge_graph.db
```
**Solution:** Run `01_ingest_data.py` first to create the knowledge base.

**Redis Connection Failed:**
```
Redis connection failed: [Errno 111] Connection refused
```
**Solution:** Start Redis server or run without caching (performance will be slower).
```bash
# Start Redis with Docker
docker run -d -p 6379:6379 redis:alpine
```

**OpenSearch Unavailable:**
```
OpenSearch client is None (service may not be running)
```
**Solution:** Ensure OpenSearch is running and accessible. Scripts will continue with reduced functionality.

**Memory Issues:**
```
MemoryError during processing
```
**Solution:** Reduce batch sizes in scripts or increase system RAM.

## Expected Results

After running all scripts successfully, you should have:

1. ✅ Complete knowledge database with 8,000+ ingredients
2. ✅ Classification accuracy ≥80% (vegan) and ≥75% (keto)  
3. ✅ Performance ≥2 recipes/second with <500ms average latency
4. ✅ Cached classifications for instant lookup
5. ✅ Full validation reports and performance metrics

## Integration with Notebook

These scripts provide the foundation for the analysis in `nb/src/task.ipynb`. The notebook demonstrates:
- System architecture and design decisions
- End-to-end classification pipeline
- Performance analysis and visualization
- Component-level testing with deepeval

Run the scripts first, then explore the notebook for detailed analysis and insights. 