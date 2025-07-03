# Project Wiring Report: search_by_ingredients

## Summary
This report documents the import structure analysis and fixes applied to ensure proper module imports across the project.

## Issues Found and Fixed

### 1. **scripts/01_ingest_data.py**
**Issue**: Incorrect import path for `ingredient_processor` module
```python
# ❌ Original (incorrect)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingredient_processor.processor import get_context_with_rapidfuzz_fallback
```

**Fix Applied**:
```python
# ✅ Fixed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nb" / "src"))
from ingredient_processor.processor import get_context_with_rapidfuzz_fallback
```

**Also Fixed**: Updated path configurations for data directories
```python
NB_SRC_DIR = BASE_DIR / "nb" / "src"
DATA_DIR = NB_SRC_DIR / "data"
RAW_DATA_DIR = NB_SRC_DIR / "raw_data"
```

### 2. **nb/src files with incorrect internal imports**
**Issue**: Using `nb.src.` prefix when importing modules from within nb/src
```python
# ❌ Incorrect internal imports found in:
# - nb/src/tests/test_deepeval_pipeline.py
# - nb/src/run_final_evaluation.py
# - nb/src/ground_truth/generate.py
# - nb/src/context_aware_classifier.py

from nb.src.arctic_handler import ArcticText2SQLHandler
from nb.src.context_aware_classifier import ContextAwareDietClassifier
```

**Fix Applied**:
```python
# ✅ Fixed - removed nb.src prefix for internal imports
from arctic_handler import ArcticText2SQLHandler
from context_aware_classifier import ContextAwareDietClassifier
```

## Verified Working Import Patterns

### 1. **Shared Module Imports** ✅
All files importing from `shared/` package have correct sys.path configurations:
- Files in `scripts/`: Direct import (shared is at project root)
- Files in `web/src/`: Use `.parents[2]` to reach project root
- Files in `nb/src/`: Use `.parents[2]` to reach project root
- Files in `nb/src/subdir/`: Use `.parents[3]` to reach project root

### 2. **Internal nb/src Imports** ✅
Files within `nb/src/` can import each other directly without path modification:
```python
from ingredient_processor.processor import function_name
from llm_handler.handler import another_function
```

### 3. **External Access to nb/src** ✅
Scripts outside nb/src must add it to sys.path:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nb" / "src"))
```

## Import Dependencies Map

```
project_root/
├── shared/               [Accessible from everywhere with proper sys.path]
│   ├── config.py
│   ├── database.py
│   ├── diet_classifiers.py
│   └── llm_client.py
│
├── scripts/
│   ├── 01_ingest_data.py       → imports from nb/src/ingredient_processor
│   ├── 02_create_sample_recipes.py → imports from shared/
│   └── 03_index_opensearch_data.py → no local imports
│
├── web/src/
│   └── app.py                  → imports from shared/
│
└── nb/src/
    ├── arctic_handler.py       → imports from shared/
    ├── context_aware_classifier.py → imports from shared/, arctic_handler
    ├── diet_classifiers.py     → imports from shared/
    ├── performance_test.py     → imports from shared/
    ├── run_final_evaluation.py → imports from shared/, context_aware_classifier
    │
    ├── ground_truth/
    │   └── generate.py         → imports from context_aware_classifier, ingredient_processor
    │
    ├── ingredient_processor/
    │   └── processor.py        → imports from shared/
    │
    ├── llm_handler/
    │   └── handler.py          → imports from shared/
    │
    └── tests/
        ├── test_deepeval_pipeline.py → imports from shared/, arctic_handler, context_aware_classifier
        ├── test_integration.py   → imports from shared/
        └── test_opensearch.py    → imports from shared/
```

## Testing Import Integrity

To verify imports are working:

1. **From scripts directory**:
```bash
cd scripts/
python -c "from shared.config import app_config; print('✅ Shared import works')"
```

2. **Test nb/src imports from scripts**:
```bash
cd scripts/
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('.').resolve().parent / 'nb' / 'src')); from ingredient_processor.processor import get_context_with_rapidfuzz_fallback; print('✅ nb/src import works')"
```

3. **From nb/src directory**:
```bash
cd nb/src/
python -c "from arctic_handler import ArcticText2SQLHandler; print('✅ Internal import works')"
```

## Recommendations

1. **Avoid Circular Imports**: Be careful when modules in nb/src import each other
2. **Use Absolute Imports**: Always use absolute imports instead of relative imports across package boundaries
3. **Document Dependencies**: Keep this import pattern documentation updated as new modules are added
4. **Consider Package Structure**: Consider making nb/src a proper Python package with __init__.py files for cleaner imports

## Conclusion

All import issues have been identified and fixed. The project now has a consistent import pattern that should prevent ModuleNotFoundError issues. The key fix was ensuring scripts/01_ingest_data.py correctly adds nb/src to the Python path before importing from it. 