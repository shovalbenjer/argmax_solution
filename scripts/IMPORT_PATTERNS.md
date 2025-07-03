# Import Patterns for search_by_ingredients Project

## Directory Structure Overview

```
search_by_ingredients/
├── shared/                  # Shared modules accessible from anywhere
├── scripts/                 # Utility scripts
├── web/src/                 # Web application
└── nb/src/                  # Notebook environment modules
    ├── ingredient_processor/
    ├── llm_handler/
    ├── ground_truth/
    ├── sql_generator/
    ├── utils/
    └── tests/
```

## Import Rules by Directory

### 1. Scripts Directory (`scripts/`)

**For importing from `shared/`:**
```python
# No sys.path modification needed as shared is at project root
from shared.config import app_config
from shared.database import db_manager
```

**For importing from `nb/src/`:**
```python
import sys
from pathlib import Path
# Add nb/src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nb" / "src"))

# Now can import nb/src modules
from ingredient_processor.processor import get_context_with_rapidfuzz_fallback
```

### 2. Web Directory (`web/src/`)

**For importing from `shared/`:**
```python
import sys
from pathlib import Path
# Go up two levels to project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.config import app_config
from shared.database import db_manager
```

### 3. Notebook Source Directory (`nb/src/`)

**For importing from `shared/`:**
```python
import sys
from pathlib import Path
# Go up two levels to project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.config import app_config
```

**For importing other nb/src modules:**
```python
# From within nb/src/module.py, no path modification needed
from ingredient_processor.processor import function_name
from llm_handler.handler import another_function
```

### 4. Subdirectories in nb/src (`nb/src/subdir/`)

**For importing from `shared/`:**
```python
import sys
from pathlib import Path
# Go up three levels to project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.config import app_config
```

**For importing other nb/src modules:**
```python
# Add nb/src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingredient_processor.processor import function_name
```

### 5. Test Files (`nb/src/tests/`)

**For importing from `shared/`:**
```python
import sys
from pathlib import Path
# Go up three levels to project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.config import app_config
```

**For importing nb/src modules:**
```python
# Already in nb/src, so parent directory works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingredient_processor.processor import function_name
```

## Common Patterns

### Path Resolution Formula
```python
# To get to project root from any file:
# Count directories from file to project root
# Use .parents[N] where N is the count

# Examples:
# scripts/script.py -> .parents[1]
# web/src/app.py -> .parents[2]
# nb/src/module.py -> .parents[2]
# nb/src/subdir/module.py -> .parents[3]
```

### Module Organization
- `shared/`: Contains reusable modules used across web and notebook
- `nb/src/`: Contains notebook-specific modules
- Modules in `nb/src/` can import each other directly
- External directories need sys.path modification to import from `nb/src/`

## Troubleshooting Import Errors

1. **ModuleNotFoundError**: Check sys.path configuration
2. **Count parent directories**: Use `Path(__file__).resolve().parents` and count levels
3. **Print sys.path**: Add `print(sys.path)` to debug path issues
4. **Use absolute imports**: Avoid relative imports across package boundaries 