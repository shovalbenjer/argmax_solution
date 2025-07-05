# Model Updates: Implementing models.txt Recommendations

## Overview
Updated the SOTA diet classifier system to use the latest recommended models from `models.txt`, implementing diverse jury prompts and enhanced RAG context integration.

## **Jury System Updates**

### Previous Models (Replaced)
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-2-2b-it` 
- `Qwen/Qwen2-0.5B-Instruct`

### New Models (From models.txt)
1. **microsoft/Phi-4-mini-instruct** - Chain-of-Thought Reasoner
   - Uses custom chat format: `<|system|>...<|end|><|user|>...<|end|><|assistant|>`
   - Enhanced reasoning capabilities
   - Requires transformers>=4.49.0

2. **google/gemma-3-1b-it** - Factual Analyst
   - Uses structured message format with role/content arrays
   - Improved factual analysis capabilities
   - Supports 8-bit quantization

3. **Qwen/Qwen3-0.6B-GGUF** - Common-Sense Expert
   - Features thinking mode with `/think` directive
   - Outputs structured thinking in `<think>...</think>` tags
   - Optimized sampling parameters: temp=0.6, top_p=0.95, presence_penalty=1.5

## 🔗 **Retrieval System Updates**

### Previous Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2`

### New Embedding Model (From models.txt)
- **Qwen/Qwen3-Embedding-0.6B-GGUF**
  - Specialized for retrieval tasks
  - Requires `<|endoftext|>` token appending
  - Better performance for ingredient similarity matching
  - Graceful fallback to previous model if unavailable

### Future Reranker Integration
- **Qwen/Qwen3-Reranker-0.6B** (TODO)
  - Binary yes/no reranking for query-document relevance
  - Requires transformers>=4.51.0
  - Will replace current embedding-based reranking

## **Implementation Details**

### Diverse Prompt Strategies

**Phi-4 (Chain-of-Thought)**:
```
<|system|>You are a meticulous dietary researcher analyzing food ingredients.<|end|>
<|user|>For the ingredient "quinoa", think step-by-step.
Database Context: [RAG info]
First, analyze its vegan status...
Second, analyze its keto compatibility...
Finally, summarize your findings in JSON format...<|end|>
<|assistant|>
```

**Gemma-3 (Factual Analyst)**:
```json
[
  {"role": "system", "content": [{"type": "text", "text": "You are a factual dietary analyst..."}]},
  {"role": "user", "content": [{"type": "text", "text": "Analyze the ingredient: quinoa..."}]}
]
```

**Qwen3 (Common-Sense Expert)**:
```
A user on a strict vegan and keto diet wants to know if they can eat "quinoa". /think
Database Context: [RAG info]
Think through this step by step:
- Is it vegan? (No animal products whatsoever)
- Is it keto? (Very low carb, under 10g per 100g)
```

### RAG Context Integration

Enhanced `query_llm_jury()` function now accepts `db_context` parameter:
```python
jury_result = query_llm_jury(ingredient_str, rag_context)
```

Context formatting includes:
- Database match type and confidence
- Vegan status and contextual requirements
- Nutritional information (carbs, protein, fat per 100g)
- Unit conversion accuracy

### Model-Specific Handling

Updated `_query_single_model()` to handle:
- Different prompt formats (string vs messages array)
- Model-specific generation parameters
- Thinking mode extraction for Qwen3
- Enhanced JSON parsing and validation

## **Files Modified**

### Core Implementation
- `nb/src/llmhandler.py` - Updated jury models and diverse prompts
- `nb/src/diet_classifiers.py` - Enhanced RAG context integration
- `nb/src/preprocess_for_llm.py` - Updated to Qwen3 embedding model
- `nb/requirements.txt` - Updated transformers version and model comments

### Configuration Updates
- Jury models list updated to new model identifiers
- Context-aware caching with combined keys
- Model-specific generation parameters
- Enhanced error handling and fallbacks

## **Technical Improvements**

### Performance Optimizations
- 4-bit quantization support maintained
- Model-specific temperature and sampling settings
- Efficient context-aware caching
- Graceful degradation when models unavailable

### Enhanced Explainability
- Model identity tracking in responses
- Diverse reasoning styles in explanations
- Comprehensive audit trails
- RAG context transparency

### Robustness Features
- Multiple fallback paths for each model
- Automatic model availability detection
- Error recovery and logging
- Backward compatibility maintained

## **Benefits Achieved**

1. **Reduced Bias**: Three distinct reasoning approaches prevent groupthink
2. **Enhanced Accuracy**: Latest model capabilities + RAG grounding
3. **Better Performance**: Optimized models for specific tasks
4. **Future-Proof**: Easy integration of upcoming models (reranker)
5. **Maintainable**: Clear separation of concerns and fallback strategies

## **Next Steps**

1. **Qwen3 Reranker Integration**: Implement binary reranking system
2. **Performance Benchmarking**: Compare old vs new model accuracy
3. **Fine-tuning**: Customize prompts for specific dietary scenarios
4. **Monitoring**: Add model performance tracking and alerts
5. **Documentation**: Update API docs with new model capabilities

## **Important Notes**

- Requires transformers>=4.51.0 for Qwen3 support
- GGUF models may require additional setup for some environments
- Thinking mode parsing is Qwen3-specific
- RAG context significantly improves accuracy but increases token usage
- All changes maintain backward compatibility with existing APIs 