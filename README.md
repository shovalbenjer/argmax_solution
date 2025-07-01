# 🥑 Search By Ingredients Challenge
![Argmax](https://argmaxml.com/wp-content/uploads/2024/04/Argmax_logo_inline.svg)

### **Project Overview**

This project implements a sophisticated, multi-tiered system to classify recipes as **Vegan** and **Keto-friendly**. The architecture is designed for maximum performance, accuracy, and full traceability by leveraging a high-performance Python stack (`uv`, `Polars`, `Numba`), advanced models for parsing and data retrieval, and a dual-LLM strategy managed by **Ollama** and tracked with **MLflow**.

### **Core Technology Stack**

*   **Package Management (`uv`):** We use `uv`, a high-performance Python package installer and resolver, to ensure rapid and reliable dependency management.
*   **Data Manipulation (`Polars`):** All data processing is handled by Polars, a lightning-fast DataFrame library written in Rust. It provides a more memory-efficient and performant alternative to pandas.
*   **Performance (`Numba`):** Critical numerical functions are JIT-compiled with Numba, translating Python code into optimized machine code for C-like speed.
*   **LLM Serving (`Ollama`):** The local "Student" LLM is managed and served by Ollama, which simplifies the deployment and integration of large language models.
*   **Experiment Tracking (`MLflow`):** All ground truth generation and evaluation runs are tracked with MLflow for a complete audit trail and reproducibility.

### **Key Features & Architecture: The Hybrid Retrieval Cascade**

The system's core is a "Hybrid Retrieval Cascade," an intelligent, multi-step process for analyzing each ingredient.

*   **1. High-Accuracy Parsing:** An ingredient string is first processed by a specialized NLP model to reliably extract its core components.
*   **2. Intelligent Knowledge Retrieval:** The extracted ingredient name is then used to query our `knowledge_graph.db`. This is not a simple lookup; it's an intelligent translation step:
    *   **Text-to-SQL (`RSL-SQL`):** We use a **Robust Schema Linking (RSL-SQL)** model to translate the natural language ingredient name (e.g., "all-purpose flour") into a precise SQL query. This allows for flexible and intelligent probing of our structured knowledge graph, moving beyond simple keyword matching.
    *   **Fallback Searches:** If the Text-to-SQL model fails, the system falls back to fuzzy and exact-match searches.
*   **3. The Teacher/Student LLM Paradigm:**
    *   **"Teacher" Model (Google Gemini):** A powerful, state-of-the-art model from Google's Gemini family is used via its API to create a high-quality, nuanced ground truth dataset. This provides the "source of truth" for our evaluations.
    *   **"Student" Judge (Qwen via Ollama):** A small, fast, and efficient quantized model acts as the final judge for new recipes. It receives the rich, structured context from our retrieval cascade and makes the final classification.

### **Models & Capabilities**

*   **Parsing Model (`ingredient-parser-nlp`):**
    *   **Capability:** Extracts `name`, `quantity`, and `unit` from unstructured ingredient text.
    *   **Advantage:** Achieves **97.8% word-level accuracy**, providing a highly reliable and structured input for the rest of the pipeline, which minimizes cascading errors.

*   **Text-to-SQL Model (`rsl-text2sql`):**
    *   **Capability:** Generates complex SQL queries from natural language inputs by robustly linking terms to the database schema.
    *   **Advantage:** This is the core of our "intelligent retrieval." It allows the system to understand user-like queries about ingredients and fetch precise data from the `knowledge_graph.db`, making our analysis far more robust than simple lookups.

*   **Final "Student" Judge LLM (`Qwen3-0.6B-GGUF`):**
    *   **Capability:** A 0.6B parameter model with a 32k context length, optimized for instruction following and reasoning.
    *   **Advantage:** Run efficiently via **Ollama**, it features a unique **"thinking mode"** that can be toggled on for complex logical tasks, providing a balance between high-speed performance for simple ingredients and deep reasoning for complex ones.
