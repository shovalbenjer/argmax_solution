Phase 1: Foundational Setup & Knowledge Base Reconstruction
Goal: Establish a clean, reproducible local environment and rebuild the core data asset. This demonstrates that your solution is robust and not dependent on pre-existing artifacts.
Start Docker Services: The entire backend (Ollama for LLMs, Redis for caching) runs in Docker. This is the first step.
Action: In your terminal, run:
Apply to context_awar...
Run
Reasoning: This starts all necessary services in the background and rebuilds the images if needed. This is the standard first step for any developer joining the project, as outlined in the official repository.
Rebuild the Knowledge Database: You mentioned the database was deleted. This is a perfect opportunity to demonstrate the automated data ingestion pipeline.
Action: Run the master ingestion script.
Apply to context_awar...
Run
Reasoning: This script is the heart of the offline system. It ingests raw data, connects to OpenSearch for recipes, and creates the structured knowledge_graph.db. Running this successfully proves the data pipeline is fully functional.
Validate Data Quality: Never trust a data pipeline without verification. This step shows a commitment to data integrity.
Action: Run the data quality validation script.
Apply to context_awar...
Run
Reasoning: This script checks the newly created database against defined benchmarks (row counts, null values, etc.). It produces a report confirming that the knowledge base is sound, a critical step before any modeling or classification.
Phase 2: System Validation & Addressing Edge Cases
Goal: Prove the system's accuracy and performance, with a special focus on demonstrating a deep understanding of the problem's complexities by tackling the edge_cases.txt file head-on.
Run Accuracy & Performance Benchmarks: First, confirm the system meets its baseline KPIs.
Action: Execute the validation scripts in order.
Apply to context_awar...
Run
Reasoning: This validates the core claims of the system: it's accurate according to the ground truth data, and it's fast. As a consultancy, Argmax values solutions that are not just theoretically sound but also production-ready.
Pre-populate the Cache: Demonstrate the full performance optimization strategy.
Action: Run the pre-computation script.
Apply to context_awar...
Run
Reasoning: This populates the Redis cache with pre-computed classifications. This step is crucial for showcasing the cache-first architecture and achieving the sub-second latency targets for known ingredients, a key feature for real-world applications.
Address edge_cases.txt in the Test Suite: This is where you demonstrate the "intellectual curiosity" Argmax is looking for. Instead of just mentioning the edge cases, we will build a dedicated test suite for them.
Action (Plan): Create a new test file: nb/src/tests/test_edge_cases.py. I will provide the content for this file, which will use pytest to specifically test the scenarios outlined in edge_cases.txt (e.g., composite ingredients, title-vs-reality mismatches, regional synonyms).
Proposed Content for nb/src/tests/test_edge_cases.py:
Apply to context_awar...
Reasoning: By creating and running these targeted tests, you are not just building a system, you are demonstrating a deep understanding of its potential failure points. This is a sign of a mature engineer and researcher.
Update task.ipynb to Run All Tests: The final notebook should be the master dashboard that runs everything and presents the results.
Action (Plan): I will add a final cell to your task.ipynb that executes the entire pytest suite, including the new test_edge_cases.py.
Proposed task.ipynb cell:
Apply to context_awar...
Reasoning: This makes the notebook a true, one-click validation tool, presenting all evidence of the system's robustness in a single, shareable format.
Phase 3: Productionizing with Docker and CI/CD
Goal: Go beyond a local project and present a professional, deployable solution using modern MLOps practices.
Finalize Docker Configuration: Ensure the Docker setup is ready for deployment.
Action (Plan): I will review docker-compose.yml and the Dockerfiles to ensure they correctly copy all necessary files (including the new scripts and tests) and set the right working directories. The setup seems robust, so no changes are likely needed, but this review is a critical pre-flight check.
Implement a CI/CD Pipeline with GitHub Actions: This is the most impressive step. We will create a workflow that automatically tests your code on every push to GitHub.
Action (Plan): Create a file at .github/workflows/main.yml.
Proposed Content for .github/workflows/main.yml:
Apply to context_awar...
Reasoning: This demonstrates a professional workflow. It ensures that any future changes are automatically validated, preventing regressions. For Argmax, this shows you think about code quality and maintainability, not just one-off solutions.
Phase 4: Final Submission
Goal: Package and submit the project according to the exact instructions provided by Argmax.
Polish the README.md: Your main project README.md is the first thing a recruiter will see.
Action (Plan): I will help you draft a clean, concise README.md that summarizes the project, highlights the advanced architecture (Function-Calling, Caching), and provides clear instructions on how to run everything.
Submit the Project: Follow the instructions from the Argmax repository to the letter.
Action:
Create a new private GitHub repository under your account.
Push your entire project, including the new .github/workflows/main.yml file, to this new repository.
Invite the user argmax2025 as a collaborator.
Fill out the official application form for your region (US or IL).
By following this plan, you will present a submission that is not only technically excellent but also demonstrates a deep, professional understanding of how to build and deploy real-world AI systems.