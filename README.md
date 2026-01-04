# CrediTrust Intelligence: RAG-Powered Consumer Insights Engine

## 📌 Executive Summary
CrediTrust Intelligence is an advanced analytics platform designed to convert thousands of unstructured customer complaints into strategic business assets. By leveraging **Retrieval-Augmented Generation (RAG)**, the system allows stakeholders to query massive datasets using natural language to identify emerging risks, product friction points, and compliance signals in real-time.



## 🛠️ System Architecture
The platform is built on a modular AI pipeline:
1. **Data Engineering Layer:** Automated cleaning and filtering of multi-product financial narratives.
2. **Semantic Indexing:** High-dimensional vectorization using `all-MiniLM-L6-v2` and storage in **ChromaDB**.
3. **RAG Orchestration:** A context-aware retrieval system that fetches relevant narratives to ground LLM responses in factual evidence.
4. **Intelligence Interface:** A Streamlit/Gradio-based dashboard for non-technical stakeholders to perform deep-dive analysis.

## 📈 Key Capabilities
* **Cross-Product Analysis:** Compare customer sentiment across Credit Cards, Loans, and Savings.
* **Evidence-Backed Synthesis:** Every AI response is cited with specific, retrieved complaint excerpts to ensure transparency.
* **Trend Identification:** Rapidly detect systemic issues that manual reading would miss for weeks.

---

## 📂 Project Structure
* `src/`: Production-ready Python modules for data processing and indexing.
* `vector_store/`: Persistent vector database containing semantic embeddings.
* `notebooks/`: Exploratory analysis and statistical distribution of consumer feedback.
* `app.py`: The entry point for the intelligent chat interface.
