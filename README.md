# llm-semantic-search

This project implements a **four-stage semantic search pipeline** designed for efficient, accurate, and scalable information retrieval using Large Language Models (LLMs). The pipeline combines keyword filtering, dense vector-based retrieval using Cohere embeddings, approximate nearest neighbor (ANN) search with Annoy, external vector database support via Weaviate, and a re-ranking module based on LLM inference.


---

## Stage-wise Pipeline

### Stage 1: Keyword-Based Retrieval (BM25)

- Bag-of-Words with Term Frequency–Inverse Document Frequency (TF-IDF) weighting
- Utilizes `rank_bm25.BM25Okapi` to compute scores between the query and document tokens.
- Efficiently retrieves documents with exact or partial keyword matches.
- Helps to reduce initial search space for downstream dense retrieval.

**Concepts Involved**:
- Inverted Index
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Probabilistic IR Models

---

### Stage 2: Embedding Generation

- Embedding Type: Dense vector representations (768+ dimensional)
- Input queries and documents are embedded into high-dimensional space using `Cohere's Embed API v3`.
- Produces fixed-length semantic vectors.
- Captures contextual meaning beyond surface-level keywords.

**Concepts Involved**:
- Transformer-based Sentence Embeddings
- Transfer Learning
- Semantic Vector Space

---

### Stage 3: Dense Vector Retrieval

- Performs approximate nearest neighbor (ANN) search using **Annoy** to speed up vector similarity computations.
- Reduces latency while maintaining high accuracy.
- Retrieves the `top_k` nearest documents based on angular distance (cosine).

**Concepts Involved**:
- Vector Similarity Search
- Approximate kNN (Annoy Trees)
- Angular / Cosine Distance

---

### Stage 4: Reranking

- Uses cosine similarity or optionally a cross-encoder model to re-rank the top-k documents.
- Ensures that most semantically relevant results are pushed to the top.
- Vector DB: Weaviate
- Reranking Strategy: Hybrid mode (BM25 + vector similarity)
                      Uses Weaviate’s built-in reranker modules like rerank-cohere
- Reduces noise from ANN output
- Applies LLM-based reranking for higher semantic granularity

**Concepts Involved**:
- Dense Retrieval + Reranking Architecture (used in OpenAI, Google, etc.)
- Cosine Similarity
- Cross-Attention-based Scoring

---

## UMAP Embedding Visualization

UMAP (Uniform Manifold Approximation and Projection) is a dimension reduction technique that helps to capture the global structure while preserving local neighborhood distances. Each point in the chart corresponds to an article, and proximity indicates semantic closeness in the embedding space.
Visulize how documents cluster in embedding space, I used **UMAP (Uniform Manifold Approximation and Projection)** to reduce the embeddings to 2D for plotting.

![UMAP](charts/umap_plot.png)

---

## Technologies Used

- **BM25** (`rank_bm25`) – Probabilistic IR model for keyword scoring.
- **Cohere Embed v3** – Proprietary LLM for generating semantic sentence/document embeddings.
- **Annoy** – Approximate Nearest Neighbor search using angular distance.
- **UMAP** – Dimensionality reduction to 2D for visualization.
- **Pandas / Numpy / Scikit-learn** – Data wrangling and numerical computations.
- **Matplotlib / Altair** – Visualization of UMAP clusters.
- **Jupyter Notebooks** – Experimentation and iteration.

---

## Data Used

- **Wikipedia Articles Subset**
  - Each document contains:
    - `title`
    - `text`
    - `embedding` (generated using Cohere)
- You can replace this with any custom corpus (e.g., product descriptions, research papers, etc.)

