# Vector Databases and Similarity Search

## Introduction

Vector databases are specialized storage systems designed to efficiently store, index, and query high-dimensional vector embeddings. They have become a critical component in modern AI applications, particularly for semantic search, recommendation systems, and retrieval-augmented generation (RAG) pipelines.

## Similarity Metrics

### Cosine Similarity

Cosine similarity measures the angle between two vectors, ranging from -1 (opposite) to 1 (identical direction). It is defined as: cos(A, B) = (A dot B) / (||A|| * ||B||). Cosine similarity is invariant to vector magnitude, making it suitable for comparing embeddings of different lengths of text. It is the most commonly used metric for text embeddings.

### Euclidean Distance (L2)

L2 distance measures the straight-line distance between two points in vector space. It is defined as: d(A, B) = sqrt(sum((a_i - b_i)^2)). Unlike cosine similarity, L2 distance is sensitive to vector magnitude. For normalized vectors, L2 distance and cosine similarity are monotonically related.

### Inner Product (Dot Product)

The inner product is the simplest similarity measure: IP(A, B) = sum(a_i * b_i). When vectors are normalized, the inner product equals cosine similarity. Some embedding models (like OpenAI's) produce normalized vectors, making inner product sufficient.

## Indexing Algorithms

### Flat Index (Brute Force)

A flat index stores all vectors and computes distances to every vector at query time. This provides exact results but has O(n) query time, making it impractical for large collections. It serves as a baseline for evaluating approximate methods.

### IVF (Inverted File Index)

IVF partitions the vector space into Voronoi cells using k-means clustering. At query time, only vectors in the nearest cells (controlled by the nprobe parameter) are searched. This provides a tunable speed-accuracy tradeoff. Typical configurations use 100-1000 cells with nprobe set to 5-20% of the number of cells.

### HNSW (Hierarchical Navigable Small World)

HNSW builds a multi-layer graph where each layer is a navigable small-world network. Search starts from the top layer (sparse, long-range connections) and progressively moves to lower layers (dense, short-range connections). HNSW provides excellent recall (>95%) with low latency and is the default choice in many vector databases.

### Product Quantization (PQ)

PQ compresses vectors by splitting them into subvectors and quantizing each independently. This dramatically reduces memory usage (e.g., 384-dimensional float32 vectors from 1536 bytes to 48 bytes with 8-byte subquantizers) at the cost of some accuracy. PQ is often combined with IVF for large-scale deployments.

## FAISS (Facebook AI Similarity Search)

FAISS is an open-source library developed by Meta for efficient similarity search. It provides implementations of all major indexing algorithms and supports both CPU and GPU execution.

### Key Features
- Multiple index types: Flat, IVF, HNSW, PQ, and composites (e.g., IVF-PQ)
- GPU acceleration for both indexing and search
- Support for filtered search via IDSelector
- Efficient batch operations
- Memory-mapped indices for datasets larger than RAM
- Python bindings via SWIG

### Choosing an Index
For small datasets (<10K vectors), a flat index is sufficient. For medium datasets (10K-1M), IVF with flat or HNSW provide good tradeoffs. For large datasets (>1M), IVF-PQ or HNSW with PQ compression is recommended. The `index_factory` string provides a concise way to specify complex index configurations, e.g., "IVF256,PQ32" for an IVF index with 256 cells and 32-byte product quantization.

## Comparison of Vector Databases

### FAISS
Strengths: Battle-tested at Meta's scale, excellent performance, flexible index types, GPU support, no infrastructure overhead. Weaknesses: Library not a database, no built-in persistence management, no metadata filtering, requires manual sharding for distributed deployments.

### ChromaDB
Strengths: Easy to use, built-in document storage, metadata filtering, automatic persistence. Weaknesses: Less mature, limited scale testing, fewer index options, slower for large datasets.

### Pinecone
Strengths: Fully managed, automatic scaling, metadata filtering, high availability. Weaknesses: Proprietary cloud service, cost at scale, vendor lock-in, latency for non-US regions.

### Weaviate
Strengths: Rich query language (GraphQL), hybrid search, multi-modal support, modular vectorization. Weaknesses: More complex deployment, higher resource requirements.

### Milvus
Strengths: Designed for billion-scale, distributed architecture, multiple index types, active development. Weaknesses: Complex deployment, significant resource requirements for distributed mode.

## Best Practices

### Dimensionality Considerations

Higher-dimensional embeddings generally capture more semantic nuance but require more storage and compute. For most applications, 384-768 dimensions provide a good balance. Dimensionality reduction techniques like PCA can be applied post-hoc if needed.

### Index Maintenance

Vector indices may need periodic rebuilding as the data distribution changes significantly. For IVF indices, the centroids should be retrained when substantial new data is added. HNSW indices are more robust to incremental additions but cannot efficiently delete vectors.

### Hybrid Search

Combining dense vector search with sparse keyword search (BM25) often outperforms either approach alone. The fusion can be done via reciprocal rank fusion (RRF) or learned score combination. This is particularly effective for queries containing specific technical terms or proper nouns that embedding models may not handle well.
