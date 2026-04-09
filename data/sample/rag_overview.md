# Retrieval-Augmented Generation (RAG)

## Introduction

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by providing them with relevant external knowledge at inference time. Instead of relying solely on the parametric knowledge stored in the model's weights during training, RAG systems retrieve relevant documents from an external knowledge base and include them in the prompt context. This approach was first formalized by Lewis et al. in 2020.

## Why RAG Matters

### Reducing Hallucinations

LLMs are prone to generating plausible-sounding but factually incorrect information, known as hallucinations. By grounding generation in retrieved documents, RAG significantly reduces hallucination rates. The model can cite specific sources and is less likely to fabricate information when relevant context is available.

### Knowledge Currency

LLMs have a knowledge cutoff date determined by their training data. RAG enables models to access up-to-date information without expensive retraining. This is crucial for applications requiring current information, such as news summarization, customer support, and technical documentation.

### Domain Specialization

RAG allows a general-purpose LLM to specialize in a particular domain by providing access to domain-specific documents. This is often more cost-effective and flexible than fine-tuning, as the knowledge base can be updated independently of the model.

## RAG Pipeline Architecture

### Document Ingestion

The first stage involves loading and preprocessing documents from various sources (PDFs, web pages, databases, APIs). Documents are cleaned, normalized, and prepared for chunking. Common preprocessing steps include removing headers/footers, handling tables, and extracting text from different formats.

### Chunking

Documents are split into smaller segments (chunks) that can be independently embedded and retrieved. The chunk size is a critical parameter: chunks too small may lack sufficient context, while chunks too large may dilute the relevance signal. Common strategies include fixed-size chunking, recursive character splitting, semantic chunking, and sentence-window approaches.

### Embedding

Each chunk is converted into a dense vector representation using an embedding model. Popular choices include sentence-transformers models like all-MiniLM-L6-v2 (lightweight, 384 dimensions) and all-mpnet-base-v2 (higher quality, 768 dimensions). The choice of embedding model affects both retrieval quality and computational cost.

### Vector Storage

Chunk embeddings are stored in a vector database or index for efficient similarity search. Options range from simple FAISS indices to managed services like Pinecone, Weaviate, or Milvus. The choice depends on scale requirements, update frequency, filtering needs, and deployment constraints.

### Retrieval

At query time, the user's question is embedded using the same model, and the most similar chunks are retrieved from the vector store. Common retrieval methods include cosine similarity search, maximum marginal relevance (MMR) for diversity, and hybrid search combining dense and sparse (BM25) retrieval.

### Generation

Retrieved chunks are formatted into a prompt alongside the user's question and sent to the LLM. The prompt typically instructs the model to answer based only on the provided context, cite sources, and indicate when the context is insufficient to answer the question.

## Advanced RAG Techniques

### Query Transformation

Before retrieval, the query can be transformed to improve results. Techniques include query expansion (adding related terms), hypothetical document embedding (HyDE, generating a hypothetical answer to embed instead of the question), and multi-query generation (creating multiple query variants to retrieve a broader set of documents).

### Re-Ranking

After initial retrieval, a cross-encoder re-ranker can score each document-query pair more accurately than the bi-encoder used for initial retrieval. Cross-encoders like ms-marco-MiniLM-L-6-v2 process the query and document jointly, producing more nuanced relevance scores at the cost of higher latency.

### Contextual Compression

Retrieved documents can be compressed to include only the most relevant portions. This reduces token usage and noise in the prompt, improving both generation quality and cost efficiency.

### Self-RAG and Corrective RAG

Self-RAG introduces a self-reflection mechanism where the model evaluates whether retrieval is needed, whether retrieved documents are relevant, and whether the generated response is supported by the evidence. Corrective RAG (CRAG) adds a lightweight evaluator that triggers web search as a fallback when retrieved documents score below a confidence threshold.

## Evaluation

RAG systems are evaluated on multiple dimensions: retrieval quality (precision, recall, NDCG), generation quality (faithfulness, relevance, completeness), and end-to-end performance (answer correctness). Frameworks like RAGAS provide automated metrics for these dimensions, though human evaluation remains important for nuanced assessment.
