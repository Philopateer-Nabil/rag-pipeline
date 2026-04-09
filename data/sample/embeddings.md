# Text Embeddings for NLP

## What Are Embeddings?

Text embeddings are dense vector representations of text that capture semantic meaning in a continuous vector space. Unlike sparse representations (like bag-of-words or TF-IDF), embeddings place semantically similar texts close together in the vector space, enabling efficient similarity computation. Modern embedding models produce vectors of 384 to 4096 dimensions, encoding nuanced semantic relationships.

## Evolution of Embedding Models

### Word2Vec and GloVe

Early embedding methods operated at the word level. Word2Vec (Mikolov et al., 2013) learned embeddings by predicting surrounding words (Skip-gram) or predicting a word from its context (CBOW). GloVe (Pennington et al., 2014) learned embeddings from global co-occurrence statistics. These methods produced a single fixed vector per word, unable to handle polysemy (words with multiple meanings).

### Contextual Embeddings (ELMo, BERT)

ELMo (2018) introduced context-dependent embeddings using bidirectional LSTMs, where the same word receives different embeddings based on its surrounding context. BERT (2018) advanced this with Transformer-based bidirectional encoding, producing highly contextual representations that became the foundation for modern NLP.

### Sentence Transformers

Sentence-BERT (Reimers and Gurevych, 2019) adapted BERT for producing sentence-level embeddings efficiently. Instead of using BERT's [CLS] token (which performs poorly for similarity tasks), Sentence-BERT uses siamese networks trained on natural language inference data. This produced embeddings where cosine similarity directly corresponds to semantic similarity.

## Popular Embedding Models

### all-MiniLM-L6-v2

A compact model producing 384-dimensional embeddings. It uses a 6-layer MiniLM architecture distilled from a larger model, offering an excellent speed-quality tradeoff. It processes up to 256 tokens per input and achieves strong performance on semantic textual similarity benchmarks. This is often the recommended starting point for applications where inference speed matters.

Performance characteristics: ~14M parameters, 80MB model size, encodes 14,000 sentences per second on a V100 GPU, and achieves an average score of 68.06 on the STS benchmark.

### all-mpnet-base-v2

A higher-quality model producing 768-dimensional embeddings. Based on the MPNet architecture with 12 layers, it provides better performance at the cost of higher latency and memory usage. Recommended when quality is more important than speed.

### Instructor Models

Instructor embeddings allow task-specific customization by prepending a natural language instruction to the input. For example, "Represent the document for retrieval:" produces different embeddings than "Represent the query for retrieval:". This improves performance on specific tasks without fine-tuning.

### Newer Models

BGE (BAAI General Embedding), E5 (Embeddings from Bidirectional Encoder Representations), and GTE (General Text Embeddings) represent the latest generation of embedding models. These models are trained on larger and more diverse datasets, often including instruction-aware training. The MTEB (Massive Text Embedding Benchmark) leaderboard tracks performance across 56 datasets.

## Training Embedding Models

### Contrastive Learning

Most modern embedding models are trained with contrastive learning objectives. Given an anchor text and a positive (semantically similar) text, the model learns to maximize their similarity while minimizing similarity with negative (dissimilar) examples. InfoNCE loss is the most common objective: L = -log(exp(sim(a, p)/tau) / sum(exp(sim(a, n_i)/tau))), where tau is a temperature parameter.

### Hard Negative Mining

The quality of negative examples significantly impacts embedding quality. Hard negatives (examples that are superficially similar but semantically different) provide more informative gradients than random negatives. Common strategies include using BM25 to find lexically similar but semantically different passages, mining negatives from the same batch (in-batch negatives), and using a trained cross-encoder to identify borderline cases.

### Matryoshka Representation Learning

MRL trains embeddings that are useful at multiple dimensionalities. The loss function is applied not just to the full embedding but also to truncated versions (e.g., first 64, 128, 256 dimensions). This allows users to trade off between quality and efficiency at inference time by truncating the embedding to fewer dimensions without retraining.

## Practical Considerations

### Normalization

Most similarity search operations assume normalized embeddings (unit length vectors). Normalizing embeddings before storage ensures that cosine similarity equals dot product, simplifying and speeding up retrieval. Many embedding models output normalized embeddings by default.

### Chunking and Embedding Strategy

The embedding model's maximum token length constrains the input size. For all-MiniLM-L6-v2, this is 256 tokens (roughly 200 words). Documents longer than this must be chunked before embedding. The chunk size should match the typical query length and information density for optimal retrieval performance.

### Dimensionality Reduction

If storage or computation cost is a concern, embeddings can be reduced to fewer dimensions using PCA or random projection. A reduction from 384 to 128 dimensions typically loses less than 5% retrieval accuracy while reducing storage by 67%.

### Asymmetric Search

In many retrieval scenarios, queries and documents have different characteristics (queries are short, documents are longer). Some embedding models handle this by training with asymmetric examples. Others use prefix instructions to differentiate between query and document embedding modes.

## Embedding Evaluation

### Benchmarks

The MTEB (Massive Text Embedding Benchmark) evaluates embeddings across seven task categories: classification, clustering, pair classification, reranking, retrieval, STS (semantic textual similarity), and summarization. It provides a comprehensive picture of embedding model capabilities across diverse use cases.

### Task-Specific Evaluation

For RAG applications, the most relevant metrics are retrieval recall@k and NDCG@k on domain-specific queries. It is important to evaluate on data representative of the target domain, as embedding models trained primarily on web text may underperform on specialized domains like legal, medical, or scientific text.
