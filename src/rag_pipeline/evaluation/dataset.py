"""Synthetic evaluation dataset based on the sample AI/ML documents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalSample:
    """A single evaluation QA pair with ground truth."""

    question: str
    ground_truth: str


EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        question="What is the Transformer architecture and who introduced it?",
        ground_truth=(
            "The Transformer architecture was introduced by Vaswani et al. in 2017 in the paper "
            "'Attention Is All You Need'. It relies entirely on self-attention mechanisms to "
            "capture dependencies between input and output positions, unlike previous models "
            "that used RNNs or CNNs."
        ),
    ),
    EvalSample(
        question="How does scaled dot-product attention work?",
        ground_truth=(
            "Scaled dot-product attention computes attention as: "
            "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V, where Q, K, V are query, "
            "key, and value matrices, and d_k is the dimension of key vectors. The scaling "
            "by sqrt(d_k) prevents dot products from growing too large."
        ),
    ),
    EvalSample(
        question="What is the difference between self-attention and cross-attention?",
        ground_truth=(
            "Self-attention computes attention within a single sequence where queries, keys, "
            "and values all come from the same input. Cross-attention computes attention "
            "between two different sequences, where queries come from one sequence (e.g., "
            "decoder) and keys and values come from another (e.g., encoder output)."
        ),
    ),
    EvalSample(
        question="What is Flash Attention and what are its benefits?",
        ground_truth=(
            "Flash Attention is an IO-aware implementation of exact attention that uses "
            "tiling to reduce memory reads/writes between GPU HBM and on-chip SRAM. "
            "It achieves 2-4x speedups and memory savings without approximating the "
            "attention computation."
        ),
    ),
    EvalSample(
        question="What is RAG and why is it useful?",
        ground_truth=(
            "Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant "
            "external documents at inference time and including them in the prompt context. "
            "It reduces hallucinations, enables access to up-to-date information without "
            "retraining, and allows domain specialization."
        ),
    ),
    EvalSample(
        question="What is the role of re-ranking in a RAG pipeline?",
        ground_truth=(
            "After initial retrieval, a cross-encoder re-ranker scores each document-query "
            "pair more accurately than the bi-encoder used for initial retrieval. "
            "Cross-encoders process the query and document jointly, producing more nuanced "
            "relevance scores at the cost of higher latency."
        ),
    ),
    EvalSample(
        question="What is FAISS and what are its key features?",
        ground_truth=(
            "FAISS (Facebook AI Similarity Search) is an open-source library by Meta for "
            "efficient similarity search. It supports multiple index types (Flat, IVF, HNSW, "
            "PQ), GPU acceleration, filtered search, batch operations, and memory-mapped "
            "indices for datasets larger than RAM."
        ),
    ),
    EvalSample(
        question="What is cosine similarity and when is it used?",
        ground_truth=(
            "Cosine similarity measures the angle between two vectors, ranging from -1 to 1. "
            "It is defined as cos(A, B) = (A dot B) / (||A|| * ||B||). It is invariant to "
            "vector magnitude and is the most commonly used metric for comparing text embeddings."
        ),
    ),
    EvalSample(
        question="What is LoRA and how does it reduce the cost of fine-tuning?",
        ground_truth=(
            "LoRA (Low-Rank Adaptation) freezes pre-trained weights and injects trainable "
            "low-rank decomposition matrices into each layer. For weight matrix W (d x k), "
            "it adds W' = W + BA where B is d x r and A is r x k, with rank r typically "
            "4-64. This reduces trainable parameters by over 99%."
        ),
    ),
    EvalSample(
        question="What is the all-MiniLM-L6-v2 embedding model?",
        ground_truth=(
            "all-MiniLM-L6-v2 is a compact sentence-transformers model producing "
            "384-dimensional embeddings. It uses a 6-layer MiniLM architecture, processes "
            "up to 256 tokens per input, has ~14M parameters (80MB), and encodes 14,000 "
            "sentences per second on a V100 GPU."
        ),
    ),
    EvalSample(
        question="What is chain-of-thought prompting?",
        ground_truth=(
            "Chain-of-thought (CoT) prompting instructs the model to show its reasoning "
            "step by step before giving a final answer. This dramatically improves "
            "performance on complex reasoning tasks including math, logic, and multi-step "
            "analysis. The simplest form adds 'Let's think step by step' to the prompt."
        ),
    ),
    EvalSample(
        question="What is the 'lost in the middle' phenomenon?",
        ground_truth=(
            "LLMs tend to pay more attention to information at the beginning and end of "
            "the context window, losing track of information in the middle. For RAG systems, "
            "this means the most relevant documents should be placed at the beginning of "
            "the context."
        ),
    ),
    EvalSample(
        question="How does DPO differ from RLHF?",
        ground_truth=(
            "DPO (Direct Preference Optimization) simplifies RLHF by directly optimizing "
            "the language model on preference pairs without training a separate reward model. "
            "It reframes the RLHF objective to extract the optimal policy directly from "
            "preference data using a classification loss, making it simpler and more stable."
        ),
    ),
    EvalSample(
        question="What are the advantages of HNSW indexing?",
        ground_truth=(
            "HNSW (Hierarchical Navigable Small World) builds a multi-layer graph where "
            "search starts from sparse top layers with long-range connections and moves "
            "to dense lower layers. It provides excellent recall (>95%) with low latency "
            "and is the default choice in many vector databases."
        ),
    ),
]
