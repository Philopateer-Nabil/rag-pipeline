# Transformer Architecture

## Overview

The Transformer architecture, introduced in the seminal paper "Attention Is All You Need" by Vaswani et al. in 2017, revolutionized natural language processing and subsequently many other domains of machine learning. Unlike previous sequence-to-sequence models that relied on recurrent neural networks (RNNs) or convolutional neural networks (CNNs), the Transformer relies entirely on self-attention mechanisms to capture dependencies between input and output positions.

## Key Components

### Self-Attention Mechanism

The self-attention mechanism computes a weighted sum of all positions in the input sequence for each position. It operates using three learned linear projections: Query (Q), Key (K), and Value (V). The attention weights are computed as the scaled dot-product of queries and keys, followed by a softmax normalization. This allows each position to attend to all other positions in the sequence, capturing long-range dependencies efficiently.

The formula for scaled dot-product attention is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V, where d_k is the dimension of the key vectors.

### Multi-Head Attention

Rather than performing a single attention function, multi-head attention runs multiple attention operations in parallel (typically 8 or 16 heads). Each head learns different attention patterns, allowing the model to jointly attend to information from different representation subspaces. The outputs of all heads are concatenated and linearly projected to produce the final output.

### Positional Encoding

Since the Transformer has no inherent notion of sequence order (unlike RNNs), positional encodings are added to the input embeddings to inject information about token positions. The original paper used sinusoidal functions of different frequencies, though learned positional embeddings have since become common. More recently, relative positional encodings like RoPE (Rotary Position Embedding) and ALiBi have been developed to improve generalization to longer sequences.

### Feed-Forward Networks

Each Transformer layer contains a position-wise feed-forward network consisting of two linear transformations with a non-linear activation function (typically ReLU or GELU) in between. This component operates independently on each position and provides additional representational capacity.

### Layer Normalization and Residual Connections

Each sub-layer (self-attention and feed-forward) is wrapped with a residual connection followed by layer normalization. This helps with gradient flow during training and stabilizes the learning process. Pre-norm (applying layer norm before the sub-layer) has become preferred over post-norm in modern implementations.

## Encoder-Decoder Structure

The original Transformer uses an encoder-decoder architecture. The encoder processes the input sequence through a stack of identical layers, each containing self-attention and feed-forward sub-layers. The decoder similarly processes its input but additionally includes cross-attention layers that attend to the encoder output. The decoder uses causal masking to prevent positions from attending to future positions during training.

## Variants and Evolution

### Encoder-Only Models (BERT-style)
BERT and its successors use only the encoder portion of the Transformer, trained with masked language modeling objectives. These models excel at understanding tasks like classification, named entity recognition, and question answering.

### Decoder-Only Models (GPT-style)
GPT and similar models use only the decoder portion with causal (left-to-right) attention masking. These are the foundation for modern large language models like GPT-4, LLaMA, and Mistral. They excel at text generation and have shown emergent capabilities at scale.

### Encoder-Decoder Models (T5-style)
Models like T5 and BART maintain the full encoder-decoder structure and are particularly effective for sequence-to-sequence tasks like translation, summarization, and question answering.

## Scaling Laws

Research has shown that Transformer performance scales predictably with model size, dataset size, and compute budget. The Chinchilla scaling laws suggest that for compute-optimal training, model size and training data should be scaled roughly equally. This insight has influenced the development of more recent models that prioritize training data quality and quantity alongside model scale.
