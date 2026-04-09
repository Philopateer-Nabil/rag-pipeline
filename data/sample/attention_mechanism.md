# Attention Mechanisms in Deep Learning

## History and Motivation

Attention mechanisms were first introduced in the context of neural machine translation by Bahdanau et al. in 2014. The core motivation was to address the information bottleneck in encoder-decoder architectures, where the entire input sequence had to be compressed into a single fixed-length context vector. Attention allows the decoder to selectively focus on different parts of the input sequence at each decoding step.

## Types of Attention

### Additive Attention (Bahdanau)

Additive attention computes alignment scores using a feed-forward network with a single hidden layer. Given the decoder hidden state and each encoder hidden state, the alignment score is computed as: score(s_t, h_i) = v^T * tanh(W_1 * s_t + W_2 * h_i), where v, W_1, and W_2 are learned parameters. This was the original form of attention in neural machine translation.

### Multiplicative Attention (Luong)

Luong attention simplifies the score computation to a dot product: score(s_t, h_i) = s_t^T * W * h_i, or simply s_t^T * h_i for dot-product attention. This is computationally more efficient than additive attention and became the basis for the Transformer's attention mechanism.

### Scaled Dot-Product Attention

The Transformer uses scaled dot-product attention, which normalizes the dot product by the square root of the key dimension. This prevents the dot products from growing too large in magnitude for high-dimensional keys, which would push the softmax into regions with extremely small gradients.

## Self-Attention vs Cross-Attention

Self-attention (or intra-attention) computes attention within a single sequence, where queries, keys, and values all come from the same input. This is the primary mechanism used in both encoder and decoder layers of the Transformer.

Cross-attention computes attention between two different sequences. In the Transformer decoder, cross-attention queries come from the decoder while keys and values come from the encoder output. This is how the decoder accesses information from the input sequence.

## Advanced Attention Variants

### Sparse Attention

Standard self-attention has O(n^2) complexity in sequence length, which limits the maximum sequence length that can be processed. Sparse attention patterns like those in Longformer, BigBird, and Sparse Transformer restrict each position to attend to only a subset of other positions, reducing complexity to O(n * sqrt(n)) or even O(n * log(n)).

### Linear Attention

Methods like Performer and Random Feature Attention approximate the softmax attention kernel using random features, achieving O(n) complexity. These methods decompose the attention computation so that it can be computed without explicitly forming the n x n attention matrix.

### Flash Attention

Flash Attention is an IO-aware implementation of exact attention that uses tiling to reduce memory reads/writes between GPU high-bandwidth memory (HBM) and on-chip SRAM. It achieves significant speedups (2-4x) and memory savings without approximating the attention computation. Flash Attention 2 further optimizes by improving parallelism and work partitioning across GPU thread blocks.

### Multi-Query and Grouped-Query Attention

Multi-Query Attention (MQA) shares key and value projections across all attention heads while keeping separate query projections. This dramatically reduces the key-value cache size during inference, improving throughput for autoregressive generation. Grouped-Query Attention (GQA) is a compromise that groups heads into sets sharing the same key-value projections, balancing quality and efficiency. GQA is used in models like LLaMA 2 and Mistral.

## Attention in Vision

### Vision Transformer (ViT)

ViT applies the Transformer architecture directly to image patches, treating each 16x16 patch as a token. This demonstrated that pure attention-based models can match or exceed CNNs on image classification when trained on sufficient data. Subsequent work like DeiT showed that ViT can be trained effectively on smaller datasets with proper regularization and data augmentation.

### Cross-Attention in Multimodal Models

Models like CLIP, Flamingo, and LLaVA use cross-attention to align representations from different modalities (text and images). This enables capabilities like zero-shot image classification, visual question answering, and image captioning.

## Attention Visualization and Interpretability

Attention weights are often visualized to interpret what the model focuses on. However, research has shown that attention weights do not always reliably indicate feature importance. Methods like attention rollout and gradient-weighted attention provide more reliable interpretability by accounting for the residual connections and multiple layers in the Transformer.
