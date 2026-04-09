# Prompt Engineering for LLMs

## Introduction

Prompt engineering is the practice of designing and optimizing input prompts to elicit desired behaviors from large language models. As LLMs have become more capable, prompt engineering has evolved from simple task descriptions to sophisticated techniques that can significantly improve model performance without any parameter updates.

## Fundamental Techniques

### Zero-Shot Prompting

Zero-shot prompting provides the task description without any examples. Modern instruction-tuned models perform remarkably well in zero-shot settings. The key is to be specific about the desired output format, constraints, and evaluation criteria. For example, instead of "Summarize this text," a better prompt is "Summarize the following text in 3 bullet points, focusing on the key technical contributions."

### Few-Shot Prompting

Few-shot prompting includes examples of input-output pairs before the actual task. This helps the model understand the expected format, style, and reasoning pattern. Research shows that the choice and order of examples significantly affects performance. Best practices include selecting diverse examples that cover edge cases, ordering examples from simple to complex, and ensuring examples are representative of the actual task distribution.

### Chain-of-Thought (CoT)

Chain-of-thought prompting instructs the model to show its reasoning step by step before giving a final answer. This dramatically improves performance on complex reasoning tasks including math, logic, and multi-step analysis. The simplest form adds "Let's think step by step" to the prompt. More sophisticated variants provide explicit reasoning traces in few-shot examples.

CoT is most effective for tasks requiring multiple reasoning steps and can actually hurt performance on simple tasks where direct answers are more appropriate.

## Advanced Techniques

### ReAct (Reasoning + Acting)

ReAct combines chain-of-thought reasoning with the ability to take actions (like searching for information or running code). The model alternates between thinking (reasoning about what to do next) and acting (executing a tool or query). This pattern is the foundation for modern LLM agent frameworks like LangChain agents and AutoGPT.

### Tree of Thought (ToT)

Tree of Thought extends chain-of-thought by exploring multiple reasoning paths simultaneously. At each step, the model generates several possible next steps, evaluates them, and selects the most promising ones to continue. This enables backtracking and exploration, improving performance on tasks requiring search or planning.

### Self-Consistency

Self-consistency generates multiple chain-of-thought reasoning paths and selects the most common final answer. This is based on the intuition that correct reasoning paths are more likely to converge on the same answer. It improves accuracy at the cost of multiple inference calls.

### Retrieval-Augmented Prompting

In RAG systems, the prompt structure for generation is critical. Effective RAG prompts typically include: a system instruction establishing the model's role, the retrieved context with clear delineation, the user's question, and instructions for handling insufficient context. The prompt should instruct the model to answer based only on the provided context and to acknowledge when the context is insufficient.

## Prompt Templates for RAG

### Basic RAG Template

A basic RAG prompt template structures the interaction as follows: "You are a helpful assistant. Use the following context to answer the question. If the context doesn't contain enough information to answer, say so clearly. Context: {context} Question: {question} Answer:"

### Citation-Aware Template

For applications requiring source attribution: "Based on the provided sources, answer the question. Cite your sources using [Source N] notation. Sources: {numbered_sources} Question: {question} Answer with citations:"

### Conversational RAG Template

For multi-turn conversations, the prompt must include conversation history and handle follow-up questions that reference previous turns. The system must determine whether each new message requires retrieval or can be answered from the existing context.

## System Prompts

System prompts establish the model's persona, capabilities, and constraints. Effective system prompts are specific about the model's role, explicitly state what the model should and should not do, define the output format, and include examples of edge cases. System prompts should be tested thoroughly, as small changes can significantly affect behavior.

## Common Pitfalls

### Prompt Injection

Prompt injection attacks manipulate the model by embedding instructions in user input that override the system prompt. Defenses include input sanitization, output validation, and using separate model calls for different trust levels. No defense is completely reliable, so critical applications should validate model outputs independently.

### Position Bias

LLMs tend to pay more attention to information at the beginning and end of the context window (the "lost in the middle" phenomenon). For RAG systems, this means the most relevant documents should be placed at the beginning of the context, or the prompt should explicitly instruct the model to consider all provided information equally.

### Verbosity Bias

Models tend to prefer generating longer responses, which may include unnecessary details or repetition. Explicit length constraints and instructions to be concise help control this. Evaluation should consider whether longer responses actually provide more value or just more words.

## Evaluation of Prompts

Prompt evaluation should measure task-specific metrics (accuracy, F1, BLEU, etc.), consistency across similar inputs, robustness to rephrasing, and failure modes on edge cases. A/B testing with real users provides the most reliable signal for production applications. Automated evaluation using a stronger model (e.g., GPT-4 evaluating GPT-3.5 outputs) has become a popular proxy for human evaluation.
