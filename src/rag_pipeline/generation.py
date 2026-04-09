"""LLM generation using Ollama or local HuggingFace models via LangChain."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline.config import GenerationConfig

logger = structlog.get_logger()

RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions based on the provided context.
Use ONLY the following context to answer the question. If the context does not
contain enough information to answer the question, say "I don't have enough
information in the provided context to answer this question."

Do not make up information. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

# Simpler template for smaller models like flan-t5
RAG_PROMPT_TEMPLATE_SIMPLE = """\
Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""


def get_llm(config: GenerationConfig) -> BaseChatModel:
    """Create an LLM instance based on the configured backend.

    Args:
        config: Generation configuration with backend, model name, and parameters.

    Returns:
        A LangChain chat model instance.
    """
    if config.llm_backend == "huggingface":
        return _get_huggingface_llm(config)
    return _get_ollama_llm(config)


def _get_ollama_llm(config: GenerationConfig) -> BaseChatModel:
    """Create an Ollama LLM instance."""
    from langchain_ollama import ChatOllama

    logger.info("creating_llm", backend="ollama", model=config.model_name)
    return ChatOllama(
        model=config.model_name,
        base_url=config.base_url,
        temperature=config.temperature,
        num_predict=config.max_tokens,
    )


class _Seq2SeqLLM(BaseLLM):
    """LangChain LLM wrapper for encoder-decoder models (T5, BART)."""

    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 256
    temperature: float = 0.1
    do_sample: bool = True

    @property
    def _llm_type(self) -> str:
        return "seq2seq-local"

    def _generate(self, prompts: list[str], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs):
        from langchain_core.outputs import Generation, LLMResult

        generations = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


def _get_huggingface_llm(config: GenerationConfig):
    """Create a local HuggingFace LLM.

    Uses a direct Seq2Seq wrapper for T5/BART encoder-decoder models,
    and HuggingFacePipeline for causal LMs.
    """
    model_name = config.model_name
    is_seq2seq = "t5" in model_name.lower() or "bart" in model_name.lower()
    logger.info("creating_llm", backend="huggingface", model=model_name, is_seq2seq=is_seq2seq)

    if is_seq2seq:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return _Seq2SeqLLM(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=min(config.max_tokens, 512),
            temperature=max(config.temperature, 0.01),
            do_sample=config.temperature > 0,
        )

    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    pipe = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": min(config.max_tokens, 512),
            "temperature": max(config.temperature, 0.01),
            "do_sample": config.temperature > 0,
        },
    )
    return ChatHuggingFace(llm=pipe)


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string.

    Args:
        documents: Retrieved documents to format.

    Returns:
        Formatted context string with source annotations.
    """
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(config: GenerationConfig):
    """Build a LangChain RAG chain.

    Args:
        config: Generation configuration.

    Returns:
        A runnable chain that takes context and question, returns an answer string.
    """
    from langchain_core.prompts import PromptTemplate

    llm = get_llm(config)

    # Use simpler plain-text prompt for seq2seq models (T5/BART)
    is_seq2seq = config.llm_backend == "huggingface" and (
        "t5" in config.model_name.lower() or "bart" in config.model_name.lower()
    )
    template = RAG_PROMPT_TEMPLATE_SIMPLE if is_seq2seq else RAG_PROMPT_TEMPLATE

    if is_seq2seq:
        prompt = PromptTemplate.from_template(template)
    else:
        prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


def generate(
    question: str,
    documents: list[Document],
    config: GenerationConfig,
) -> str:
    """Generate an answer using retrieved context and an LLM.

    Args:
        question: The user's question.
        documents: Retrieved and optionally re-ranked documents.
        config: Generation configuration.

    Returns:
        The generated answer string.
    """
    context = format_context(documents)
    chain = build_rag_chain(config)

    logger.info(
        "generating",
        backend=config.llm_backend,
        model=config.model_name,
        context_docs=len(documents),
    )
    answer = chain.invoke({"context": context, "question": question})
    logger.info("generation_complete", answer_length=len(answer))

    return answer
