"""Click-based CLI for the RAG pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import structlog

from rag_pipeline.config import load_settings

logger = structlog.get_logger()


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Path to config YAML file")
@click.pass_context
def cli(ctx: click.Context, config: str):
    """RAG Pipeline - Retrieval-Augmented Generation with FAISS and Ollama."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.argument("path")
@click.pass_context
def ingest(ctx: click.Context, path: str):
    """Ingest documents from a file or directory.

    PATH is a file or directory containing PDF, Markdown, or text files.
    """
    from rag_pipeline.pipeline import RAGPipeline

    settings = load_settings(ctx.obj["config_path"])
    pipeline = RAGPipeline(settings=settings)

    click.echo(f"Ingesting documents from: {path}")
    try:
        num_chunks = pipeline.ingest(path)
        click.echo(f"Successfully ingested {num_chunks} chunks.")
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--top-k", "-k", type=int, help="Number of documents to retrieve")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["similarity", "mmr", "hybrid"]),
    help="Retrieval strategy",
)
@click.option("--no-rerank", is_flag=True, help="Disable re-ranking")
@click.option("--show-sources", is_flag=True, help="Show source documents")
@click.pass_context
def query(
    ctx: click.Context,
    question: str,
    top_k: int | None,
    strategy: str | None,
    no_rerank: bool,
    show_sources: bool,
):
    """Ask a question and get an answer from the RAG pipeline.

    QUESTION is the question to ask.
    """
    from rag_pipeline.pipeline import RAGPipeline

    settings = load_settings(ctx.obj["config_path"])
    if top_k is not None:
        settings.retrieval.top_k = top_k
    if strategy is not None:
        settings.retrieval.strategy = strategy
    if no_rerank:
        settings.reranker.enabled = False

    pipeline = RAGPipeline(settings=settings)

    try:
        result = pipeline.query(question)
    except FileNotFoundError:
        click.echo("Error: No vector index found. Run 'rag ingest' first.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"\nQuestion: {result.question}")
    click.echo(f"\nAnswer: {result.answer}")
    click.echo(
        f"\n[Retrieved: {result.num_chunks_retrieved}, "
        f"After rerank: {result.num_chunks_after_rerank}, "
        f"Latency: {result.latency_ms:.0f}ms]"
    )

    if show_sources:
        click.echo("\n--- Sources ---")
        for i, doc in enumerate(result.source_documents, 1):
            source = doc.metadata.get("source", "unknown")
            click.echo(f"\n[{i}] {source}")
            click.echo(doc.page_content[:300])


@cli.command()
@click.option("--output", "-o", default="evaluation_report", help="Output file base name")
@click.pass_context
def evaluate(ctx: click.Context, output: str):
    """Run RAGAS-style evaluation on the pipeline."""
    from rag_pipeline.evaluation.runner import run_evaluation

    settings = load_settings(ctx.obj["config_path"])

    click.echo("Running evaluation...")
    try:
        report = run_evaluation(settings, output_base=output)
        click.echo("\nEvaluation complete!")
        click.echo(f"  JSON report: {output}.json")
        click.echo(f"  Markdown summary: {output}.md")
        click.echo(f"\n  Average faithfulness:      {report['summary']['avg_faithfulness']:.3f}")
        click.echo(f"  Average answer relevancy:  {report['summary']['avg_answer_relevancy']:.3f}")
        ctx_prec = report['summary']['avg_context_precision']
        click.echo(f"  Average context precision:  {ctx_prec:.3f}")
    except FileNotFoundError:
        click.echo("Error: No vector index found. Run 'rag ingest' first.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during evaluation: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--host", "-h", default=None, help="Override host")
@click.option("--port", "-p", type=int, default=None, help="Override port")
@click.pass_context
def serve(ctx: click.Context, host: str | None, port: int | None):
    """Start the FastAPI server."""
    import uvicorn

    from rag_pipeline.api.app import create_app

    settings = load_settings(ctx.obj["config_path"])
    if host:
        settings.api.host = host
    if port:
        settings.api.port = port

    app = create_app(settings=settings)

    click.echo(f"Starting server on {settings.api.host}:{settings.api.port}")
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)


@cli.command()
@click.option("--port", "-p", type=int, default=8501, help="Streamlit port")
@click.pass_context
def ui(ctx: click.Context, port: int):
    """Launch the Streamlit chat UI."""
    import subprocess

    from rag_pipeline import __file__ as pkg_init

    ui_path = str(Path(pkg_init).parent / "ui.py")
    click.echo(f"Starting Streamlit UI on port {port}...")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", ui_path,
            "--server.port", str(port),
            "--server.headless", "true",
        ],
        check=False,
    )


if __name__ == "__main__":
    cli()
