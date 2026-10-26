
"""Evaluate repository retrieval quality."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core.config import get_settings
from core.repository_registry import get_repository_registry
from evaluation.dataset import EVALUATION_DATASET
from evaluation.metrics import precision_at_k, recall_at_k
from evaluation.reranker import rerank_sources
from rag import create_repository_retriever

logger = logging.getLogger(__name__)


def normalize_source_path(source: str) -> str:
    """Normalize a source path for comparison with expected paths."""
    return source.replace("\\", "/").removeprefix("./").lstrip("/")


def evaluate_retrieval(
    repository_id: str,
    k: int = 5,
    debug: bool = False,
    use_reranker: bool = False,
) -> dict:
    """Evaluate retrieval using Precision@K and Recall@K."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    settings = get_settings()
    registry = get_repository_registry(settings)
    repository_path = registry.resolve(repository_id)

    retriever = create_repository_retriever(
        repository_id=repository_id,
        settings=settings,
    )

    if debug:
        print(f"Repository ID: {repository_id}")
        print(f"Repository path: {repository_path}")
        print(f"K: {k}")
        print(f"Reranker enabled: {use_reranker}")

        try:
            vector_store = retriever.vectorstore

            filtered_docs = vector_store.similarity_search(
                "repository source code",
                k=k,
                filter={"repository_id": repository_id},
            )
            print(f"Filtered document count: {len(filtered_docs)}")

            unfiltered_docs = vector_store.similarity_search(
                "repository source code",
                k=k,
            )
            print(f"Unfiltered document count: {len(unfiltered_docs)}")

            for index, document in enumerate(filtered_docs[:3], start=1):
                print(
                    f"Document {index}: metadata={document.metadata}, "
                    f"preview={document.page_content[:150]!r}"
                )
        except Exception:
            logger.exception("Retrieval diagnostics failed")

    results = []
    precision_scores = []
    recall_scores = []

    for sample in EVALUATION_DATASET:
        question = sample["question"]
        expected_sources = [
            normalize_source_path(source)
            for source in sample["expected_sources"]
        ]

        try:
            documents = retriever.invoke(question)

            if documents is None:
                raise RuntimeError(
                    f"Retriever returned None for question: {question!r}"
                )

            retrieved_sources = []

            for document in documents:
                metadata = document.metadata or {}
                source = (
                    metadata.get("source")
                    or metadata.get("file_path")
                    or metadata.get("path")
                )

                if not source or not isinstance(source, str):
                    continue

                source = normalize_source_path(source)

                if source not in retrieved_sources:
                    retrieved_sources.append(source)

            baseline_sources = list(retrieved_sources)

            if use_reranker and baseline_sources:
                try:
                    reranked_sources = rerank_sources(
                        question=question,
                        sources=baseline_sources,
                        settings=settings,
                    )

                    # Accept only a valid permutation of retrieved sources.
                    if (
                        isinstance(reranked_sources, list)
                        and all(
                            isinstance(source, str)
                            for source in reranked_sources
                        )
                        and len(reranked_sources) == len(baseline_sources)
                        and len(set(reranked_sources))
                        == len(reranked_sources)
                        and set(reranked_sources) == set(baseline_sources)
                    ):
                        retrieved_sources = reranked_sources
                    else:
                        logger.warning(
                            "Invalid reranker output for %r; "
                            "using baseline ranking",
                            question,
                        )
                        retrieved_sources = baseline_sources

                except Exception:
                    logger.exception(
                        "Reranking failed for %r; using baseline ranking",
                        question,
                    )
                    retrieved_sources = baseline_sources

            if retrieved_sources is None:
                retrieved_sources = []

            precision = precision_at_k(
                retrieved_sources,
                expected_sources,
                k,
            )
            recall = recall_at_k(
                retrieved_sources,
                expected_sources,
                k,
            )

            result = {
                "question": question,
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "top_k_sources": retrieved_sources[:k],
                "precision_at_k": precision,
                "recall_at_k": recall,
            }
            results.append(result)

            # Exclude unanswerable questions from retrieval quality averages.
            if expected_sources:
                precision_scores.append(precision)
                recall_scores.append(recall)

            if debug:
                print(f"\nQuestion: {question}")
                print(f"Expected sources: {expected_sources}")
                print(f"Retrieved sources: {retrieved_sources}")
                print(f"Precision@{k}: {precision:.3f}")
                print(f"Recall@{k}: {recall:.3f}")

        except Exception as exc:
            logger.exception("Evaluation failed for question: %s", question)

            results.append(
                {
                    "question": question,
                    "expected_sources": expected_sources,
                    "retrieved_sources": [],
                    "top_k_sources": [],
                    "precision_at_k": 0.0,
                    "recall_at_k": 0.0,
                    "error": str(exc),
                }
            )

            if expected_sources:
                precision_scores.append(0.0)
                recall_scores.append(0.0)

    mean_precision = (
        sum(precision_scores) / len(precision_scores)
        if precision_scores
        else 0.0
    )
    mean_recall = (
        sum(recall_scores) / len(recall_scores)
        if recall_scores
        else 0.0
    )

    return {
        "repository_id": repository_id,
        "k": k,
        "reranker_enabled": use_reranker,
        "total_questions": len(EVALUATION_DATASET),
        "retrieval_questions": len(precision_scores),
        "metrics": {
            f"mean_precision_at_{k}": round(mean_precision, 4),
            f"mean_recall_at_{k}": round(mean_recall, 4),
        },
        "results": results,
    }


def main() -> None:
    """Run retrieval evaluation and save its JSON report."""
    parser = argparse.ArgumentParser(
        description="Evaluate RepoContext AI retrieval quality."
    )
    parser.add_argument(
        "--repository-id",
        required=True,
        help="Registered repository ID to evaluate.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top retrieved sources to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results.json",
        help="Path for the JSON evaluation report.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieval diagnostics and per-question metrics.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable LLM-based source reranking.",
    )

    args = parser.parse_args()

    report = evaluate_retrieval(
        repository_id=args.repository_id,
        k=args.k,
        debug=args.debug,
        use_reranker=args.rerank,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Evaluation report saved to: {output_path}")
    print(f"Questions evaluated: {report['total_questions']}")
    print(f"Answerable questions: {report['retrieval_questions']}")
    print(f"Metrics: {json.dumps(report['metrics'])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
