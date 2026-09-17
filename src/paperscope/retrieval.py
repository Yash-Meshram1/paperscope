from __future__ import annotations
from dataclasses import asdict
import re
import math



EXCLUDED_SECTIONS = {
    "references",
    "acknowledgments",
    "acknowledgements",
}

def is_excluded_section(section_heading: str) -> bool:
    if not section_heading:
        return False

    heading = re.sub(r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+|[a-z])[.)]?\s+)+", "", section_heading.strip().lower())
    return heading.rstrip(".:").strip() in EXCLUDED_SECTIONS


def diversify_by_paper(
    results,
    top_k: int = 5,
    max_per_paper: int = 2
):
    validate_budget(top_k)
    if top_k == 0:
        return []
    diversified = []
    paper_counts = {}

    for result in results:
        paper_id = result["chunk"]["paper_id"]

        count = paper_counts.get(paper_id, 0)

        if count >= max_per_paper:
            continue

        diversified.append(result)
        paper_counts[paper_id] = count + 1

        if len(diversified) == top_k:
            break

    return diversified


def dense_retrieve(
    query: str,
    embedding_model,
    client,
    collection_name: str,
    top_k: int = 5,
    paper_id: str | None = None
):
    validate_budget(top_k)
    if top_k == 0:
        return []
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    query_filter = None

    if paper_id is not None:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=paper_id)
                )
            ]
        )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        query_filter=query_filter,
        limit=top_k * 3
    )

    filtered_results = []

    for point in results.points:
        section_heading = point.payload.get("section_heading", "")

        if is_excluded_section(section_heading):
            continue

        filtered_results.append(point)

        if len(filtered_results) == top_k:
            break

    return filtered_results

def bm25_retrieve(
    query: str,
    bm25,
    chunks,
    top_k: int = 5,
    paper_id: str | None = None
):
    validate_budget(top_k)
    if top_k == 0:
        return []
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices:
        chunk = chunks[index]

        if paper_id is not None and chunk.paper_id != paper_id:
            continue

        if is_excluded_section(chunk.section_heading):
            continue

        results.append({
            "score": float(scores[index]),
            "chunk": chunk
        })

        if len(results) == top_k:
            break

    return results

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k: int = 60
):
    """
    Combine dense and BM25 rankings using Reciprocal Rank Fusion.

    RRF score:
        1 / (k + rank)

    Results appearing in both retrieval systems receive contributions
    from both rankings.
    """

    scores = {}
    chunks_by_id = {}
    rankings = [[point.payload for point in dense_results],
                [asdict(result["chunk"]) for result in bm25_results]]
    for ranking in rankings:
        seen = set()
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = chunk["chunk_id"]
            if chunk_id in chunks_by_id and chunks_by_id[chunk_id] != chunk:
                raise ValueError(f"Conflicting duplicate chunk_id: {chunk_id}")
            chunks_by_id[chunk_id] = chunk
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for chunk_id, score in ranked:
        results.append(
            {
                "rrf_score": score,
                "chunk": chunks_by_id[chunk_id]
            }
        )

    return results


def hybrid_retrieve(
    query: str,
    embedding_model,
    client,
    collection_name: str,
    bm25: BM25Okapi,
    chunks,
    paper_id: str | None = None,
    candidate_k: int = 10,
    top_k: int = 5,
    rrf_k: int = 60
):
    """
    Perform hybrid retrieval using:

        Dense retrieval
              +
        BM25 retrieval
              Ã¢â€ â€œ
        Reciprocal Rank Fusion
    """

    validate_budget(top_k)
    if top_k == 0:
        return []
    validate_budget(candidate_k, positive=True)
    dense_results = dense_retrieve(
    query=query,
    embedding_model=embedding_model,
    client=client,
    collection_name=collection_name,
    top_k=candidate_k,
    paper_id=paper_id
    )

    bm25_results = bm25_retrieve(
    query=query,
    bm25=bm25,
    chunks=chunks,
    top_k=candidate_k,
    paper_id=paper_id
    )

    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        k=rrf_k
    )

    return fused_results[:top_k]


def reranked_retrieve(
    query: str,
    embedding_model,
    client,
    collection_name: str,
    bm25: BM25Okapi,
    chunks,
    reranker,
    paper_id: str | None = None,
    candidate_k: int = 15,
    top_k: int = 5,
    rrf_k: int = 60
):
    """
    Full retrieval pipeline:

        query
          Ã¢â€ â€œ
        Dense + BM25
          Ã¢â€ â€œ
        RRF fusion
          Ã¢â€ â€œ
        candidate chunks
          Ã¢â€ â€œ
        Cross-encoder reranking
          Ã¢â€ â€œ
        final top-k evidence
    """

    validate_budget(top_k)
    if top_k == 0:
        return []
    validate_budget(candidate_k, positive=True)
    candidates = hybrid_retrieve(
        query=query,
        embedding_model=embedding_model,
        client=client,
        collection_name=collection_name,
        bm25=bm25,
        chunks=chunks,
        paper_id=paper_id,
        candidate_k=candidate_k,
        top_k=candidate_k,
        rrf_k=rrf_k
    )

    if not candidates:
        return []

    pairs = [
        (
            query,
            result["chunk"]["text"]
        )
        for result in candidates
    ]

    scores = reranker.predict(pairs)
    if len(scores) != len(candidates) or any(not math.isfinite(float(score)) for score in scores):
        raise ValueError('Reranker must return one finite score per candidate')

    reranked_results = []

    for result, score in zip(candidates, scores):
        reranked_results.append(
            {
                "rerank_score": float(score),
                "rrf_score": result["rrf_score"],
                "chunk": result["chunk"]
            }
        )

    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    if paper_id is not None:
        return reranked_results[:top_k]

    return diversify_by_paper(
    reranked_results,
    top_k=top_k,
    max_per_paper=2
)

def validate_budget(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError("Budget must be a positive integer" if positive else "Budget must be a nonnegative integer")


def retrieve_from_papers(query, paper_ids, embedding_model, client,
                         collection_name, bm25, chunks, reranker,
                         per_paper_k=None, top_k=5):
    """Global cap; reserve one per available paper, then fill by score.

    If k is smaller than the number of available papers, choose the k best
    paper winners. Optional per_paper_k caps each paper's candidate supply.
    Empty papers do not consume slots. Final evidence is sorted by score.
    """
    validate_budget(top_k)
    if per_paper_k is not None:
        validate_budget(per_paper_k, positive=True)
    if not top_k:
        return []
    combined, winners = [], []
    for paper_id in dict.fromkeys(paper_ids):
        results = reranked_retrieve(
            query=query, embedding_model=embedding_model, client=client,
            collection_name=collection_name, bm25=bm25, chunks=chunks,
            reranker=reranker, paper_id=paper_id,
            candidate_k=max(15, top_k),
            top_k=min(top_k, per_paper_k) if per_paper_k else top_k)
        if results:
            winners.append(results[0])
            combined.extend(results)
    key = lambda x: (-x["rerank_score"], x["chunk"]["chunk_id"])
    selected = sorted(winners, key=key)[:top_k]
    seen = {r["chunk"]["chunk_id"] for r in selected}
    for result in sorted(combined, key=key):
        if len(selected) >= top_k:
            break
        if result["chunk"]["chunk_id"] not in seen:
            selected.append(result)
            seen.add(result["chunk"]["chunk_id"])
    return sorted(selected, key=key)


from paperscope.paper_aliases import detect_paper_ids


def smart_retrieve(query, embedding_model, client, collection_name, bm25,
                   chunks, reranker, top_k=5):
    """Return at most top_k unique evidence chunks; zero skips retrieval.

    Named papers use coverage reservation; generic/unknown aliases use the
    existing two-per-paper diversity policy, which may underfill the budget.
    """
    validate_budget(top_k)
    if not top_k:
        return []
    args = dict(query=query, embedding_model=embedding_model, client=client,
                collection_name=collection_name, bm25=bm25, chunks=chunks,
                reranker=reranker, top_k=top_k)
    paper_ids = detect_paper_ids(query)
    if paper_ids:
        return retrieve_from_papers(paper_ids=paper_ids, **args)
    return reranked_retrieve(candidate_k=max(20, top_k), **args)
