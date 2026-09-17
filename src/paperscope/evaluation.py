"""Paper-level development metrics, independently bounded to the first K."""
from paperscope.retrieval import validate_budget


def score_results(results, expected_paper_ids, k=5):
    validate_budget(k, positive=True)
    expected = set(expected_paper_ids)
    if not expected:
        raise ValueError('Expected paper IDs must not be empty')
    evidence = results[:k]
    papers = [r['chunk']['paper_id'] for r in evidence]
    return {
        'recall': len(expected.intersection(papers)) / len(expected),
        'reciprocal_rank': next((1 / rank for rank, paper in enumerate(papers, 1)
                                 if paper in expected), 0.0),
        'retrieved_papers': papers,
        'chunk_ids': [r['chunk']['chunk_id'] for r in evidence],
        'actual_result_count': len(results),
        'scored_result_count': len(evidence),
    }


def evaluate(questions, retrieve, k=5):
    """retrieve is the same engine.retrieve callable used by the UI."""
    from time import perf_counter
    from paperscope.paper_aliases import detect_paper_ids
    validate_budget(k, positive=True)
    if not questions:
        raise ValueError('Evaluation questions must not be empty')
    rows = []
    for item in questions:
        query = item['question']
        expected = item.get('expected_paper_ids') or [item['expected_paper_id']]
        start = perf_counter()
        results = retrieve(query, top_k=k)
        rows.append({**item, **score_results(results, expected, k),
                     'route': 'named_papers' if detect_paper_ids(query) else 'generic',
                     'latency_ms': (perf_counter() - start) * 1000})
    return {'mrr_at_k': sum(r['reciprocal_rank'] for r in rows) / len(rows),
            'paper_recall_at_k': sum(r['recall'] for r in rows) / len(rows),
            'k': k, 'rows': rows}
