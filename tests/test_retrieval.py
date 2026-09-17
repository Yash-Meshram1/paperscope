import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch, Mock

from paperscope import retrieval as r
from paperscope.models import Chunk
from paperscope.paper_aliases import detect_paper_ids
from paperscope.evaluation import score_results, evaluate


def chunk(paper='dpr_karpukhin_2020', index=0, heading='Methods'):
    return Chunk(f'{paper}_{index}', paper, paper, heading, None, 1, 2, index, f'evidence {index}')


def result(paper, index=0, score=1):
    return {'chunk': asdict(chunk(paper, index)), 'rerank_score': score, 'rrf_score': .01}


ARGS = dict(embedding_model=None, client=None, collection_name='test', bm25=None, chunks=[], reranker=None)


class RoutingTests(unittest.TestCase):
    def fake_rerank(self, **kwargs):
        paper = kwargs.get('paper_id', 'generic')
        score = 100 if paper == 'dpr_karpukhin_2020' else 10
        return [result(paper, i, score-i) for i in range(kwargs['top_k'])]

    def test_named_budgets(self):
        for query, count in [('DPR', 1), ('DPR vs ColBERT', 2), ('DPR vs ColBERT vs HyDE', 3)]:
            for k in (1, 2, 5, 8):
                with self.subTest(query=query, k=k), patch.object(r, 'reranked_retrieve', side_effect=self.fake_rerank):
                    rows = r.smart_retrieve(query, **ARGS, top_k=k)
                    self.assertEqual(len(rows), k)
                    self.assertEqual(len({v['chunk']['paper_id'] for v in rows}), min(k, count))
                    self.assertEqual(len({v['chunk']['chunk_id'] for v in rows}), k)

    def test_generic_and_unknown(self):
        for query in ('How is retrieval improved?', 'Explain UnknownPaperXYZ', 'craggy raptors'):
            with patch.object(r, 'reranked_retrieve', return_value=[]) as call:
                r.smart_retrieve(query, **ARGS, top_k=4)
                self.assertEqual(call.call_args.kwargs['top_k'], 4)
                self.assertNotIn('paper_id', call.call_args.kwargs)

    def test_aliases(self):
        self.assertEqual(detect_paper_ids('Corrective RAG / CRAG'), ['crag_yan_2024'])
        self.assertEqual(detect_paper_ids('Self-RAG and self rag'), ['self_rag_asai_2023'])
        self.assertEqual(detect_paper_ids('raptors and craggy'), [])

    def test_empty_named_paper_reallocates(self):
        def fake(**kw):
            return [] if kw['paper_id']=='colbert_khattab_2020' else self.fake_rerank(**kw)
        with patch.object(r, 'reranked_retrieve', side_effect=fake):
            self.assertEqual(len(r.smart_retrieve('DPR vs ColBERT', **ARGS)), 5)

    def test_empty_evidence_and_zero(self):
        with patch.object(r, 'hybrid_retrieve', return_value=[]):
            self.assertEqual(r.reranked_retrieve('q', **ARGS), [])
        with patch.object(r, 'reranked_retrieve') as call:
            self.assertEqual(r.smart_retrieve('DPR', **ARGS, top_k=0), [])
            call.assert_not_called()
        for k in (-1, 1.5, True):
            with self.assertRaises(ValueError): r.smart_retrieve('q', **ARGS, top_k=k)

    def test_single_paper_reranker_not_capped_at_two(self):
        rows = [result('dpr', i) for i in range(5)]
        args = {**ARGS, 'reranker': Mock(predict=Mock(return_value=[5,4,3,2,1]))}
        with patch.object(r, 'hybrid_retrieve', return_value=rows):
            self.assertEqual(len(r.reranked_retrieve('q', **args, paper_id='dpr')), 5)
            self.assertEqual(len(r.reranked_retrieve('q', **args)), 2)

    def test_filter_propagation(self):
        with patch.object(r, 'dense_retrieve', return_value=[]) as dense, patch.object(r, 'bm25_retrieve', return_value=[]) as sparse:
            args = {k:v for k,v in ARGS.items() if k!='reranker'}
            r.hybrid_retrieve('q', **args, paper_id='dpr')
            self.assertEqual(dense.call_args.kwargs['paper_id'], 'dpr')
            self.assertEqual(sparse.call_args.kwargs['paper_id'], 'dpr')

    def test_excluded_heading_and_bm25_filter(self):
        for heading in ('References', '7. References', 'VIII. ACKNOWLEDGMENTS', 'A. Acknowledgements'):
            self.assertTrue(r.is_excluded_section(heading))
        self.assertFalse(r.is_excluded_section('Appendix'))
        chunks = [chunk('dpr',0,'7. References'),chunk('other',1),chunk('dpr',2)]
        rows = r.bm25_retrieve('q',Mock(get_scores=Mock(return_value=[3,2,1])),chunks,paper_id='dpr')
        self.assertEqual([v['chunk'].chunk_id for v in rows], ['dpr_2'])

    def test_duplicate_ids(self):
        c = chunk()
        point = SimpleNamespace(payload=asdict(c))
        fused = r.reciprocal_rank_fusion([point,point],[{'chunk':c}])
        self.assertEqual(len(fused),1)
        self.assertAlmostEqual(fused[0]['rrf_score'], 2/61)
        other = asdict(c); other['text']='conflicting text'
        with self.assertRaisesRegex(ValueError,'Conflicting duplicate'):
            r.reciprocal_rank_fusion([SimpleNamespace(payload=other)],[{'chunk':c}])

    def test_evaluator_slices_independently(self):
        rows = [result('a',i) for i in range(5)] + [result('b')]
        scored = score_results(rows,['a','b'],5)
        self.assertEqual(scored['recall'],.5)
        self.assertEqual(scored['actual_result_count'],6)
        self.assertEqual(score_results(rows,['b'],5)['reciprocal_rank'],0)
        self.assertEqual(score_results([],['b'],5)['recall'],0)
        run = evaluate([{'question':'DPR','expected_paper_id':'b'}],lambda *a,**kw:rows)
        self.assertEqual(run['mrr_at_k'],0)


if __name__ == '__main__': unittest.main()
