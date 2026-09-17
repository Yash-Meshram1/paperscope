import unittest
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
from qdrant_client import QdrantClient, models
from paperscope.retrieval import dense_retrieve
from test_retrieval import chunk
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from paperscope.engine import Config, load_engine


class QdrantTests(unittest.TestCase):
    def test_loader_alignment_and_reduced_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data/processed').mkdir(parents=True)
            chunks = [chunk('a',i) for i in range(2)]
            (root/'data/processed/chunks.json').write_text(json.dumps([asdict(c) for c in chunks]))
            vectors = np.zeros((2,1024),dtype='float32')
            vectors[0,0]=1.003; vectors[1,1]=.998
            np.save(root/'data/processed/embeddings.npy',vectors)
            client=QdrantClient(path=str(root/'storage/qdrant'))
            client.create_collection('research_papers',vectors_config=models.VectorParams(size=1024,distance=models.Distance.COSINE))
            client.upsert('research_papers',[models.PointStruct(id=i,vector=vectors[i].tolist(),payload=asdict(c)) for i,c in enumerate(chunks)])
            client.close()
            embedding=MagicMock()
            embedding.get_sentence_embedding_dimension.return_value=1024
            embedding.__getitem__.return_value.auto_model.config._commit_hash='test'
            reranker=MagicMock(); reranker.model.config._commit_hash='test'
            with patch('sentence_transformers.SentenceTransformer',return_value=embedding), patch('sentence_transformers.CrossEncoder',return_value=reranker):
                engine=load_engine(Config(root))
                self.assertEqual(len(engine.chunks),2)
                engine.close()
                vectors[1]=vectors[0]
                np.save(root/'data/processed/embeddings.npy',vectors)
                with self.assertRaisesRegex(ValueError,'vectors differ'):
                    load_engine(Config(root))

    def test_dense_filter_and_numbered_references(self):
        client = QdrantClient(':memory:')
        try:
            client.create_collection('test', vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
            chunks = [chunk('a',0,'9. References'),chunk('b',1),chunk('a',2)]
            client.upsert('test',[models.PointStruct(id=i,vector=[1.,0.],payload=asdict(c)) for i,c in enumerate(chunks)])
            embedding = SimpleNamespace(encode=lambda *a,**kw:np.array([1.,0.]))
            rows = dense_retrieve('q',embedding,client,'test',top_k=5,paper_id='a')
            self.assertEqual([v.payload['chunk_id'] for v in rows],['a_2'])
        finally:
            client.close()
