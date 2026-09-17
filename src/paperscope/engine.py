"""One resource owner per process. Startup never indexes or replaces a collection."""
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import RLock
import hashlib
import json
import os

from paperscope.models import Chunk
from paperscope.retrieval import smart_retrieve
from paperscope.generation import generate_answer


@dataclass(frozen=True)
class Config:
    root: Path
    collection: str = 'research_papers'
    embedding_model: str = 'Qwen/Qwen3-Embedding-0.6B'
    reranker_model: str = 'Qwen/Qwen3-Reranker-0.6B'
    device: str = 'cpu'
    ollama_model: str = 'qwen3.5:9b'
    ollama_url: str = 'http://localhost:11434/api/chat'

    @classmethod
    def from_env(cls):
        return cls(root=Path(os.getenv('PAPERSCOPE_ROOT', Path(__file__).resolve().parents[2])),
                   device=os.getenv('PAPERSCOPE_DEVICE', 'cpu'),
                   ollama_url=os.getenv('PAPERSCOPE_OLLAMA_URL', 'http://localhost:11434/api/chat'),
                   ollama_model=os.getenv('PAPERSCOPE_OLLAMA_MODEL', 'qwen3.5:9b'))


def validate_chunks(chunks):
    ids = [c.chunk_id for c in chunks]
    if not ids:
        raise ValueError('No saved chunks found')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate chunk_id in saved chunks; rebuild artifacts separately')
    if any(not c.chunk_id or not c.paper_id or not c.text.strip() for c in chunks):
        raise ValueError('Chunks require IDs and nonempty text')


class Engine:
    def __init__(self, config, chunks, client, embedding_model, reranker, bm25, provenance):
        self.config = config
        self.chunks = chunks
        self.client = client
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.bm25 = bm25
        self.provenance = provenance
        self.lock = RLock()

    def retrieve(self, query, top_k=5):
        if not query.strip():
            raise ValueError('Enter a question')
        with self.lock:
            return smart_retrieve(query, self.embedding_model, self.client,
                                  self.config.collection, self.bm25, self.chunks,
                                  self.reranker, top_k=top_k)

    def answer(self, query, evidence):
        if not evidence:
            return 'The provided papers do not contain enough information to answer this question.'
        with self.lock:
            return generate_answer(query, evidence, model=self.config.ollama_model,
                                   ollama_url=self.config.ollama_url)

    def close(self):
        self.client.close()


def load_engine(config):
    import numpy as np
    from rank_bm25 import BM25Okapi
    from qdrant_client import QdrantClient
    from importlib.metadata import version
    from datetime import datetime, timezone

    path = config.root / 'data/processed/chunks.json'
    raw = path.read_bytes()
    chunks = [Chunk(**c) for c in json.loads(raw)]
    validate_chunks(chunks)
    embedding_path = config.root / 'data/processed/embeddings.npy'
    vectors = np.load(embedding_path, mmap_mode='r', allow_pickle=False)
    if vectors.shape != (len(chunks), 1024) or not np.isfinite(vectors).all():
        raise ValueError('Saved embeddings must be finite, aligned N x 1024 vectors')
    norms = np.linalg.norm(vectors.astype('float32'), axis=1)
    # Historical embeddings were produced at reduced precision. Compare vector
    # directions because cosine storage may preserve or normalize their norms.
    if not np.allclose(norms, 1, atol=.01):
        raise ValueError('Saved embeddings are not normalized')
    storage = config.root / 'storage/qdrant'
    if not (storage / 'meta.json').exists():
        raise FileNotFoundError(f'Missing persisted Qdrant storage: {storage}')
    client = QdrantClient(path=str(storage))
    try:
        if not client.collection_exists(config.collection):
            raise ValueError(f'Missing collection: {config.collection}')
        info = client.get_collection(config.collection)
        vector_config = info.config.params.vectors
        if getattr(vector_config, 'size', None) != 1024 or str(vector_config.distance).lower() != 'cosine':
            raise ValueError('Collection must use unnamed 1024-dimensional cosine vectors')
        # Verify payloads and row-index vector alignment, including stale points.
        records, offset = [], None
        while True:
            page, offset = client.scroll(config.collection, limit=128, offset=offset,
                                         with_payload=True, with_vectors=True)
            records.extend(page)
            if offset is None:
                break
        if len(records) != len(chunks):
            raise ValueError('Qdrant count differs from saved chunks; indexing required separately')
        seen = set()
        for point in records:
            if not isinstance(point.id, int) or not 0 <= point.id < len(chunks):
                raise ValueError('Qdrant point IDs do not match saved row indices')
            expected = asdict(chunks[point.id])
            if point.payload != expected or expected['chunk_id'] in seen:
                raise ValueError('Qdrant payloads differ from saved chunks')
            seen.add(expected['chunk_id'])
            stored = np.asarray(point.vector, dtype='float32')
            if (stored.shape != (1024,) or not np.isfinite(stored).all()
                    or np.linalg.norm(stored) == 0
                    or not np.allclose(stored / np.linalg.norm(stored),
                                       vectors[point.id] / norms[point.id], atol=1e-5)):
                raise ValueError('Qdrant vectors differ from saved embeddings')
        # Serving uses existing local model caches; downloading is a separate step.
        from sentence_transformers import SentenceTransformer, CrossEncoder
        embedding = SentenceTransformer(config.embedding_model, device=config.device, local_files_only=True)
        if embedding.get_sentence_embedding_dimension() != 1024:
            raise ValueError('Embedding model dimension differs from persisted vectors')
        reranker = CrossEncoder(config.reranker_model, device=config.device, local_files_only=True)
        provenance = {
            'loaded_at': datetime.now(timezone.utc).isoformat(),
            'corpus_sha256': hashlib.sha256(raw).hexdigest(),
            'embeddings_sha256': hashlib.sha256(embedding_path.read_bytes()).hexdigest(),
            'saved_vector_norm_range': [float(norms.min()), float(norms.max())],
            'config': {**asdict(config), 'root': str(config.root)},
            'versions': {p:version(p) for p in ('numpy','qdrant-client','rank-bm25','sentence-transformers','transformers','torch')},
            'reranker_adapter': 'CrossEncoder.predict; historical runtime equivalence unverified',
            'model_revisions': {
                'embedding': getattr(embedding[0].auto_model.config, '_commit_hash', None),
                'reranker': getattr(reranker.model.config, '_commit_hash', None),
            },
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(__file__).parent.glob('*.py')},
            'retrieval_parameters': {'contract':'global-top-k-v1', 'generic_candidate_k':'max(20,k)',
                                     'named_candidate_k':'max(15,k)', 'rrf_k':60,
                                     'generic_max_per_paper':2, 'named_selection':'reserve one then fill by score'},
        }
        return Engine(config, chunks, client, embedding, reranker,
                      BM25Okapi([c.text.lower().split() for c in chunks]), provenance)
    except Exception:
        client.close()
        raise
