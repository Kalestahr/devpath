import json, duckdb, numpy as np
from minsearch import Index, VectorSearch
from embedder import Embedder
from sentence_transformers import CrossEncoder

embedder = Embedder()
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print('Loading cross-encoder reranker...')
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print('Reranker loaded')
    return _reranker


def load_chunks():
    conn = duckdb.connect('devpath_pipeline.duckdb')
    rows = conn.execute(
        'SELECT id, source, dev_type, content FROM devpath.career_chunks'
    ).fetchall()
    conn.close()
    return [{'id': r[0], 'source': r[1], 'dev_type': r[2] or '', 'content': r[3]} for r in rows]


def build_indexes(chunks):
    print(f'Building indexes for {len(chunks)} chunks...')
    text_index = Index(text_fields=['content', 'dev_type'], keyword_fields=['source'])
    text_index.fit(chunks)
    print('Text index built')
    contents = [c['content'] for c in chunks]
    print('Encoding embeddings (takes ~60 seconds)...')
    X = embedder.encode_batch(contents)
    print(f'Embeddings shape: {X.shape}')
    vector_index = VectorSearch(keyword_fields=['source'])
    vector_index.fit(X, chunks)
    print('Vector index built')
    return text_index, vector_index, X


def rrf(result_lists, k=60, n=5):
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc['id']
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[k] for k in ranked[:n]]


def text_search(query, text_index, n=5):
    return text_index.search(query, num_results=n, boost_dict={"dev_type": 3.0, "content": 1.0})


def vector_search(query, vector_index, n=5):
    v = embedder.encode(query)
    return vector_index.search(v, num_results=n)


def hybrid_search(query, text_index, vector_index, n=5):
    t = text_search(query, text_index, n=10)
    v = vector_search(query, vector_index, n=10)
    return rrf([t, v], n=n)


def rerank(query, candidates, n=5):
    """Re-score candidates with a cross-encoder and return the top n.
    Runs locally, no LLM tokens used."""
    if not candidates:
        return candidates
    reranker = get_reranker()
    pairs = [(query, c['content']) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:n]]


def hybrid_search_reranked(query, text_index, vector_index, n=5, candidate_pool=15):
    """Hybrid search followed by cross-encoder reranking.
    Pulls a larger candidate pool from RRF, then reranks down to n."""
    candidates = hybrid_search(query, text_index, vector_index, n=candidate_pool)
    return rerank(query, candidates, n=n)


if __name__ == '__main__':
    chunks = load_chunks()
    text_idx, vec_idx, X = build_indexes(chunks)
    q = 'How do I become a data engineer?'
    results = hybrid_search_reranked(q, text_idx, vec_idx)
    print(f'Test query: {q}')
    for r in results:
        print(f'  [{r["source"]}] {r["id"]}: {r["content"][:80]}...')