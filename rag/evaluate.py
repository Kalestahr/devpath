import sys, os

_here = os.path.dirname(os.path.abspath(__file__))          # .../devpath/rag/
_root = os.path.dirname(_here)                               # .../devpath/
if _root not in sys.path:
    sys.path.insert(0, _root)

from dotenv import load_dotenv
load_dotenv()

import json, re, time, random
from collections import defaultdict
from groq import Groq
from index import load_chunks, build_indexes, text_search, vector_search, hybrid_search, hybrid_search_reranked

groq_client = Groq()

GT_MODEL       = "openai/gpt-oss-20b"
JUDGE_MODEL    = "openai/gpt-oss-20b"
RAG_MODEL      = "openai/gpt-oss-120b"
RAG_MODEL_FALLBACK = "openai/gpt-oss-20b"


def _is_daily_quota_error(msg):
    """True if this 429 is a per-day (TPD) limit, not a per-minute burst limit."""
    return bool(re.search(r'per day|TPD|tokens per day|requests per day|RPD', msg, re.IGNORECASE))


def _parse_wait_seconds(msg, attempt, base_wait):
    """Best-effort parse of Groq's 'try again in ...' hint. Falls back to
    exponential backoff if the message doesn't match a known format."""
    m = re.search(r'try again in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?', msg, re.IGNORECASE)
    if m and any(m.groups()):
        h = int(m.group(1) or 0)
        mnt = int(m.group(2) or 0)
        s = float(m.group(3) or 0)
        return int(h * 3600 + mnt * 60 + s) + 10  # +10s buffer
    return base_wait * (attempt + 1)


def groq_call_with_retry(model, messages, max_retries=6, base_wait=90):
    for attempt in range(max_retries):
        try:
            r = groq_client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return r
        except Exception as e:
            msg = str(e)
            if "rate_limit_exceeded" in msg or "429" in msg:
                if _is_daily_quota_error(msg):
                    print(f"  [429] Daily token quota exhausted on {model}. Not retrying.")
                    return None
                wait = _parse_wait_seconds(msg, attempt, base_wait)
                print(f"  [429] Rate limited on {model}. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    print(f"  [ERROR] Gave up after {max_retries} retries.")
    return None


def groq_call_with_fallback(primary_model, fallback_model, messages, **kwargs):
    """Try primary_model; if it fails specifically because of an exhausted
    daily quota, retry once on fallback_model instead of giving up."""
    r = groq_call_with_retry(primary_model, messages, **kwargs)
    if r is None:
        print(f"  Falling back to {fallback_model} for this call...")
        r = groq_call_with_retry(fallback_model, messages, **kwargs)
    return r

# Ground truth generation

def generate_questions(chunk, n=3):
    prompt = f'''Generate {n} questions about this document. Keep questions short and specific.
Return ONLY a Python list on one line like: ["Q1?", "Q2?", "Q3?"]
No explanation, no thinking, just the list.

Document: {chunk["content"][:600]}'''
    r = groq_call_with_retry(GT_MODEL, [{'role': 'user', 'content': prompt}])
    if r is None:
        return []
    text = r.choices[0].message.content.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    questions = re.findall(r'"([^"]+\?)"', text)
    return questions[:n] if questions else []

# LLM-as-judge
def llm_as_judge(question, answer, prompt_variant):
    judge_prompt = f'''Rate this answer on a scale of 1-5 for relevance and accuracy.
Return ONLY a JSON object: {{"score": <1-5>, "reason": "<brief reason>"}}

Question: {question}
Answer: {answer[:500]}
Prompt variant used: {prompt_variant}'''
    r = groq_call_with_retry(JUDGE_MODEL, [{'role': 'user', 'content': judge_prompt}])
    if r is None:
        return {"score": 3, "reason": "Rate limited"}
    text = r.choices[0].message.content.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {"score": 3, "reason": "Could not evaluate"}

# Retrieval metrics
def hit_rate(relevance_list):
    return sum(1 for r in relevance_list if any(x == 1 for x in r)) / len(relevance_list)

def mrr(relevance_list):
    scores = []
    for r in relevance_list:
        for i, v in enumerate(r):
            if v == 1:
                scores.append(1 / (i + 1))
                break
        else:
            scores.append(0)
    return sum(scores) / len(scores)

def evaluate(ground_truth, search_fn):
    relevance = []
    for q, doc_id in ground_truth:
        results = search_fn(q)
        rel = [1 if r['id'] == doc_id else 0 for r in results]
        relevance.append(rel)
    return {
        'hit_rate': round(hit_rate(relevance), 4),
        'mrr': round(mrr(relevance), 4)
    }

# RAG prompt comparison
def evaluate_rag_prompts(ground_truth, search_fn):
    PROMPT_V1 = """You are DevPath, a tech career planning assistant.
Answer the question using only the provided context. Be concise."""

    PROMPT_V2 = """You are DevPath, a tech career planning assistant.
Answer with specific statistics and percentages from the context.
Always cite your source. Give concrete actionable next steps."""

    scores_v1, scores_v2 = [], []
    sample = ground_truth[:10]

    for i, (q, doc_id) in enumerate(sample):
        results = search_fn(q)
        context = '\n\n'.join(r['content'][:400] for r in results[:3])
        for prompt, scores, variant in [
            (PROMPT_V1, scores_v1, 'concise'),
            (PROMPT_V2, scores_v2, 'detailed_with_citations')
        ]:
            r = groq_call_with_fallback(
                RAG_MODEL,
                RAG_MODEL_FALLBACK,
                [
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {q}"}
                ]
            )
            if r is None:
                scores.append(3)
                continue
            answer = r.choices[0].message.content.strip()
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            judgment = llm_as_judge(q, answer, variant)
            scores.append(judgment.get('score', 3))
        if i < len(sample) - 1:
            time.sleep(2)

    avg_v1 = round(sum(scores_v1) / len(scores_v1), 2) if scores_v1 else 0
    avg_v2 = round(sum(scores_v2) / len(scores_v2), 2) if scores_v2 else 0
    return {
        'prompt_v1_concise': {'avg_score': avg_v1, 'description': 'Concise answers'},
        'prompt_v2_detailed': {'avg_score': avg_v2, 'description': 'Detailed with citations'},
        'winner': 'prompt_v2_detailed' if avg_v2 >= avg_v1 else 'prompt_v1_concise'
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print('Building indexes...')
chunks = load_chunks()
text_idx, vec_idx, X = build_indexes(chunks)

gt_path = 'data/processed/ground_truth.json'
if not os.path.exists(gt_path):
    print('Generating ground truth questions...')

    # Stratified sample: up to 6 chunks per id-prefix so all 5 sources
    # (jb2025, onet, so, so2025, wef) are represented in ground truth.
    # Old code used chunks[:30] which only hit jb2025 + onet ids.
    random.seed(42)
    groups = defaultdict(list)
    for chunk in chunks:
        prefix = chunk['id'].split('_')[0]
        groups[prefix].append(chunk)

    sample_chunks = []
    for prefix, group in sorted(groups.items()):
        picked = random.sample(group, min(6, len(group)))
        sample_chunks.extend(picked)
        print(f'  Sampling {len(picked)} from prefix "{prefix}" ({len(group)} total)')

    ground_truth = []
    for chunk in sample_chunks:
        try:
            qs = generate_questions(chunk, n=3)
            for q in qs:
                ground_truth.append([q, chunk['id']])
            print(f'  {chunk["id"]}: {len(qs)} questions')
        except Exception as e:
            print(f'  Error on {chunk["id"]}: {e}')
        time.sleep(1)

    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f'Ground truth saved: {len(ground_truth)} questions')
else:
    with open(gt_path) as f:
        ground_truth = json.load(f)
    print(f'Loaded {len(ground_truth)} ground truth questions')

if not ground_truth:
    print('ERROR: No ground truth generated.')
    sys.exit(1)


# Retrieval evaluation
print('\n=== RETRIEVAL EVALUATION ===')
print('Evaluating text_search...')
text_eval = evaluate(ground_truth, lambda q: text_search(q, text_idx))
print(f'  Hit Rate: {text_eval["hit_rate"]}  MRR: {text_eval["mrr"]}')

print('Evaluating vector_search...')
vec_eval = evaluate(ground_truth, lambda q: vector_search(q, vec_idx))
print(f'  Hit Rate: {vec_eval["hit_rate"]}  MRR: {vec_eval["mrr"]}')

print('Evaluating hybrid_search...')
hyb_eval = evaluate(ground_truth, lambda q: hybrid_search(q, text_idx, vec_idx))
print(f'  Hit Rate: {hyb_eval["hit_rate"]}  MRR: {hyb_eval["mrr"]}')

print('Evaluating hybrid_search + reranking (takes longer, cross-encoder runs locally)...')
hyb_rerank_eval = evaluate(ground_truth, lambda q: hybrid_search_reranked(q, text_idx, vec_idx))
print(f'  Hit Rate: {hyb_rerank_eval["hit_rate"]}  MRR: {hyb_rerank_eval["mrr"]}')

_candidates = {
    'text_search':     (text_eval,       lambda q: text_search(q, text_idx)),
    'vector_search':   (vec_eval,        lambda q: vector_search(q, vec_idx)),
    'hybrid_search':   (hyb_eval,        lambda q: hybrid_search(q, text_idx, vec_idx)),
    'hybrid_reranked': (hyb_rerank_eval, lambda q: hybrid_search_reranked(q, text_idx, vec_idx)),
}
best_method_name = max(_candidates, key=lambda k: _candidates[k][0]['hit_rate'])
best_search_fn = _candidates[best_method_name][1]
print(f'\nBest retrieval method by hit rate: {best_method_name}')

# RAG evaluation -
eval_results_path = 'data/processed/eval_results.json'
cached_rag_eval = None
if os.path.exists(eval_results_path):
    try:
        with open(eval_results_path) as f:
            cached = json.load(f)
        cached_rag_eval = cached.get('rag_evaluation')
        if cached_rag_eval:
            print('\n=== RAG EVALUATION (LLM-as-Judge) ===')
            print('  Loaded from cache (delete data/processed/eval_results.json to re-run).')
    except Exception:
        pass

if cached_rag_eval is None:
    print('\n=== RAG EVALUATION (LLM-as-Judge) ===')
    print(f'Comparing prompt variants using {best_method_name} (takes ~2 min)...')
    cached_rag_eval = evaluate_rag_prompts(ground_truth, best_search_fn)

rag_eval = cached_rag_eval
print(f'  Prompt V1 (concise): avg score {rag_eval["prompt_v1_concise"]["avg_score"]}/5')
print(f'  Prompt V2 (detailed+citations): avg score {rag_eval["prompt_v2_detailed"]["avg_score"]}/5')
print(f'  Winner: {rag_eval["winner"]}')

results = {
    'text': text_eval,
    'vector': vec_eval,
    'hybrid': hyb_eval,
    'hybrid_reranked': hyb_rerank_eval,
    'best_retrieval_method': best_method_name,
    'rag_evaluation': rag_eval
}
with open(eval_results_path, 'w') as f:
    json.dump(results, f, indent=2)

print('\n=== EVALUATION SUMMARY ===')
print(f'Method            Hit Rate   MRR')
print(f'text_search       {text_eval["hit_rate"]:.4f}     {text_eval["mrr"]:.4f}')
print(f'vector_search     {vec_eval["hit_rate"]:.4f}     {vec_eval["mrr"]:.4f}')
print(f'hybrid (RRF)      {hyb_eval["hit_rate"]:.4f}     {hyb_eval["mrr"]:.4f}')
print(f'hybrid + rerank   {hyb_rerank_eval["hit_rate"]:.4f}     {hyb_rerank_eval["mrr"]:.4f}')
print(f'\nBest method: {best_method_name} (used for RAG prompt comparison and in production agent)')
print(f'\nRAG Prompt Comparison:')
print(f'  V1 concise:           {rag_eval["prompt_v1_concise"]["avg_score"]}/5')
print(f'  V2 detailed+citations: {rag_eval["prompt_v2_detailed"]["avg_score"]}/5')
print(f'  Selected: {rag_eval["winner"]}')