from dotenv import load_dotenv
load_dotenv()

import os, json, re
import logfire

if os.getenv('LOGFIRE_TOKEN'):
    logfire.configure()

from dataclasses import dataclass
from pydantic import BaseModel
from groq import AsyncGroq
from index import load_chunks, build_indexes, hybrid_search_reranked

print("Building indexes...")
_chunks = load_chunks()
_text_idx, _vec_idx, _ = build_indexes(_chunks)
print("Ready.")

groq_client = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = "qwen/qwen3.6-27b"

class AskRequest(BaseModel):
    question: str
    skills: list[str] = []
    target_role: str = ''
    region: str = ''

@dataclass
class Deps:
    skills: list[str]
    target_role: str
    region: str = ''

SYSTEM = '''
You are DevPath, a tech career planning assistant.
You help developers plan their careers using data from:
- Stack Overflow Developer Survey 2024 (65,437 developers, 185 countries)
- Stack Overflow Developer Survey 2025 (49,191 developers, 177 countries)
- JetBrains Developer Ecosystem Survey 2025 (24,534 developers, 194 countries)
- O*NET 29.0 Database (U.S. Dept of Labor - global skills standard)
- WEF Future of Jobs Report 2025 (55 economies, regional breakdowns)

Rules:
- Always cite your source with specific numbers e.g. "According to SO Survey 2025, X% of data engineers use Python"
- Use specific numbers and percentages from search results - never make up stats
- When the user mentions a country or region, prioritize region-specific data
- Make at least 2 searches before answering
- Be practical and give concrete next steps
'''.strip()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the career knowledge base for relevant information about roles and skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_source",
            "description": "Search a specific source. Available: stackoverflow_2024, stackoverflow_2025, jetbrains_2025, onet_2024, wef_2025",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "source": {"type": "string", "description": "Source name"}
                },
                "required": ["query", "source"]
            }
        }
    }
]

SOURCE_MAP = {
    "stackoverflow": "stackoverflow_2024",
    "stackoverflow_2024": "stackoverflow_2024",
    "stackOverflow_2024": "stackoverflow_2024",
    "stackoverflow_2025": "stackoverflow_2025",
    "stackOverflow_2025": "stackoverflow_2025",
    "stack_overflow": "stackoverflow_2024",
    "jetbrains": "jetbrains_2025",
    "jetbrains_2025": "jetbrains_2025",
    "onet": "onet_2024",
    "onet_2024": "onet_2024",
    "wef": "wef_2025",
    "wef_2025": "wef_2025",
}

async def rewrite_query(question: str, deps: Deps) -> str:
    """Rewrite user question into search-optimized query."""
    context = f"Skills: {', '.join(deps.skills) or 'not specified'}. Target: {deps.target_role or 'not specified'}. Region: {deps.region or 'not specified'}."
    prompt = f'''Rewrite this career question as a short search query (5-8 words max) for a developer career knowledge base.
Return ONLY the search query, nothing else.

Context: {context}
Question: {question}
Search query:'''
    try:
        r = await groq_client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=30,
        )
        rewritten = r.choices[0].message.content.strip()
        rewritten = re.sub(r'<think>.*?</think>', '', rewritten, flags=re.DOTALL).strip()
        rewritten = rewritten.strip('"\'')
        return rewritten if len(rewritten) > 3 else question
    except Exception:
        return question

def run_tool(name: str, args: dict) -> str:
    with logfire.span('tool_call', tool=name, query=args.get('query', '')):
        if name == "search":
            query = args.get("query", "")
            results = hybrid_search_reranked(query, _text_idx, _vec_idx, n=3)
            return '\n\n'.join(f"[{r['source']}]\n{r['content'][:600]}" for r in results)
        elif name == "search_by_source":
            raw_source = args.get("source", "")
            source = SOURCE_MAP.get(raw_source, raw_source)
            query = args.get("query", "")
            results = hybrid_search_reranked(query, _text_idx, _vec_idx, n=5, candidate_pool=20)
            filtered = [r for r in results if r.get('source') == source]
            if not filtered:
                return '\n\n'.join(f"[{r['source']}]\n{r['content'][:600]}" for r in results[:3])
            return '\n\n'.join(f"[{r['source']}]\n{r['content'][:600]}" for r in filtered[:3])
        return "Unknown tool."

async def run_agent(question: str, deps: Deps) -> str:
    # Query rewriting - improves retrieval
    rewritten = await rewrite_query(question, deps)
    logfire.info('query_rewrite', original=question, rewritten=rewritten)

    context = (
        f"User skills: {', '.join(deps.skills) if deps.skills else 'not specified'}. "
        f"Target role: {deps.target_role or 'not specified'}. "
        f"Region: {deps.region or 'not specified'}."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{context}\n\nQuestion: {question}\nSearch hint: {rewritten}"}
    ]

    with logfire.span('agent_run', question=question, rewritten_query=rewritten, region=deps.region, target_role=deps.target_role):
        for iteration in range(6):
            try:
                with logfire.span('llm_call', iteration=iteration):
                    response = await groq_client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                    logfire.info('llm_response',
                        input_tokens=response.usage.prompt_tokens if response.usage else 0,
                        output_tokens=response.usage.completion_tokens if response.usage else 0,
                    )
            except Exception:
                try:
                    with logfire.span('llm_fallback'):
                        response = await groq_client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                        )
                    answer = response.choices[0].message.content or "No answer generated."
                    return re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
                except Exception as e:
                    return f"An error occurred: {str(e)}"

            msg = response.choices[0].message
            if not msg.tool_calls:
                answer = msg.content or "No answer generated."
                return re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            })

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

    return "Maximum search iterations reached."

class AgentShim:
    async def run(self, question: str, deps: Deps = None):
        if deps is None:
            deps = Deps(skills=[], target_role='', region='')
        output = await run_agent(question, deps)
        class Result:
            pass
        r = Result()
        r.output = output
        return r

agent = AgentShim()