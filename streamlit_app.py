from dotenv import load_dotenv
load_dotenv()

import asyncio
import streamlit as st
import os, json

from agent import agent, Deps
from monitoring import init_tables, log_feedback, log_query_time, get_feedback_log, get_query_times, get_query_log

@st.cache_resource
def _ensure_monitoring_tables():
    init_tables()
    return True

_ensure_monitoring_tables()

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(
    page_title='DevPath',
    page_icon='📍',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .stApp, .stApp > div { background-color: #0D1117 !important; }
    section[data-testid="stSidebar"] { background-color: #010409 !important; border-right: 1px solid #21262D; }
    p, label, span, .stMarkdown p { color: #C9D1D9 !important; }
    h1, h2, h3 { color: #F0F6FC !important; }
    a { color: #58A6FF !important; }
    hr { border-color: #21262D !important; }
    .stButton button { background-color: #161B22 !important; color: #C9D1D9 !important; border-color: #30363D !important; }
    .dp-title { font-size: 5rem !important; font-weight: 900 !important; letter-spacing: -0.04em; color: #F0F6FC !important; line-height: 1 !important; margin: 0 0 0.5rem 0 !important; }
    .dp-sub { font-size: 1.05rem; color: #8B949E !important; margin: 0 0 1rem 0; }
    .dp-badge { display: inline-flex; align-items: center; gap: 7px; background: #161B22; border: 1px solid #30363D; border-radius: 20px; padding: 5px 14px 5px 10px; font-size: 0.8rem; color: #8B949E !important; margin-bottom: 1rem; }
    .dp-dot { width: 8px; height: 8px; border-radius: 50%; background: #3FB950; flex-shrink: 0; }
    .ex-card { background: #161B22; border: 1px solid #21262D; border-radius: 10px; padding: 16px; font-size: 0.9rem; color: #8B949E !important; line-height: 1.5; }
    .src-label { font-size: 0.65rem !important; font-weight: 700 !important; letter-spacing: 0.1em; text-transform: uppercase; color: #484F58 !important; }
</style>
""", unsafe_allow_html=True)

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "chat"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Your profile")
    st.caption("Fill in your details for more personalized answers")
    st.divider()
    skills_input = st.text_input("Current skills", placeholder="e.g. Python, SQL, Excel")
    target_role = st.selectbox("Target role", [
        "", "Data Engineer", "Data Scientist", "ML Engineer",
        "Backend Developer", "Frontend Developer", "Full-stack Developer",
        "DevOps Engineer", "Cloud Engineer", "Security Engineer", "Mobile Developer",
    ])
    region = st.text_input("Country or region", placeholder="e.g. Philippines, Germany")
    st.divider()
    st.markdown('<p class="src-label">Data sources</p>', unsafe_allow_html=True)
    st.markdown("[Stack Overflow Survey 2024](https://survey.stackoverflow.co/2024/)")
    st.markdown("[Stack Overflow Survey 2025](https://survey.stackoverflow.co/2025/)")
    st.markdown("[JetBrains Ecosystem 2025](https://devecosystem-2025.jetbrains.com/)")
    st.markdown("[O\*NET 29.0 — US Dept of Labor](https://www.onetcenter.org/database.html)")
    st.markdown("[WEF Future of Jobs 2025](https://www.weforum.org/publications/the-future-of-jobs-report-2025/)")
    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("Chat", use_container_width=True):
        st.session_state.page = "chat"
    if col2.button("Stats", use_container_width=True):
        st.session_state.page = "stats"

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Stats Page ────────────────────────────────────────────────────────────────
if st.session_state.page == "stats":
    st.markdown('<p class="dp-title">DevPath</p>', unsafe_allow_html=True)
    st.markdown("## Monitoring Dashboard")

    fb = get_feedback_log()
    times = get_query_times()
    n_positive = sum(1 for f in fb if f["rating"] == 1)
    n_negative = sum(1 for f in fb if f["rating"] == -1)
    avg_time = round(sum(times) / len(times), 2) if times else 0

    import pandas as pd

    tab_usage, tab_retrieval, tab_dataset = st.tabs(["Live Usage", "Retrieval & Prompts", "Dataset"])

    with tab_usage:
        st.caption("Live activity across every visitor to this app (not just your own browser tab), updates as people use it.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", len(times))
        col2.metric("Positive Feedback", n_positive)
        col3.metric("Negative Feedback", n_negative)
        col4.metric("Avg Response Time", f"{avg_time}s")

        if times:
            st.markdown("**Response Times**")
            df_times = pd.DataFrame({"Query": range(1, len(times)+1), "Seconds": times})
            st.line_chart(df_times.set_index("Query"))
        else:
            st.caption("No queries logged yet - ask something in Chat to see this fill in.")

        if n_positive + n_negative > 0:
            st.markdown("**Feedback Distribution**")
            df_fb = pd.DataFrame({
                "Type": ["Positive", "Negative"],
                "Count": [n_positive, n_negative]
            })
            st.bar_chart(df_fb.set_index("Type"))
        else:
            st.markdown("**Feedback Distribution**")
            st.caption("No feedback yet - go to Chat, ask a question, then click Helpful or Not helpful under the answer to populate this.")

        if times:
            st.markdown("**Response Time Distribution**")
            buckets = ["<2s", "2-5s", "5-10s", "10-20s", "20s+"]
            counts = [0, 0, 0, 0, 0]
            for t in times:
                if t < 2:
                    counts[0] += 1
                elif t < 5:
                    counts[1] += 1
                elif t < 10:
                    counts[2] += 1
                elif t < 20:
                    counts[3] += 1
                else:
                    counts[4] += 1
            df_buckets = pd.DataFrame({"Range": buckets, "Queries": counts})
            df_buckets["Range"] = pd.Categorical(df_buckets["Range"], categories=buckets, ordered=True)
            st.bar_chart(df_buckets.set_index("Range"))

        query_log = get_query_log()
        if query_log:
            df_q = pd.DataFrame(query_log)
            df_q["date"] = pd.to_datetime(df_q["time"]).dt.date
            st.markdown("**Queries per Day**")
            daily_queries = df_q.groupby("date").size().rename("Queries")
            st.bar_chart(daily_queries)

        if fb:
            df_fb_time = pd.DataFrame(fb)
            df_fb_time["date"] = pd.to_datetime(df_fb_time["time"]).dt.date
            df_fb_time["type"] = df_fb_time["rating"].map({1: "Positive", -1: "Negative"})
            daily_fb = df_fb_time.groupby(["date", "type"]).size().unstack(fill_value=0)
            if not daily_fb.empty:
                st.markdown("**Feedback per Day**")
                st.bar_chart(daily_fb)
        else:
            st.markdown("**Feedback per Day**")
            st.caption("No feedback yet - go to Chat, ask a question, then click Helpful or Not helpful under the answer to populate this.")

        if fb:
            st.markdown("**Recent Feedback**")
            for f in fb[-5:]:
                emoji = "👍" if f["rating"] == 1 else "👎"
                st.caption(f"{emoji} {f['question'][:80]}...")

    with tab_retrieval:
        st.caption("One-time evaluation results from rag/evaluate.py - these don't change as people use the app.")

        try:
            with open('data/processed/eval_results.json') as f:
                eval_results = json.load(f)
            selected = eval_results['best_retrieval_method']

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{selected.replace('_', ' ').title()} Hit Rate", eval_results[selected]['hit_rate'])
            col2.metric(f"{selected.replace('_', ' ').title()} MRR", eval_results[selected]['mrr'])
            col3.metric("Selected Method", selected)

            with st.expander("Compare all retrieval methods", expanded=True):
                df_eval = pd.DataFrame({
                    "Method": ["text_search", "vector_search", "hybrid (RRF)", "hybrid + rerank"],
                    "Hit Rate": [eval_results['text']['hit_rate'], eval_results['vector']['hit_rate'], eval_results['hybrid']['hit_rate'], eval_results['hybrid_reranked']['hit_rate']],
                    "MRR": [eval_results['text']['mrr'], eval_results['vector']['mrr'], eval_results['hybrid']['mrr'], eval_results['hybrid_reranked']['mrr']],
                })
                st.dataframe(df_eval, use_container_width=True)
                st.bar_chart(df_eval.set_index("Method"))

            if 'rag_evaluation' in eval_results:
                st.divider()
                st.markdown("**RAG Prompt Comparison (LLM-as-Judge)**")
                rag = eval_results['rag_evaluation']
                v1 = rag['prompt_v1_concise']['avg_score']
                v2 = rag['prompt_v2_detailed']['avg_score']
                improvement = round(v2 - v1, 2)
                pct = round((improvement / v1) * 100) if v1 else 0
                col1, col2 = st.columns(2)
                col1.metric("Prompt V1 (Concise)", f"{v1}/5")
                col2.metric("Prompt V2 (Detailed+Citations)", f"{v2}/5", delta=f"+{improvement} ({pct}% better than V1)")
                st.success(f"Winner: {rag['winner']} - used in production agent")
                st.caption("Scored 1-5 by an LLM judge on relevance and accuracy using a strict rubric, so scores in the 2-3 range are typical, not a sign the answers are poor.")
                with st.expander("See the score breakdown as a chart", expanded=True):
                    df_rag = pd.DataFrame({
                        "Prompt": ["V1: concise", "V2: detailed + citations"],
                        "Avg Score": [v1, v2],
                    })
                    st.bar_chart(df_rag.set_index("Prompt"))
        except Exception:
            st.caption("Run rag/evaluate.py to see evaluation results.")

    with tab_dataset:
        st.caption("What's in the knowledge base right now.")
        data = {
            "Source": ["SO Survey 2024", "SO Survey 2025", "JetBrains 2025", "O*NET 29.0", "WEF 2025"],
            "Respondents": ["65,437", "49,191", "24,534", "900+ occupations", "55 economies"],
            "Countries": ["185", "177", "194", "US (global standard)", "55"],
            "Chunks": [34, 32, 18, 7, 96]
        }
        df_data = pd.DataFrame(data)
        st.dataframe(df_data, use_container_width=True)
        st.markdown("**Chunks per Source**")
        st.bar_chart(df_data.set_index("Source")["Chunks"])

# ── Chat Page ─────────────────────────────────────────────────────────────────
else:
    st.markdown('<p class="dp-title">DevPath</p>', unsafe_allow_html=True)
    st.markdown('<p class="dp-sub">Tech career planning grounded in real developer data from 65,000+ respondents across 185 countries</p>', unsafe_allow_html=True)
    st.markdown('<div class="dp-badge"><span class="dp-dot"></span>SO 2024 · SO 2025 · JetBrains 2025 · O*NET 29.0 · WEF 2025</div>', unsafe_allow_html=True)
    st.divider()

    if len(st.session_state.messages) == 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="ex-card">How do I become a data engineer with Python and SQL?</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="ex-card">What skills are most in demand for ML engineers in 2025?</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="ex-card">Is data science a viable career in Southeast Asia?</div>', unsafe_allow_html=True)
        st.divider()

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    is_generating = st.session_state.pending_prompt is not None

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                col1, col2, _ = st.columns([1, 1, 8])
                # Disabled while a response is generating - clicking any
                # button mid-generation would cancel the in-flight agent.run()
                # call, since Streamlit only runs one script pass at a time.
                if col1.button("Helpful", key=f"up_{i}", disabled=is_generating):
                    prev_q = st.session_state.messages[i-1]["content"] if i > 0 else ""
                    log_feedback(prev_q, 1)
                    st.toast("Thanks for the feedback!")
                if col2.button("Not helpful", key=f"dn_{i}", disabled=is_generating):
                    prev_q = st.session_state.messages[i-1]["content"] if i > 0 else ""
                    log_feedback(prev_q, -1)
                    st.toast("Thanks, we will improve!")

    if not is_generating:
        if prompt := st.chat_input("Ask about your tech career..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.pending_prompt = prompt
            st.rerun()
    else:
        st.chat_input("Ask about your tech career...", disabled=True)
        prompt = st.session_state.pending_prompt
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.caption("Searching knowledge base...")
            import time
            start = time.time()
            try:
                deps = Deps(
                    skills=[s.strip() for s in skills_input.split(",") if s.strip()],
                    target_role=target_role,
                    region=region
                )
                result = asyncio.run(agent.run(prompt, deps=deps))
                answer = result.output
            except Exception as e:
                answer = f"Error: {str(e)}"
            elapsed = round(time.time() - start, 2)
            log_query_time(prompt, elapsed)
            placeholder.empty()
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.pending_prompt = None
        st.rerun()