from dotenv import load_dotenv
load_dotenv()

import asyncio
import streamlit as st
import os, json

from agent import agent, Deps
from monitoring import init_tables, log_feedback, log_query_time, get_feedback_log, get_query_times

init_tables()

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
    st.divider()

    fb = get_feedback_log()
    times = get_query_times()
    n_queries = len(times)
    n_positive = sum(1 for f in fb if f["rating"] == 1)
    n_negative = sum(1 for f in fb if f["rating"] == -1)
    avg_time = round(sum(times) / len(times), 2) if times else 0

    st.caption("Totals below are across all sessions, not just this browser tab.")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", n_queries)
    col2.metric("Positive Feedback", n_positive)
    col3.metric("Negative Feedback", n_negative)
    col4.metric("Avg Response Time", f"{avg_time}s")

    st.divider()

    if times:
        import pandas as pd
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Response Times**")
            df_times = pd.DataFrame({"Query": range(1, len(times)+1), "Seconds": times})
            st.line_chart(df_times.set_index("Query"))

        with col2:
            st.markdown("**Feedback Distribution**")
            if n_positive + n_negative > 0:
                df_fb = pd.DataFrame({
                    "Type": ["Positive", "Negative"],
                    "Count": [n_positive, n_negative]
                })
                st.bar_chart(df_fb.set_index("Type"))
            else:
                st.caption("No feedback yet.")

    if fb:
        st.markdown("**Recent Feedback**")
        for f in fb[-5:]:
            emoji = "👍" if f["rating"] == 1 else "👎"
            st.caption(f"{emoji} {f['question'][:80]}...")

    st.divider()
    st.markdown("**Evaluation Results**")
    try:
        with open('data/processed/eval_results.json') as f:
            eval_results = json.load(f)
        selected = eval_results['best_retrieval_method']
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{selected.replace('_', ' ').title()} Hit Rate", eval_results[selected]['hit_rate'])
        col2.metric(f"{selected.replace('_', ' ').title()} MRR", eval_results[selected]['mrr'])
        col3.metric("Selected Method", selected)

        df_eval = {
            "Method": ["text_search", "vector_search", "hybrid (RRF)", "hybrid + rerank"],
            "Hit Rate": [eval_results['text']['hit_rate'], eval_results['vector']['hit_rate'], eval_results['hybrid']['hit_rate'], eval_results['hybrid_reranked']['hit_rate']],
            "MRR": [eval_results['text']['mrr'], eval_results['vector']['mrr'], eval_results['hybrid']['mrr'], eval_results['hybrid_reranked']['mrr']],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(df_eval), use_container_width=True)

        if 'rag_evaluation' in eval_results:
            st.markdown("**RAG Prompt Comparison (LLM-as-Judge)**")
            rag = eval_results['rag_evaluation']
            col1, col2 = st.columns(2)
            col1.metric("Prompt V1 (Concise)", f"{rag['prompt_v1_concise']['avg_score']}/5")
            col2.metric("Prompt V2 (Detailed+Citations)", f"{rag['prompt_v2_detailed']['avg_score']}/5")
            st.success(f"Winner: {rag['winner']} - used in production agent")
    except Exception:
        st.caption("Run rag/evaluate.py to see evaluation results.")

    st.divider()
    st.markdown("**Dataset Coverage**")
    data = {
        "Source": ["SO Survey 2024", "SO Survey 2025", "JetBrains 2025", "O*NET 29.0", "WEF 2025"],
        "Respondents": ["65,437", "49,191", "24,534", "900+ occupations", "55 economies"],
        "Countries": ["185", "177", "194", "US (global standard)", "55"],
        "Chunks": [34, 32, 18, 7, 96]
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(data), use_container_width=True)

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

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                col1, col2, _ = st.columns([1, 1, 8])
                if col1.button("Helpful", key=f"up_{i}"):
                    prev_q = st.session_state.messages[i-1]["content"] if i > 0 else ""
                    log_feedback(prev_q, 1)
                    st.toast("Thanks for the feedback!")
                if col2.button("Not helpful", key=f"dn_{i}"):
                    prev_q = st.session_state.messages[i-1]["content"] if i > 0 else ""
                    log_feedback(prev_q, -1)
                    st.toast("Thanks, we will improve!")

    if prompt := st.chat_input("Ask about your tech career..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
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