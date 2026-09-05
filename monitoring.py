import duckdb
import time
from datetime import datetime

DB_PATH = 'devpath_pipeline.duckdb'


def _connect_with_retry(retries: int = 3, delay: float = 0.3):
    # docker-compose runs the api and ui services against the same duckdb
    # file, and DuckDB needs an exclusive lock for writes, so a brief
    # collision between the two services (or between Streamlit reruns) is
    # expected occasionally. A few short retries ride out that window
    # instead of surfacing a TransactionException to the user.
    last_error = None
    for attempt in range(retries):
        try:
            return duckdb.connect(DB_PATH)
        except duckdb.TransactionException as e:
            last_error = e
            time.sleep(delay)
    raise last_error


def init_tables():
    conn = _connect_with_retry()
    conn.execute('CREATE SCHEMA IF NOT EXISTS devpath')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS devpath.feedback_log (
            question VARCHAR,
            rating INTEGER,
            logged_at TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS devpath.query_times (
            question VARCHAR,
            seconds DOUBLE,
            logged_at TIMESTAMP
        )
    ''')
    conn.close()


def log_feedback(question: str, rating: int):
    conn = _connect_with_retry()
    conn.execute(
        'INSERT INTO devpath.feedback_log VALUES (?, ?, ?)',
        [question, rating, datetime.now()]
    )
    conn.close()


def log_query_time(question: str, seconds: float):
    conn = _connect_with_retry()
    conn.execute(
        'INSERT INTO devpath.query_times VALUES (?, ?, ?)',
        [question, seconds, datetime.now()]
    )
    conn.close()


def get_feedback_log():
    conn = _connect_with_retry()
    rows = conn.execute(
        'SELECT question, rating, logged_at FROM devpath.feedback_log ORDER BY logged_at'
    ).fetchall()
    conn.close()
    return [{'question': r[0], 'rating': r[1], 'time': r[2].isoformat()} for r in rows]


def get_query_times():
    conn = _connect_with_retry()
    rows = conn.execute(
        'SELECT seconds FROM devpath.query_times ORDER BY logged_at'
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]