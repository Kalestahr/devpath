import duckdb
from datetime import datetime

DB_PATH = 'devpath_pipeline.duckdb'


def init_tables():
    conn = duckdb.connect(DB_PATH)
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
    conn = duckdb.connect(DB_PATH)
    conn.execute(
        'INSERT INTO devpath.feedback_log VALUES (?, ?, ?)',
        [question, rating, datetime.now()]
    )
    conn.close()


def log_query_time(question: str, seconds: float):
    conn = duckdb.connect(DB_PATH)
    conn.execute(
        'INSERT INTO devpath.query_times VALUES (?, ?, ?)',
        [question, seconds, datetime.now()]
    )
    conn.close()


def get_feedback_log():
    conn = duckdb.connect(DB_PATH)
    rows = conn.execute(
        'SELECT question, rating, logged_at FROM devpath.feedback_log ORDER BY logged_at'
    ).fetchall()
    conn.close()
    return [{'question': r[0], 'rating': r[1], 'time': r[2].isoformat()} for r in rows]


def get_query_times():
    conn = duckdb.connect(DB_PATH)
    rows = conn.execute(
        'SELECT seconds FROM devpath.query_times ORDER BY logged_at'
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]