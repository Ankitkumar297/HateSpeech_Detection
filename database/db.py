import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

DB_DIR = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    DB_PATH = os.path.join("/tmp", "predictions.db")
else:
    DB_PATH = os.path.join(DB_DIR, "predictions.db")

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    except Exception:
        pass
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_text TEXT NOT NULL,
            clean_text TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT DEFAULT '127.0.0.1'
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction ON predictions(prediction)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)")
    conn.commit()

    # Seed realistic initial data if table is empty
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    if count == 0:
        seed_sample_data(conn)
    conn.close()

def seed_sample_data(conn):
    sample_records = [
        ("Sustainable green energy solutions are essential to combating global climate change.", "sustainable green energy solutions essential combating global climate change", "Neutral Language", 98.4, {"Hate Speech": 0.5, "Offensive Language": 1.1, "Neutral Language": 98.4}, -12),
        ("Shut the fuck up you annoying idiot, nobody asked for your opinion!", "shut fuck annoying idiot nobody asked opinion", "Offensive Language", 96.8, {"Hate Speech": 2.1, "Offensive Language": 96.8, "Neutral Language": 1.1}, -10),
        ("Go back to your country you filthy immigrants, you do not belong here!", "go back country filthy immigrants belong", "Hate Speech", 99.1, {"Hate Speech": 99.1, "Offensive Language": 0.6, "Neutral Language": 0.3}, -9),
        ("The artificial intelligence research paper was accepted at the annual conference.", "artificial intelligence research paper accepted annual conference", "Neutral Language", 99.5, {"Hate Speech": 0.2, "Offensive Language": 0.3, "Neutral Language": 99.5}, -8),
        ("What a piece of shit movie, waste of damn money.", "piece shit movie waste damn money", "Offensive Language", 94.2, {"Hate Speech": 1.8, "Offensive Language": 94.2, "Neutral Language": 4.0}, -6),
        ("Death to all minorities, burn down their neighborhoods now!", "death minorities burn neighborhoods", "Hate Speech", 98.9, {"Hate Speech": 98.9, "Offensive Language": 0.8, "Neutral Language": 0.3}, -5),
        ("Thank you so much for your kind assistance and quick reply.", "thank much kind assistance quick reply", "Neutral Language", 99.2, {"Hate Speech": 0.3, "Offensive Language": 0.5, "Neutral Language": 99.2}, -4),
        ("Damn it, I spilled hot coffee all over my new white shirt!", "damn spilled hot coffee new white shirt", "Offensive Language", 91.5, {"Hate Speech": 1.2, "Offensive Language": 91.5, "Neutral Language": 7.3}, -3),
        ("We went for a beautiful hike in the national park and took many photographs.", "went beautiful hike national park took many photographs", "Neutral Language", 99.7, {"Hate Speech": 0.1, "Offensive Language": 0.2, "Neutral Language": 99.7}, -2),
        ("You dirty ethnic trash, you will never be equal to real citizens.", "dirty ethnic trash never equal real citizens", "Hate Speech", 97.6, {"Hate Speech": 97.6, "Offensive Language": 1.9, "Neutral Language": 0.5}, -1),
        ("Learning a new language opens up fascinating cultural perspectives and friendships.", "learning new language opens fascinating cultural perspectives friendships", "Neutral Language", 99.1, {"Hate Speech": 0.4, "Offensive Language": 0.5, "Neutral Language": 99.1}, 0)
    ]
    cursor = conn.cursor()
    now = datetime.now()
    for text, clean, pred, conf, probs, hour_offset in sample_records:
        t_stamp = (now + timedelta(hours=hour_offset)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO predictions (user_text, clean_text, prediction, confidence, probabilities, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (text, clean, pred, conf, json.dumps(probs), t_stamp))
    conn.commit()

def insert_prediction(user_text, clean_text, prediction, confidence, probabilities, ip_address="127.0.0.1"):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO predictions (user_text, clean_text, prediction, confidence, probabilities, timestamp, ip_address)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_text, clean_text, prediction, confidence, json.dumps(probabilities), now_str, ip_address))
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pred_id

def get_history(limit=10, offset=0, search=None, category=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if search:
        query += " AND user_text LIKE ?"
        params.append(f"%{search}%")
    
    if category and category != "All":
        query += " AND prediction = ?"
        params.append(category)
        
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Get total count with same filters
    count_query = "SELECT COUNT(*) FROM predictions WHERE 1=1"
    count_params = []
    if search:
        count_query += " AND user_text LIKE ?"
        count_params.append(f"%{search}%")
    if category and category != "All":
        count_query += " AND prediction = ?"
        count_params.append(category)
    
    cursor.execute(count_query, count_params)
    total_count = cursor.fetchone()[0]

    history = []
    for r in rows:
        history.append({
            "id": r["id"],
            "user_text": r["user_text"],
            "clean_text": r["clean_text"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "probabilities": json.loads(r["probabilities"]) if r["probabilities"] else {},
            "timestamp": r["timestamp"]
        })
    conn.close()
    return {
        "items": history,
        "total": total_count,
        "limit": limit,
        "offset": offset
    }

def delete_prediction(prediction_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def get_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Hate Speech'")
    hate_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Offensive Language'")
    offensive_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Neutral Language'")
    neutral_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = cursor.fetchone()[0] or 0.0

    # Hourly / daily trend data (last 7 days or recent distribution)
    cursor.execute("""
        SELECT strftime('%Y-%m-%d', timestamp) as day, 
               COUNT(*) as total_day,
               SUM(CASE WHEN prediction = 'Hate Speech' THEN 1 ELSE 0 END) as hate_day,
               SUM(CASE WHEN prediction = 'Offensive Language' THEN 1 ELSE 0 END) as offensive_day,
               SUM(CASE WHEN prediction = 'Neutral Language' THEN 1 ELSE 0 END) as neutral_day
        FROM predictions 
        GROUP BY day 
        ORDER BY day DESC 
        LIMIT 7
    """)
    trend_rows = cursor.fetchall()
    trends = [dict(r) for r in reversed(trend_rows)]

    conn.close()
    return {
        "total_predictions": total,
        "hate_speech_count": hate_count,
        "offensive_language_count": offensive_count,
        "neutral_count": neutral_count,
        "average_confidence": round(float(avg_conf), 2),
        "trends": trends
    }

def get_all_for_export():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, prediction, confidence, user_text FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
