import sqlite3
import os
from datetime import datetime

DB_PATH = "memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS object_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_name TEXT,
            first_seen TEXT,
            last_seen TEXT,
            duration_seconds INTEGER,
            status TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS llm_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            scene_description TEXT,
            suggestion TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS action_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action_type TEXT,
            detail TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_object(object_name, first_seen, last_seen, duration_seconds, status):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO object_log (object_name, first_seen, last_seen, duration_seconds, status)
        VALUES (?, ?, ?, ?, ?)
    ''', (object_name, first_seen, last_seen, duration_seconds, status))
    conn.commit()
    conn.close()

def log_llm(scene_description, suggestion):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO llm_log (timestamp, scene_description, suggestion)
        VALUES (?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), scene_description, suggestion))
    conn.commit()
    conn.close()

def log_action(action_type, detail):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO action_log (timestamp, action_type, detail)
        VALUES (?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action_type, detail))
    conn.commit()
    conn.close()

def search_objects(query):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT object_name, first_seen, last_seen, duration_seconds, status
        FROM object_log
        WHERE object_name LIKE ?
        ORDER BY last_seen DESC
        LIMIT 20
    ''', (f"%{query}%",))
    results = cursor.fetchall()
    conn.close()
    return results

def get_recent_logs(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, action_type, detail
        FROM action_log
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    results = cursor.fetchall()
    conn.close()
    return results

def get_latest_llm():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, suggestion
        FROM llm_log
        ORDER BY timestamp DESC
        LIMIT 1
    ''')
    result = cursor.fetchone()
    conn.close()
    return result