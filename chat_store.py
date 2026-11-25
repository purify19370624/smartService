"""
对话记录持久化（SQLite）

把左侧「对话记录」与每轮问答落到 db/conversations.db，重启项目后仍能列出并恢复，
不再是只存在内存里的摆设。

表结构：
    conversations(id, title, updated_at)   —— 一个会话一行，updated_at 用于倒序排列
    messages(id, session_id, role, content) —— 会话里的每条消息，按 id 递增即对话顺序

用法：
    from chat_store import save_conversation, load_messages, list_conversations
"""
import os
import sqlite3
from datetime import datetime
from typing import List

from loguru import logger

# 【声明的属性】DB_PATH：对话库文件路径（与 record_manager_cache.db 同目录）。
DB_PATH = os.path.join('./db', 'conversations.db')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


def _connect():
    """打开连接并确保表存在（每次操作独立连接，Gradio 的回调在不同线程里跑）。"""
    os.makedirs(os.path.dirname(DB_PATH) or '.', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def save_conversation(session_id: str, title: str, messages: List[dict]) -> None:
    """
    整轮覆盖保存一个会话（数据量很小，直接全量重写，省掉增量同步的复杂度）。

    :param session_id: 会话 ID（str）。
    :param title: 会话标题（一般取第一条用户消息，截断后的）。
    :param messages: 消息列表，每条 {"role": "user"|"assistant", "content": str}。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations(id, title, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at",
            (session_id, title, now),
        )
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.executemany(
            "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, m["role"], m["content"]) for m in messages],
        )
    logger.info(f"对话已入库: {session_id}（{len(messages)} 条消息）")


def load_messages(session_id: str) -> List[dict]:
    """
    读取一个会话的全部消息。

    :param session_id: 会话 ID（str）。
    :return: 消息列表 [{"role": ..., "content": ...}, ...]；会话不存在时返回空列表。
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def list_conversations(limit: int = 100) -> List[dict]:
    """
    列出最近的会话，按更新时间倒序。

    :param limit: 最多返回多少个会话（默认 100）。
    :return: [{"id": ..., "title": ..., "updated_at": ...}, ...]。
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"id": i, "title": t, "updated_at": u} for i, t, u in rows]


def delete_conversation(session_id: str) -> None:
    """删除一个会话及其消息（目前供自测清理用）。"""
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
