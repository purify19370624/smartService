"""
对话记录持久化（SQLite）

把左侧「对话记录」与每轮问答落到 db/conversations.db，重启项目后仍能列出并恢复，
不再是只存在内存里的摆设。

表结构：
    conversations(id, title, updated_at)   —— 一个会话一行，updated_at 用于倒序排列
    messages(id, session_id, role, content) —— 会话里的每条消息，按 id 递增即对话顺序
    ratings(session_id, score, updated_at) —— 每个会话一条满意度（1-5 分），重复打分即改分

用法：
    from chat_store import save_conversation, load_messages, list_conversations
"""
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

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
CREATE TABLE IF NOT EXISTS ratings (
    session_id TEXT PRIMARY KEY,
    score INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
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
    """删除一个会话及其消息、满意度评分（目前供自测清理用）。"""
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM ratings WHERE session_id = ?", (session_id,))


def rate(session_id: str, score: int) -> bool:
    """
    给一个会话的问答打 1-5 分；重复打分即改分。

    没有助手回答的会话不记录：满意度是运营指标，不能被空对话稀释。

    :param session_id: 会话 ID（str）。
    :param score: 分数，只能是 1-5（int），越界抛 ValueError。
    :return: 是否记录成功（会话没有回答时返回 False）。
    """
    if score not in (1, 2, 3, 4, 5):
        raise ValueError(f"满意度必须是 1-5 分: {score}")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        answered = conn.execute(
            "SELECT 1 FROM messages WHERE session_id = ? AND role = 'assistant' "
            "AND TRIM(content) <> '' LIMIT 1",
            (session_id,),
        ).fetchone()
        if not answered:
            return False
        conn.execute(
            "INSERT INTO ratings(session_id, score, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET score=excluded.score, updated_at=excluded.updated_at",
            (session_id, score, now),
        )
    return True


def get_rating(session_id: str) -> Optional[int]:
    """读一个会话已打的分；没打过返回 None。"""
    with _connect() as conn:
        row = conn.execute("SELECT score FROM ratings WHERE session_id = ?", (session_id,)).fetchone()
    return row[0] if row else None


def stats(days: int = 14) -> dict:
    """
    运营看板的一次性汇总（全部现从库里算，数据量小，不做缓存）。

    :param days: 「每日对话量」看最近多少天（默认 14）。
    :return: dict，含 conversations（对话数）/ qa（问答轮次）/ rated（已评分对话数）/
             avg_score（平均分，无评分为 None）/ scores（{分数: 条数}）/
             daily（[(日期, 对话数)] 按日期正序）/ recent（[(标题, 分数, 时间)]）。
    """
    with _connect() as conn:
        conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        qa = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE role = 'assistant' AND TRIM(content) <> ''"
        ).fetchone()[0]
        rated, avg_score = conn.execute("SELECT COUNT(*), AVG(score) FROM ratings").fetchone()
        scores = dict(conn.execute("SELECT score, COUNT(*) FROM ratings GROUP BY score").fetchall())
        daily = conn.execute(
            "SELECT substr(updated_at, 1, 10) AS d, COUNT(*) FROM conversations "
            "GROUP BY d ORDER BY d DESC LIMIT ?",
            (days,),
        ).fetchall()
        recent = conn.execute(
            "SELECT c.title, r.score, r.updated_at FROM ratings r "
            "JOIN conversations c ON c.id = r.session_id ORDER BY r.updated_at DESC LIMIT 10"
        ).fetchall()
    return {
        "conversations": conversations,
        "qa": qa,
        "rated": rated,
        "avg_score": avg_score,
        "scores": scores,
        "daily": sorted(daily),
        "recent": recent,
    }
