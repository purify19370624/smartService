"""
对话页自测（不调用大模型）

校验 Gradio messages 格式的「上屏 + 流式填充」：
Gradio 把 Chatbot 作为输入传给函数时，content 是 [{text, type}] 分段列表；
我们自己写入的是纯字符串。两种形态都必须能正确读出问题和累加回答。

运行：python test_chat_ui.py
"""
from types import SimpleNamespace

import main
import chat_store
from intent_classifier import Intent
import uuid


def _fake_chat(*args, **kwargs):
    """模拟闲聊链：分块产出 answer。"""
    yield {"answer": "你好"}
    yield {"answer": "，我在。"}


def _fake_rag(*args, **kwargs):
    """模拟 RAG 链：首个 chunk 带 context，之后带 answer。"""
    doc = SimpleNamespace(page_content="签收后 7 天内可无理由退货。", metadata={"source": "售后政策.docx"})
    yield {"context": [doc]}
    yield {"answer": "7 天内可以退货。"}


def _show(text, history=None):
    """调 submit_show 并取出更新后的聊天记录（第二个输出是 Chatbot 的更新对象）。"""
    return main.submit_show(text, history or [])[1]["value"]


def run():
    # 1. 空输入不追加消息（只回三个输出：输入框 / 聊天窗口 / 欢迎面板）
    empty_submit = main.submit_show("   ", [])
    assert len(empty_submit) == 3 and empty_submit[0] == ""

    # 2. 用户问题 + 空助手占位（messages 格式）
    history = _show("你好")
    assert [(m["role"], m["content"]) for m in history] == [("user", "你好"), ("assistant", "")], history

    # 打桩：不调大模型（意图分类现在由 llm_reply 并行调用，也要一并打桩）
    main.llm.stream = _fake_chat
    main.safety_check = lambda text, **kw: SimpleNamespace(is_safe=True)
    main.classify_intent = lambda text, **kw: Intent.CHAT

    # 3. 纯字符串 content（我们自己写出去的形态）
    final = list(main.llm_reply(history, ""))[-1][0]
    assert final[-1]["content"] == "你好，我在。", final[-1]["content"]

    # 4. Gradio 回传的分段列表 content（preprocess 后的形态）
    preprocessed = [
        {"role": "user", "content": [{"text": "你好", "type": "text"}]},
        {"role": "assistant", "content": [{"text": "", "type": "text"}]},
    ]
    assert main._msg_text(preprocessed[0]) == "你好"
    final = list(main.llm_reply(preprocessed, ""))[-1][0]
    assert final[-1]["content"] == "你好，我在。", final[-1]["content"]

    # 5. RAG 回答后附溯源
    main.llm.stream = _fake_rag
    history = _show("怎么退货？")
    final = list(main.llm_reply(history, ""))[-1][0]
    assert final[-1]["content"].startswith("7 天内可以退货。"), final[-1]["content"]
    assert "引用来源（1 个）" in final[-1]["content"] and "售后政策.docx" in final[-1]["content"]

    # 6. 上一条已答完再触发（空提交）不再调模型，也不覆盖上一条回答
    before = final[-1]["content"]
    assert list(main.llm_reply(final, "")) == []
    assert final[-1]["content"] == before

    # 7. 命中安全校验时只回提示
    main.safety_check = lambda text, **kw: SimpleNamespace(is_safe=False, source="test", matched_words=["x"], reason="")
    history = _show("坏话")
    final = list(main.llm_reply(history, ""))[-1][0]
    assert "安全检查" in final[-1]["content"]

    # 8. 对话记录：回答结束后写入数据库，点开能从库里恢复；新建对话拿到新 ID
    main.llm.stream = _fake_chat
    main.safety_check = lambda text, **kw: SimpleNamespace(is_safe=True)
    main.classify_intent = lambda text, **kw: Intent.CHAT
    sid = f"test-{uuid.uuid4().hex}"
    chat_store.delete_conversation(sid)
    history = _show("你好")
    history = list(main.llm_reply(history, sid))[-1][0]
    convs = main.save_conversation(history, sid)
    assert convs[0][1] == sid and sid in [s for _, s in convs], convs  # 最近更新的排最前
    assert convs[0][0] == "你好", convs[0][0]  # 侧栏只显示标题
    restored, got, _ = main.open_conversation(sid, [], "另一会话")
    assert got == sid and restored["value"] == history, (got, restored)
    assert main.open_conversation("不存在", ["x"], "保持")[1] == "保持"

    # 9. 删除某一行的对话：库里的消息、进程内的模型记忆、界面列表一起清
    convs2, chat2, sid2, _ = main.delete_conversation(sid, history, sid)
    assert chat2["value"] == [] and sid2 == "", (chat2, sid2)
    assert sid not in [s for _, s in convs2], convs2
    assert chat_store.load_messages(sid) == [] and main.llm.get_history_message(sid) == []

    empty, new_sid, _ = main.new_conversation()
    assert empty["value"] == [] and new_sid and new_sid != sid

    # 10. 满意度：没问答不能打分；有回答后能打、能改，并进运营看板统计
    sid = f"test-{uuid.uuid4().hex}"
    assert main.sync_rating([], sid)["visible"] is False          # 空对话不给打分
    assert chat_store.rate(sid, 5) is False and chat_store.get_rating(sid) is None
    history = _show("你好")
    history = list(main.llm_reply(history, sid))[-1][0]
    main.save_conversation(history, sid)
    assert main.sync_rating(history, sid)["visible"] is True
    assert chat_store.rate(sid, 2) is True and chat_store.get_rating(sid) == 2
    assert chat_store.rate(sid, 5) is True and chat_store.get_rating(sid) == 5   # 重复打分=改分
    assert main.sync_rating(history, sid)["value"] == 5                          # 切回对话回显分数
    assert "已记录 5 分" in main.rate_conversation(sid, 5)["label"]
    assert "还没有问答" in main.rate_conversation("", 5)["label"]
    stat = chat_store.stats()
    assert stat["conversations"] >= 1 and stat["qa"] >= 1 and stat["rated"] >= 1, stat
    assert stat["scores"].get(5, 0) >= 1 and stat["daily"], stat
    assert "满意度分布" in main.dashboard_html()
    chat_store.delete_conversation(sid)
    assert chat_store.get_rating(sid) is None                                     # 删对话连评分一起删

    print("[通过] 对话页自测完成：上屏 / 流式填充 / 溯源 / 空提交 / 安全拦截 / 对话记录 / 满意度")


if __name__ == "__main__":
    run()
