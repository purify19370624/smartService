from typing import Iterable

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import AddableDict
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from loguru import logger

from intent_classifier import Intent, classify_intent
from knowledge import MyKnowledge
from models import get_lc_model_client, ALI_QWEN_MAX_MODEL
from tools import TOOLS

# 【声明的属性】RERANK_SCORE_THRESHOLD：重排分阈值（模块级配置常量）。
# RAG 问答时，若召回文档中的最高重排分低于此值，视为"知识库没有依据"，改用联网搜索回答。
# 取值 [0,1]：越小越宽松（更容易用知识库作答），越大越严格（更容易转联网），可按业务调整。
RERANK_SCORE_THRESHOLD = 0.3

# 【声明的属性】qa_system_prompt：RAG 知识库问答的系统提示词。
# {context} 会被替换为检索到的文档内容；要求模型"不知道就答没找到"，
# 以约束模型基于检索上下文作答、减少幻觉。
qa_system_prompt = """你是一名电商知识问答助手，
              你将使用检索到的上下文来回答问题。如果你不知道答案，就说你没有找到答案。

              {context}"""
# 【声明的属性】qa_prompt：RAG 问答的完整提示模板（system + 历史 + 用户问题）。
# 三个占位符：{context}（检索文档）、{chat_history}（会话历史）、{input}（当前问题），
# 由 create_stuff_documents_chain 在生成时填充。
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# 【声明的属性】history_prompt：专给"查改写后问题"用的历史（归一成 user/assistant 两条发言）。
# 原因：检索要用改写后的问题，但提示模板里已有 {chat_history}，若历史再带上 current_question，
# 模型会看到两次提问、把当前问题当成上文，反复输出改写而不是回答。
history_prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("human", "{current_question}"),
    ]
)

# 【声明的属性】REWRITE_SYSTEM：多轮指代消解提示词。
# 把"它/这个/那刚才那个"等指代，结合上轮对话补全成可独立检索的完整问题。
REWRITE_SYSTEM = "把用户最新的一句话改写成可独立检索的完整问句，只补全省略和指代（如“它”“这个”），" \
                 "不改动原意，不要回答问题，只输出改写后的一句话。"


def rewrite_question(chat, history, question):
    """
    多轮指代消解（查询改写）：结合上文把追问补成完整问题，供检索与联网使用。

    :param chat: 已创建的模型客户端（沿用当前会话的模型与温度）。
    :param history: 该会话的历史消息列表（list）。
    :param question: 用户本轮原始提问（str）。
    :return: 改写后的问题；无历史或改写失败/异常时原样返回。
    """
    if not history:
        return question
    try:
        out = (history_prompt | chat).invoke({"chat_history": history, "current_question": question})
        rewritten = (out.content or "").strip().strip('"“”')
        logger.info(f"查询改写: {question} -> {rewritten}")
        return rewritten or question
    except Exception as e:
        logger.warning(f"查询改写失败，使用原问题: {str(e)}")
        return question


# 【声明的属性】chat_prompt：闲聊意图专用提示模板。
# 引导模型用友好、简短的一句话回应寒暄，不展开长篇内容（降低闲聊成本与啰嗦感）。
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是电商平台的智能客服助手。用户只是在简单寒暄闲聊，"
                   "请用友好、简短的一句话回应，不要展开长篇内容。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

def streaming_parse(chunks: Iterable[AIMessageChunk]):
    """
    流式解析器：把模型输出的 AIMessageChunk 流逐块转成 AddableDict。

    用于普通问答链（非 RAG）的流式输出，把每个 chunk 的内容包装成
    {'answer': ...} 结构，使下游（Gradio 的 llm_reply）能统一按 'answer' 键读取。

    :param chunks: 模型流式输出的 AIMessageChunk 可迭代对象（生成器）。
    :yield: 每个 chunk 生成一个 AddableDict({'answer': 文本片段})。
    """
    for chunk in chunks:
        yield AddableDict({'answer': chunk.content})

class CombineClient(MyKnowledge):
    """
    负责和大模型进行交互，并支持聊天历史记录；负责和知识库进行交互
    """
    # 【声明的属性】__chat_histories：会话历史表（类级共享字典，前置双下划线做私有化）。
    # 键为 session_id（会话标识），值为该会话独立的 ChatMessageHistory 对象，
    # 实现多用户/多会话之间的对话历史隔离、互不影响。
    __chat_histories = {}

    def _get_history(self, session_id):
        """
        获取某个会话（session_id）独立的历史对象；不存在则创建一个新的。

        :param session_id: 会话标识（str），用于区分不同用户/浏览器的对话。
        :return: 该会话对应的 ChatMessageHistory 对象。
        """
        if session_id not in self.__chat_histories:
            self.__chat_histories[session_id] = ChatMessageHistory()
        return self.__chat_histories[session_id]

    def restore_history(self, session_id, pairs):
        """
        用一个会话已落库的 (role, content) 对恢复模型侧历史，重启后还能接着上文聊。

        :param session_id: 会话标识（str）。
        :param pairs: [(role, content), ...]，role 为 "user" 或 "assistant"。
        """
        history = self._get_history(session_id)
        history.clear()
        for role, content in pairs:
            if role == "user":
                history.add_user_message(content)
            else:
                history.add_ai_message(content)

    def stream(self, question, collection=None, model=ALI_QWEN_MAX_MODEL, max_length=256, temperature=1, session_id="default", intent=None):
        """
        用户消息统一入口（生成器）：先做意图分类，再路由到不同处理链。

        路由规则：
            human   → 转人工提示；
            chat    → 闲聊链（_chat_stream）；
            service → 工具调用链（_service_stream）；
            其他    → 业务咨询 RAG 链（_rag_stream）。
        任一路线抛异常时统一兜底，返回友好错误提示，避免整个服务崩溃。

        :param question: 用户问题文本（str）。
        :param collection: 指定知识库名称（str，默认 None），为空时业务咨询链路用全部可用知识库。
        :param model: 对话模型名（默认 ALI_QWEN_MAX_MODEL）。
        :param max_length: 最大生成 token 数（默认 256）。
        :param temperature: 生成温度（默认 1）。
        :param session_id: 会话标识，用于隔离不同用户的对话历史（默认 "default"）。
        :param intent: 已算好的意图（Intent）；调用方并行算过就传进来，省一次模型往返。
        :yield: 每条流式结果以 AddableDict({'answer': ...}) 或 AddableDict({'context': ...}) 产出。
        """
        try:
            intent = intent or classify_intent(question)
            logger.info(f"意图识别结果: {intent.value} | 会话: {session_id} | 问题: {question}")

            if intent == Intent.HUMAN:
                yield AddableDict({'answer': "已为您转接人工客服，请稍候，客服将尽快接入..."})
                return
            if intent == Intent.CHAT:
                yield from self._chat_stream(question, model, max_length, temperature, session_id)
                return
            if intent == Intent.SERVICE:
                yield from self._service_stream(question, model, session_id)
                return
            # 业务咨询 -> RAG
            yield from self._rag_stream(question, collection, model, max_length, temperature, session_id)
        except Exception as e:
            logger.exception(f"处理用户消息失败: {str(e)}")
            yield AddableDict({'answer': "抱歉，系统处理您的消息时出现异常，请稍后再试或转人工客服。"})

    def _resolve_collections(self, collection):
        """
        解析业务咨询应使用的知识库：优先用传入值，否则回退到全部可用知识库。

        :param collection: 传入的知识库名称（str），可能为空、"无可用知识库"或具体库名。
        :return: 知识库名称列表（list[str]）；没有任何可用知识库时返回空列表。
        """
        if collection and collection != "无可用知识库":
            return [collection]
        try:
            # build=False：只用已建好索引的库，不在提问链路上现建（建索引很慢，交给后台预热）
            choices = self.load_knowledge(build=False)
        except Exception as e:
            logger.warning(f"加载知识库列表失败: {str(e)}")
            return []
        return [c for c in choices if c != "无可用知识库"]

    def _add_to_history(self, session_id, question, answer):
        """
        把一轮「问题-回答」写入指定会话的历史，并用滑动窗口限制长度。

        :param session_id: 会话标识（str）。
        :param question: 用户问题（str）。
        :param answer: 模型回答（str）。
        :return: None（仅更新会话历史；历史超过 6 条时仅保留最近 6 条）。
        """
        history = self._get_history(session_id)
        history.add_user_message(question)
        history.add_ai_message(answer)
        if len(history.messages) > 6:
            history.messages = history.messages[-6:]

    def _chat_stream(self, question, model, max_length, temperature, session_id):
        """
        闲聊链路：用 chat_prompt 友好、简短地回应寒暄，带会话历史流式输出。

        :param question: 用户问题（str）。
        :param model: 对话模型名（str）。
        :param max_length: 最大生成 token 数。
        :param temperature: 生成温度。
        :param session_id: 会话标识（str）。
        :yield: 流式回答 chunk（AddableDict）。
        """
        chat = get_lc_model_client(model=model, max_tokens=max_length, temperature=temperature)
        chain = chat_prompt | chat | streaming_parse
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda sid: self._get_history(sid),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        for chunk in chain_with_history.stream(
            {"input": question},
            {"configurable": {"session_id": session_id}},
        ):
            yield chunk

    def _service_stream(self, question, model, session_id):
        """
        办理/查询类链路：让模型绑定工具（查天气/查时间）进行工具调用。

        流程：模型决定调用哪个工具 → 执行工具 → 把结果回传模型生成最终回答。
        带该会话历史实现多轮记忆，工具调用失败时兜底返回友好提示。

        :param question: 用户问题（str）。
        :param model: 对话模型名（str）。注意：这里固定 temperature=0 以求稳定输出。
        :param session_id: 会话标识（str）。
        :yield: 最终回答（AddableDict，一次性产出）。
        """
        chat = get_lc_model_client(model=model, temperature=0)
        llm_with_tools = chat.bind_tools(TOOLS)
        # 带上该会话的历史，实现多轮记忆
        messages = list(self._get_history(session_id).messages) + [HumanMessage(content=question)]
        try:
            resp = llm_with_tools.invoke(messages)
        except Exception as e:
            logger.exception(f"工具调用失败: {str(e)}")
            yield AddableDict({'answer': "抱歉，工具调用出现异常，请稍后再试或转人工客服。"})
            return

        tool_calls = getattr(resp, "tool_calls", None)
        if tool_calls:
            tool_map = {t.name: t for t in TOOLS}
            tool_messages = []
            for tc in tool_calls:
                tool = tool_map.get(tc["name"])
                if tool is None:
                    continue
                try:
                    output = tool.invoke(tc["args"])
                except Exception as e:
                    output = f"工具执行失败: {str(e)}"
                tool_messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
            final = llm_with_tools.invoke(messages + [resp, *tool_messages])
            answer = final.content
        else:
            answer = resp.content

        answer = answer or "抱歉，我没有理解您的查询，请换一种说法。"
        self._add_to_history(session_id, question, answer)
        yield AddableDict({'answer': answer})

    def _web_answer_stream(self, question, model, max_length, temperature, session_id, query=None):
        """
        联网兜底链路：知识库没有可用依据时，用模型内置联网搜索回答，并附搜索来源链接。

        :param question: 用户问题（str），写入历史用。
        :param model: 对话模型名（str）。
        :param max_length: 最大生成 token 数。
        :param temperature: 生成温度。
        :param session_id: 会话标识（str）。
        :param query: 实际送去联网检索的问题（多轮已消解指代）；None 时用 question。
        :yield: 首块前先说明来源，再逐块 yield {'answer': 文本}，末尾带引用来源。
        """
        chat = get_lc_model_client(model=model, max_tokens=max_length, temperature=temperature, search=True)
        messages = list(self._get_history(session_id).messages) + [HumanMessage(content=query or question)]
        parts = []
        sources = {}  # 标题 -> 链接；百炼把联网检索结果放在 additional_kwargs["search_info"]
        for chunk in chat.stream(messages):
            text = chunk if isinstance(chunk, str) else chunk.content
            for info in (getattr(chunk, "additional_kwargs", None) or {}).get("search_info") or []:
                sources.setdefault(info.get("title") or info.get("site_name"), info.get("url"))
            if not text:
                continue
            if not parts:
                # 拿到第一个字才提示来源，模型调用失败时不会留下一句半截话
                yield AddableDict({'answer': "（知识库没有找到依据，以下为联网搜索结果）\n\n"})
            parts.append(text)
            yield AddableDict({'answer': text})

        # 口说无凭，把模型参考的网页列出来（与知识库回答的「引用来源」一致）
        links = [f"- [{t}]({u})" for t, u in sources.items() if u]
        if links:
            yield AddableDict({'answer': "\n\n---\n**🌐 联网来源**\n\n" + "\n".join(links)})
        self._add_to_history(session_id, question, "".join(parts))

    def _rag_stream(self, question, collection, model, max_length, temperature, session_id):
        """
        业务咨询链路：优先用知识库作答；知识库没有依据时改用联网搜索。

        流程：
            1. 解析要用的知识库（默认是全部）；2. 逐库预检索、按重排分合并候选；
            3. 最高重排分低于阈值 → 联网搜索回答；4. 有依据则基于命中文档流式生成回答并附溯源。

        :param question: 用户问题（str）。
        :param collection: 指定知识库名称（str）；为空时用全部可用知识库。
        :param model: 对话模型名（str）。
        :param max_length: 最大生成 token 数。
        :param temperature: 生成温度。
        :param session_id: 会话标识（str）。
        :yield: 首个 chunk 带 {'context': docs} 供溯源，随后逐块 yield {'answer': 文本}。
        """
        collections = self._resolve_collections(collection)
        if not collections:
            logger.info("没有已建好索引的知识库（可能还在后台建），本次改用联网搜索回答")
            yield from self._web_answer_stream(question, model, max_length, temperature, session_id)
            return

        # 多轮指代消解：追问里的"它/这个"要结合上文补全，否则检索出来的是别的商品/政策
        history = self._get_history(session_id).messages
        chat = get_lc_model_client(model=model, max_tokens=max_length, temperature=temperature)
        query = rewrite_question(chat, history, question)

        # 预检索：每个知识库各召回一批（各自已重排），再按重排分合并成一个候选集
        try:
            docs = []
            for name in collections:
                retriever = self.get_retrievers(name)
                if retriever is None:
                    logger.warning(f"知识库 [{name}] 索引不可用，跳过")
                    continue
                docs.extend(retriever.invoke(query))
        except Exception as e:
            logger.exception(f"知识库检索失败: {str(e)}")
            yield AddableDict({'answer': "抱歉，知识库检索出现异常，请稍后再试或转人工客服。"})
            return

        # 每个库自己已重排取 top_n=3，合起来再去掉跨库的低分尾巴
        docs.sort(key=lambda d: d.metadata.get("relevance_score", 0.0), reverse=True)
        docs = docs[:3]

        # 知识库没有依据：改用联网搜索，不强行编造
        top_score = max((d.metadata.get("relevance_score", 0.0) for d in docs), default=0.0)
        if not docs or top_score < RERANK_SCORE_THRESHOLD:
            logger.info(
                f"知识库无依据（最高重排分 {top_score:.4f} < {RERANK_SCORE_THRESHOLD}），"
                f"改用联网搜索。collections={collections}"
            )
            # 联网用消解过指代的问题，历史里仍记用户原话
            yield from self._web_answer_stream(question, model, max_length, temperature, session_id, query=query)
            return

        # 首个 chunk 带 context，供前端溯源展示
        yield AddableDict({'context': docs})

        # 基于命中的文档直接生成回答，确保"回答有依据"
        qa_chain = create_stuff_documents_chain(chat, qa_prompt)
        answer_parts = []
        for chunk in qa_chain.stream(
            {"context": docs, "input": query, "chat_history": history}
        ):
            text = chunk if isinstance(chunk, str) else chunk.content
            if text:
                answer_parts.append(text)
                yield AddableDict({'answer': text})
        # 写入历史，实现多轮记忆
        self._add_to_history(session_id, question, "".join(answer_parts))

    def clear_history(self, session_id=None) -> None:
        """
        清空会话历史。

        :param session_id: 会话标识（str）；为 None 时清空所有会话的历史，
                           否则只清空指定会话。
        :return: None。
        """
        if session_id is None:
            self.__chat_histories.clear()
        else:
            self._get_history(session_id).clear()

    def get_history_message(self, session_id="default"):
        """
        获取指定会话的历史消息列表。

        :param session_id: 会话标识（str，默认 "default"）。
        :return: 该会话的历史消息列表（list，元素为 HumanMessage/AIMessage 等）。
        """
        return self._get_history(session_id).messages
