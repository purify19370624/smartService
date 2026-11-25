# 导入Gradio库，用于创建交互式Web应用程序
import gradio as gr
from loguru import logger
# 从llm模块中导入MyLLM类，这是自定义的大型语言模型接口
from combine_client import CombineClient
from intent_classifier import classify_intent
from safety_check import safety_check
from logger import setup_logger
from models import ALI_QWEN_MAX_MODEL
import chat_store
from concurrent.futures import ThreadPoolExecutor
from html import escape
import os
import threading
import uuid

# 确保环境变量已加载（特别是 Windows 系统下）
print("=" * 60)
print("检查环境变量配置...")
api_key = os.getenv("DASHSCOPE_API_KEY")
if api_key:
    # 控制台可能是 GBK 编码（重定向输出时尤其如此），所以不用 emoji
    print(f"[OK] DASHSCOPE_API_KEY 已设置 (长度: {len(api_key)})")
else:
    print("[警告] DASHSCOPE_API_KEY 未设置！")
    print("请检查系统环境变量或在代码中直接设置")
print("=" * 60)

# 【声明的属性】CHAT_MODEL：客户对话使用的对话模型名称。
CHAT_MODEL = ALI_QWEN_MAX_MODEL

# 【声明的属性】CHAT_MAX_LENGTH：单次回答的最大生成 token 数（传给模型的 max_tokens）。
# 值越大回答越完整，但会消耗更多 API 额度、增加首字延迟；
# 4000 在完整性与成本之间做了折中，覆盖客户场景的常见业务回答。
CHAT_MAX_LENGTH = 4000

# 【声明的属性】CHAT_TEMPERATURE：生成温度（采样随机性），取值范围 [0, 2]。
# 0.7 偏随机：回答更自然、有变化，适合客服对话；
# 越低越确定（如 0 用于工具调用/意图分类等需稳定输出的场景），越高越发散、易产生幻觉。
CHAT_TEMPERATURE = 0.7

# 【声明的属性】llm：全局唯一的 CombineClient 实例。
# 负责意图路由、知识库检索、会话历史管理等核心能力，
# 被 llm_reply 及知识库管理相关函数共同复用（进程内单例，避免重复加载知识库）。
llm = CombineClient()
# 初始化 loguru 日志（控制台 + 按天滚动的文件日志），仅需在启动时执行一次
setup_logger()

# 【声明的属性】_GATE_POOL：并行跑「安全校验 + 意图分类」的线程池。
# 两者互不依赖，串行等于白等一次大模型往返；并行能把这段首字延迟砍掉一半。
_GATE_POOL = ThreadPoolExecutor(max_workers=4)


def warm_up_knowledge():
    """后台把知识库索引建好（daemon 线程，失败只记日志，不影响对话）。"""
    try:
        llm.load_knowledge()
    except Exception as e:
        logger.exception(f"知识库预热失败: {str(e)}")

def _msg_text(message):
    """
    取一条聊天消息的纯文本内容。

    Gradio 把 Chatbot 作为输入传给我们时，content 是 [{text, type}] 分段列表
    （见 gradio Chatbot.preprocess）；我们自己写入的是字符串，两种都要能读。

    :param message: messages 格式的单条消息（dict，含 role/content）。
    :return: 该消息的纯文本（str）。
    """
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


# 定义submit函数，用于处理用户提交查询后处理页面展示信息
def submit_show(query, chat_history):
    """
    用户点击「提交」或回车后的第一步：把用户问题展示到聊天窗口，回答位置先占位。

    Gradio 事件链式调用 submit_show -> llm_reply：
    本函数只负责「上屏」，真正的回答生成由后续的 llm_reply 流式填充。

    :param query: 用户在输入框输入的问题文本（str）；为空表示未输入内容。
    :param chat_history: 当前聊天记录（messages 格式），每条为
                         {"role": "user"|"assistant", "content": str}。
    :return: 元组 (清空后的输入框, 聊天窗口更新, 欢迎面板更新)：
             - 输入框：始终返回空串 ''，即提交后清空输入框；
             - 聊天窗口：追加「用户问题 + 空助手消息」两条并显示（空对话时它是隐藏的），
               空消息等待 llm_reply 填充；
             - 欢迎面板：有对话了，隐藏。
    """
    if not query or not query.strip():
        return '', gr.update(), gr.update()
    # 返回新列表，避免原地改动 Gradio 持有的组件值
    history = (chat_history or []) + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": ""},
    ]
    return '', gr.update(value=history, visible=True), gr.update(visible=False)


# 将检索到的文档整理成溯源展示块
def build_source_block(docs, snippet_length=350):
    """
    把检索命中的知识库文档整理成 Markdown「引用来源」块，附在回答下方做溯源。

    同一来源文件只列一次（去重）。用纯 Markdown 而不是 <details>：Gradio 对话气泡
    按 Markdown 渲染，HTML 块里的 Markdown 不会被解析。

    :param docs: 检索命中的 langchain Document 对象列表（来自 RAG 链返回的 context）。
                 每个 Document 需包含 page_content（正文）与 metadata（source 路径、page 页码）。
    :param snippet_length: 单个片段摘录的最大字符数，默认 350；超长用省略号截断，
                           避免摘录过长撑爆对话界面。
    :return: 溯源 Markdown 字符串；docs 为空时返回空字符串。
    """
    if not docs:
        return ""

    seen = set()
    items = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source", "")
        # 同一文件只列一次，避免重复展示
        key = source or doc.page_content
        if key in seen:
            continue
        seen.add(key)

        title = os.path.basename(source) if source else "未知文档"
        if metadata.get("page") is not None:
            title += f"（第 {metadata['page']} 页）"

        snippet = " ".join((doc.page_content or "").split())
        if len(snippet) > snippet_length:
            snippet = snippet[:snippet_length] + "…"

        items.append(f"{len(items) + 1}. `{title}`：{snippet}")

    return "\n\n---\n**📚 引用来源（{} 个）**\n\n{}".format(len(items), "\n".join(items))


# 【声明的属性】WELCOME_HTML：空对话时的欢迎面板（图标用内联 SVG，不依赖 emoji 字体）。
# 图标沿用同一套线性风格，颜色走 currentColor，由 CSS 上色。
WELCOME_HTML = """
<div class="welcome-inner">
  <div class="welcome-icon">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round">
      <path d="M4 14v-2a8 8 0 0 1 16 0v2"/>
      <path d="M4 14a2 2 0 0 1 2-2h1v6H6a2 2 0 0 1-2-2z"/>
      <path d="M20 14a2 2 0 0 0-2-2h-1v6h1a2 2 0 0 0 2-2z"/>
      <path d="M18 18a4 4 0 0 1-4 3h-2"/>
    </svg>
  </div>
  <h2>欢迎使用电商智能客服</h2>
  <p class="welcome-sub">您的专属电商运营助手</p>
  <p class="welcome-desc">您好！我是您的智能客服助手，可以为您解答平台规则、商品管理、订单处理、物流配送及运营推广等问题。</p>
  <hr>
  <p class="welcome-label">您可以向我咨询：</p>
  <div class="welcome-grid">
    <div class="welcome-card">
      <div class="welcome-card-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M6 8h12l1 12H5L6 8z"/><path d="M9 8V6a3 3 0 0 1 6 0v2"/>
        </svg>
      </div>
      <b>商品管理</b><small>商品上架、ASIN、SKU</small>
    </div>
    <div class="welcome-card">
      <div class="welcome-card-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 8l-9-5-9 5v8l9 5 9-5V8z"/><path d="M3 8l9 5 9-5"/><path d="M12 13v8"/>
        </svg>
      </div>
      <b>订单物流</b><small>订单查询、FBA、FBM</small>
    </div>
    <div class="welcome-card">
      <div class="welcome-card-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-6"/><path d="M2 20h20"/>
        </svg>
      </div>
      <b>运营推广</b><small>广告优化、销量分析</small>
    </div>
    <div class="welcome-card">
      <div class="welcome-card-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 3l7 3v6c0 4-3 7-7 9-4-2-7-5-7-9V6l7-3z"/><path d="M9 12l2 2 4-4"/>
        </svg>
      </div>
      <b>平台规则</b><small>账户健康、合规政策</small>
    </div>
  </div>
  <p class="welcome-foot">请输入您的问题，让我为您提供帮助！</p>
</div>
"""

# 定义llm_reply函数，用于生成模型回复（客户对话页，走意图路由）
def llm_reply(chat_history, session_state):
    """
    模型回复生成函数（客户对话页核心业务逻辑），是一个生成器（含 yield）。

    完整流程：
        1. 取出用户最新提问 question；
        2. 会话 ID 管理：首次为当前浏览器会话分配 uuid；
        3. 安全合规校验与意图分类并行跑（省一次模型往返），违规直接拦截返回；
        4. 调用 llm.stream 流式生成回答（按意图路由：闲聊/工具/RAG/联网兜底）；
        5. 逐块累加回答文本，并通过 yield 持续刷新聊天界面（流式打字机效果）；
        6. 生成结束后，若有命中知识库文档，在回答下方附加溯源展示。

    :param chat_history: 当前聊天记录（messages 格式），每条为
                         {"role": "user"|"assistant", "content": str}。
                         本函数取最后一条用户消息作为提问，把流式回答累加到末尾助手消息。
    :param session_state: 当前会话 ID（str）。由 gr.State 在浏览器会话内持久保存，
                          首次为空串，此时分配 uuid.uuid4().hex 作为会话标识，
                          用于隔离不同用户的对话历史。
    :yield: 每次生成过程中 yield (chat_history, session_state)，
            把最新聊天记录与会话 ID 回传给 Gradio，实现流式刷新。
    """
    # 只填充 submit_show 刚追加的空占位回答；空提交或重复触发时直接退出，
    # 既不会白调一次大模型，也不会把上一条回答覆盖掉
    if not chat_history or chat_history[-1]["role"] != "assistant" or _msg_text(chat_history[-1]):
        return

    question = next((_msg_text(m) for m in reversed(chat_history) if m["role"] == "user"), "")

    # 首次使用时为当前会话分配唯一 ID（gr.State 每个浏览器会话独立，初始为空串）
    if not session_state:
        session_state = uuid.uuid4().hex
    logger.info(f'用户查询问题:{question} | 会话:{session_state}')

    # 安全合规校验 + 意图分类并行跑（都各自带兜底，不会抛到这里）
    safety_future = _GATE_POOL.submit(safety_check, question)
    intent_future = _GATE_POOL.submit(classify_intent, question)
    safety_result = safety_future.result()
    if not safety_result.is_safe:
        logger.warning(
            f"触发安全检查: source={safety_result.source}, "
            f"matched={safety_result.matched_words}, reason={safety_result.reason}"
        )
        chat_history[-1]["content"] = "⚠️ 您的提问触发了安全检查，请调整内容后再试。"
        yield chat_history, session_state
        return
    intent = intent_future.result()

    # 流式生成回答（知识库由路由层自动关联；异常由 CombineClient.stream 兜底）
    # 占位消息的 content 可能是 Gradio 的分段列表，先改写成字符串才能逐块累加
    chat_history[-1]["content"] = ""
    # 保存检索到的知识库文档，用于回答下方展示溯源信息
    retrieved_docs = []
    for chunk in llm.stream(
        question, None,
        model=CHAT_MODEL, max_length=CHAT_MAX_LENGTH, temperature=CHAT_TEMPERATURE,
        session_id=session_state, intent=intent
    ):
        if 'context' in chunk:
            # 首个 chunk 会带检索到的上下文文档，先保存下来
            retrieved_docs = chunk['context']
        if chunk.get('answer') is not None:
            chat_history[-1]["content"] += chunk['answer']
            yield chat_history, session_state

    # 回答生成完毕后，若有命中的知识库文档，在回答下方附加溯源展示
    source_block = build_source_block(retrieved_docs)
    if source_block:
        chat_history[-1]["content"] += source_block
        yield chat_history, session_state


# ---- 知识库管理 / 分块查看 辅助函数 ----

def _doc_update(docs):
    """
    生成文档选择下拉框的更新对象。

    :param docs: 文档名称列表（list[str]），来自 llm.list_documents()。
    :return: gr.update 对象：更新下拉框的 choices 为该列表，并默认选中第一项；
             docs 为空时 value 为 None（不选中任何项）。
    """
    return gr.update(choices=docs, value=docs[0] if docs else None)


def doc_table():
    """
    把知识库文档渲染成 Markdown 表格：文件名 / 大小 / 索引状态。

    比只看下拉框直观：能一眼看出库里有什么、哪些还没建索引。

    :return: Markdown 表格字符串；知识库为空时返回一句提示。
    """
    docs = llm.describe_documents()
    if not docs:
        return "知识库为空。上传的文档会列在这里，上传后自动在后台建索引。"
    lines = ["| 文档 | 大小 | 索引 |", "|---|---|---|"]
    for d in docs:
        lines.append(f"| {d['name']} | {d['size']} | {'已建索引' if d['indexed'] else '待建（后台建索引中）'} |")
    return "\n".join(lines)


def refresh_docs():
    """
    刷新「知识库管理」页的文档下拉框与文档表格。

    :return: 元组 (选择下拉框更新对象, 文档表格 Markdown)。
    """
    docs = llm.list_documents()
    return _doc_update(docs), doc_table()


def do_upload(temp_file):
    """
    处理「上传文档」事件：保存文档并刷新下拉框与文档表格。

    :param temp_file: Gradio File 组件上传的临时文件路径（str）；未选择时为 None。
    :return: 元组 (上传状态消息, 选择下拉框更新对象, 文档表格 Markdown)。
             状态消息由 llm.upload_document 返回（如「已上传: xxx」或「文件已存在，跳过」）。
    """
    msg, docs = llm.upload_document(temp_file)
    # 上传完就在后台把它索引好，别等用户第一条提问时才现建（大文档要几十秒）
    threading.Thread(target=warm_up_knowledge, daemon=True).start()
    if msg.startswith("已上传"):
        msg += "（正在后台建索引，完成后提问即生效）"
    return msg, _doc_update(docs), doc_table()


def do_delete(file_name, confirmed):
    """
    处理「删除文档」事件：需勾选确认后才会真正执行删除。

    :param file_name: 要删除的文档名称（str，来自文档下拉框选择）。
    :param confirmed: 是否勾选「确认删除」复选框（bool）。未勾选时不执行删除，
                      仅提示并刷新列表，防止误删。
    :return: 元组 (删除状态消息, 选择下拉框更新对象, 文档表格 Markdown)。
    """
    if not confirmed:
        return "⚠️ 请先勾选「确认删除」再点删除", _doc_update(llm.list_documents()), doc_table()
    msg, docs = llm.delete_document(file_name)
    return msg, _doc_update(docs), doc_table()


def do_view_chunks(file_name):
    """
    处理「查看分块」事件：读取所选文档的切分结果并渲染为 Markdown。

    :param file_name: 要查看的文档名称（str，来自分块页下拉框选择）。
    :return: Markdown 字符串，展示该文档所有分块的序号/元数据/内容；
             未选择文档或加载失败（如缺少 docx2txt）时返回提示文本。
    """
    if not file_name:
        return "请先选择一个文档。"
    chunks = llm.get_chunks(file_name)
    if not chunks:
        return f"文档「{file_name}」没有可展示的分块（可能是空文件或加载失败，例如缺少 docx2txt）。"

    lines = [f"### 文档「{file_name}」共 {len(chunks)} 个分块\n"]
    for c in chunks:
        lines.append(f"#### 分块 {c['序号']}（{c['元数据']}）\n\n{c['内容']}\n\n---\n")
    return "\n".join(lines)


# ---- 对话记录（左侧栏，落 db/conversations.db，重启不丢） ----


def _conv_choices():
    """
    生成「对话记录」单选框的选项：从数据库按最近更新倒序读。

    :return: [(label, session_id), ...]，Radio 显示标题、回传 session_id。
    """
    return [(c["title"], c["id"]) for c in chat_store.list_conversations()]


def new_conversation():
    """
    新建对话：清空对话区、隐藏聊天窗口、露出欢迎面板，并分配新的会话 ID。
    旧对话仍留在左侧记录里。

    :return: 元组 (聊天窗口更新, 新会话 ID, 欢迎面板更新)。
    """
    return gr.update(value=[], visible=False), uuid.uuid4().hex, gr.update(visible=True)


def refresh_conversations():
    """
    页面加载时把侧栏「对话记录」刷成数据库现状。

    侧栏行是构建时渲染的，不重读的话刷新页面就会退回启动那一刻的快照
    （新对话看不到，旧对话点不了，记录形同虚设）。

    :return: 会话列表 [(标题, session_id), ...]，作为 conv_state 的新值触发重新渲染。
    """
    return _conv_choices()


def save_conversation(chat_history, session_state):
    """
    把当前对话写入数据库（每轮回答结束后调用），并刷新侧栏。

    :param chat_history: 当前聊天记录（messages 格式）。
    :param session_state: 当前会话 ID（str）。
    :return: 会话列表 [(标题, session_id), ...]，作为 conv_state 的新值触发重新渲染。
    """
    if session_state and chat_history:
        # 存库前去掉回答里的「引用来源」块：它是给用户看的，别混进模型的历史上下文
        messages = [
            {"role": m["role"], "content": _msg_text(m).split("\n\n---\n**📚 引用来源")[0]}
            for m in chat_history
        ]
        first = next((m["content"] for m in messages if m["role"] == "user"), "")
        chat_store.save_conversation(session_state, (first.strip() or "新对话")[:20], messages)
    return _conv_choices()


def open_conversation(session_id, chat_history, session_state):
    """
    点左侧对话记录：从数据库恢复该对话（界面 + 模型的多轮记忆），并切回它的会话 ID。

    :param session_id: 选中的会话 ID（str）；未选中时为 None。
    :param chat_history: 当前聊天记录，选中项不存在时原样返回。
    :param session_state: 当前会话 ID，选中项不存在时原样返回。
    :return: 元组 (聊天窗口更新, 会话 ID, 欢迎面板更新)。
    """
    if not session_id:
        return gr.update(), session_state, gr.update()
    messages = chat_store.load_messages(session_id)
    if not messages:
        return gr.update(), session_state, gr.update()
    # 同步恢复模型侧历史，让"接着上次聊"真的能记住上文（窗口与 _add_to_history 一致）
    llm.restore_history(session_id, [(m["role"], m["content"]) for m in messages[-6:]])
    return gr.update(value=messages, visible=True), session_id, gr.update(visible=False)


def delete_conversation(conv_id, chat_history, session_state):
    """
    删除侧栏某一行：库里的消息、进程内的模型记忆、界面列表一起清掉。

    :param conv_id: 被点删除的会话 ID（str）；为空时什么都不做。
    :param chat_history: 当前聊天记录；删的不是当前会话时原样返回。
    :param session_state: 当前会话 ID；删的就是它时清空，下一次提问从新对话开始。
    :return: 元组 (会话列表, 聊天窗口更新, 会话 ID, 欢迎面板更新)。
    """
    if conv_id:
        chat_store.delete_conversation(conv_id)
        llm.restore_history(conv_id, [])  # 清掉进程内的模型记忆，别删了还"记得"
        if conv_id == session_state:
            return _conv_choices(), gr.update(value=[], visible=False), "", gr.update(visible=True)
    return _conv_choices(), gr.update(), session_state, gr.update()


# ---- 满意度评分 / 运营数据看板 ----

# 【声明的属性】RATE_LABEL：满意度控件的常态提示语（打分后会被「已记录」覆盖）。
RATE_LABEL = "本次对话满意度（1-5 分，点数字即提交）"


def _answered(chat_history):
    """
    这段聊天记录里有没有真正的回答（submit_show 追加的空占位不算）。

    :param chat_history: 当前聊天记录（messages 格式），可为 None。
    :return: 至少有一条带内容的助手消息返回 True。
    """
    return any(m["role"] == "assistant" and _msg_text(m).strip() for m in chat_history or [])


def sync_rating(chat_history, session_state):
    """
    把满意度控件切到当前对话的状态：有问答才显示，并回显这条对话已打的分。

    分数挂在 session_id 上（chat_store.ratings），所以点回左侧历史对话也能看到、改掉分数。

    :param chat_history: 当前聊天记录（messages 格式）。
    :param session_state: 当前会话 ID（str）。
    :return: 满意度 Radio 的更新对象（可见性 + 已打分数 + 提示语）。
    """
    answered = _answered(chat_history)
    return gr.update(
        visible=answered,
        value=chat_store.get_rating(session_state) if answered and session_state else None,
        label=RATE_LABEL,
    )


def rate_conversation(session_id, score):
    """
    点数字即提交评分（再点别的数字就是改分）。

    没有会话或还没有回答时拒绝写入：满意度是运营指标，不能让空对话进来凑数。

    :param session_id: 当前会话 ID（str）。
    :param score: 用户点的分数（1-5，int）；未选择时为 None。
    :return: 满意度 Radio 的更新对象（只改提示语，选中的分数保持用户点的那个）。
    """
    if not session_id or score is None or not chat_store.rate(session_id, int(score)):
        return gr.update(label="⚠️ 还没有问答，先提问再评分")
    return gr.update(label=f"✅ 已记录 {int(score)} 分，感谢反馈")


def _bars(rows):
    """
    把 [(标签, 数值)] 画成一列 HTML 横条（条宽按本组最大值归一）。

    :param rows: 标签与数值的二元组列表。
    :return: HTML 字符串；rows 为空时返回空串。
    """
    top = max((n for _, n in rows), default=0) or 1
    return "".join(
        f'<div class="bar-row"><span class="bar-label">{escape(str(k))}</span>'
        f'<span class="bar-track"><span class="bar-fill" style="width:{n / top * 100:.0f}%"></span></span>'
        f'<span class="bar-value">{n}</span></div>'
        for k, n in rows
    )


def dashboard_html():
    """
    运营数据看板：核心指标 + 满意度分布 + 近 14 天对话量 + 最近评分。

    数据全部现从 db/conversations.db 算（对话 / 消息 / 评分三张表）。

    shortcut: 日趋势只能按 conversations.updated_at 归日（消息表没有时间戳），
              一个会话多轮只算在最近更新那天，要按轮次统计再给 messages 加时间戳。

    :return: 看板 HTML；传到 gr.HTML 时由 Gradio 在每次打开页面时调用。
    """
    s = chat_store.stats()
    avg = "—" if s["avg_score"] is None else f'{s["avg_score"]:.2f}'
    cover = f'{s["rated"] * 100 // s["conversations"]}%' if s["conversations"] else "—"
    cards = [("对话总数", s["conversations"]), ("问答总量", s["qa"]), ("已评分对话", s["rated"]),
             ("评分覆盖率", cover), ("平均满意度", avg)]
    score_rows = [(f"{i} 分", s["scores"].get(i, 0)) for i in (5, 4, 3, 2, 1)]
    daily_rows = [(d[5:], n) for d, n in s["daily"]]
    recent = "".join(
        f"<tr><td>{escape(title)}</td><td>{score} 分</td><td>{at}</td></tr>"
        for title, score, at in s["recent"]
    ) or '<tr><td colspan="3">还没有评分</td></tr>'
    return f"""
<div class="dash">
  <div class="kpi-row">{''.join(
      f'<div class="kpi"><b>{v}</b><small>{k}</small></div>' for k, v in cards)}</div>
  <div class="dash-grid">
    <div class="dash-card"><h4>满意度分布</h4>{_bars(score_rows)}</div>
    <div class="dash-card"><h4>近 14 天对话量</h4>{_bars(daily_rows) or "<p>暂无数据</p>"}</div>
  </div>
  <div class="dash-card"><h4>最近评分</h4>
    <table class="dash-table"><tr><th>对话</th><th>满意度</th><th>时间</th></tr>{recent}</table>
  </div>
</div>
"""


# 【声明的属性】CSS：页面样式，对标 ChatGPT——左侧浅灰会话栏、中间窄对话列、底部圆角输入条。
# Gradio 6 的 gr.Blocks 不再接收 css，样式要在 launch(css=...) 里传。
CSS = """
footer {display: none !important;}
/* 顶部那条 "Home" 导航条：多页应用才有，对话页不需要 */
.nav-holder, .nav-holder-visible {display: none !important;}

/* ---- 整体：铺满窗口，去掉 Gradio 默认的居中留白 ---- */
.gradio-container {max-width: 100% !important; padding: 0 !important; margin: 0 !important;}

/* 只有对话页锁住整页滚动（一屏铺满）；/admin 保持正常页面滚动，否则管理页拉不下去 */
html:has(.app-row), body:has(.app-row) {height: 100%; overflow: hidden !important;}
.gradio-container:has(.app-row) {height: 100vh !important; overflow: hidden !important;}
.gradio-container:has(.app-row) > .main {padding: 0 !important; height: 100% !important;}
.app-row {gap: 0 !important; flex-wrap: nowrap !important; height: 100vh; align-items: stretch !important;}

/* ---- 左侧栏 ---- */
.side-col {background: #f9f9f9; border-right: 1px solid #ececec;
           padding: 12px 10px 14px; height: 100vh; overflow-y: auto; overflow-x: hidden; gap: 4px;}
.side-col .block {padding: 0 !important; border: none !important; background: transparent !important;
                  box-shadow: none !important;}
.side-title h3 {font-size: 15px; font-weight: 600; margin: 4px 8px 10px; color: #111;}
.new-chat {width: 100% !important; justify-content: flex-start !important; border-radius: 10px !important;
           background: transparent !important; border: none !important; color: #111 !important;
           box-shadow: none !important; font-weight: 500 !important; font-size: 14px !important;}
.new-chat:hover {background: #ececec !important;}

/* 对话记录：一行一条，鼠标移上来才显示行尾的删除键（对标 ChatGPT） */
.conv-rows {gap: 1px !important; padding: 2px 0 !important;}
.conv-row {flex-wrap: nowrap !important; align-items: center !important; gap: 0 !important;
           border-radius: 8px; padding: 0 !important; min-width: 0 !important;}
.conv-row:hover {background: #ececec !important;}
.conv-row.active {background: #e6e6e6 !important;}
.conv-row .block {padding: 0 !important; border: none !important; background: transparent !important;
                  box-shadow: none !important;}
.conv-item {flex: 1 1 auto !important; display: block !important; min-width: 0 !important;
            justify-content: flex-start !important; text-align: left !important;
            border: none !important; background: transparent !important; box-shadow: none !important;
            color: #202020 !important; font-size: 14px !important; font-weight: 400 !important;
            padding: 8px 4px 8px 10px !important; overflow: hidden; white-space: nowrap;
            text-overflow: ellipsis;}
.conv-del {flex: 0 0 30px !important; width: 30px !important; min-width: 30px !important;
           border: none !important; background: transparent !important; box-shadow: none !important;
           color: #8a8a8a !important; font-size: 17px !important; line-height: 1 !important;
           padding: 0 !important; opacity: 0; transition: opacity .12s;}
.conv-row:hover .conv-del, .conv-row:focus-within .conv-del {opacity: 1;}
.conv-del:hover {color: #c0392b !important;}

/* 侧栏底部（管理 / 看板入口），像 ChatGPT 的设置项；只有第一个吃掉上方的空白 */
.side-foot {margin-top: auto !important;}
.side-foot ~ .side-foot {margin-top: 0 !important;}
.side-foot, .side-foot button {width: 100% !important; justify-content: flex-start !important;
            border: none !important; background: transparent !important; color: #444 !important;
            box-shadow: none !important; font-size: 14px !important;}
.side-foot:hover {background: #ececec !important;}

/* ---- 满意度打分条：有回答才出现，贴在输入条上方 ---- */
.rate-row {padding: 0 !important; margin: 0 0 6px !important;}
.rate-row .block, .rate-row label {border: none !important; background: transparent !important;
            box-shadow: none !important; padding: 0 !important;}
.rate-row span[data-testid="block-info"], .rate-row .info {font-size: 13px !important; color: #6b7280 !important;}
.rate-row .wrap {gap: 10px !important;}

/* ---- 中间对话列 ---- */
.chat-col {max-width: 860px !important; margin: 0 auto !important; height: 100vh;
           padding: 6px 20px 12px !important; display: flex; flex-direction: column;
           overflow: hidden !important;}
.chat-col .block {border: none !important; box-shadow: none !important; background: transparent !important;}
.chat-window {flex: 1 1 auto !important; min-height: 0 !important;}
/* 气泡：用户灰底圆角，助手无底无边（和 ChatGPT 一样） */
.chat-window .message {border: none !important; box-shadow: none !important; border-radius: 18px !important;}
.chat-window .message.user {background: #f4f4f4 !important;}
.chat-window .message.bot {background: transparent !important;}

/* ---- 底部输入条：圆角胶囊 + 圆形发送键 ---- */
.composer {align-items: flex-end !important; gap: 6px; margin-top: auto; padding: 6px 8px;
           border: 1px solid #d9d9d9; border-radius: 26px; background: #fff;
           box-shadow: 0 2px 8px rgba(0, 0, 0, .05);}
.composer .block {padding: 0 !important; border: none !important; background: transparent !important;
                  box-shadow: none !important;}
.composer textarea {border: none !important; box-shadow: none !important; outline: none !important;
                    background: transparent !important; padding: 8px 2px !important; resize: none !important;
                    font-size: 15px !important;}
.composer label {border: none !important; background: transparent !important; box-shadow: none !important;}
.send-btn {width: 36px !important; min-width: 36px !important; height: 36px !important;
           border-radius: 50% !important; padding: 0 !important; font-size: 16px !important;
           line-height: 36px !important;}

/* ---- 空对话时的欢迎面板 ---- */
.welcome {flex: 1 1 auto; min-height: 0; display: flex; align-items: center; justify-content: center;
          overflow-y: auto; padding: 8px;}
.welcome .block {border: none !important; background: transparent !important;
                 box-shadow: none !important; padding: 0 !important;}
.welcome-inner {width: 100%; max-width: 700px; background: transparent; border: none;
                border-radius: 16px; padding: 24px 34px 20px; text-align: center; color: #202020;
                font-family: inherit;}
.welcome-icon {display: flex; justify-content: center;}
.welcome-icon svg {width: 44px; height: 44px; stroke: #2f6bff;}
.welcome-card-icon svg {width: 26px; height: 26px; stroke: #2f6bff;}
.welcome-inner h2 {font-size: 22px; font-weight: 700; margin: 10px 0 6px; color: #111;}
.welcome-sub {color: #6b7280; font-size: 15px; margin: 0 0 14px;}
.welcome-desc {font-size: 14px; line-height: 1.75; color: #374151; margin: 0;}
.welcome-inner hr {border: none; border-top: 1px solid #eee; margin: 16px 0 14px;}
.welcome-label {text-align: left; font-weight: 600; font-size: 15px; margin: 0 0 12px; color: #111;}
.welcome-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 14px 26px; text-align: left;}
.welcome-card {display: flex; flex-direction: column; gap: 3px;}
.welcome-card-icon {color: #2f6bff;}
.welcome-card-icon svg {width: 26px; height: 26px;}
.welcome-card b {font-size: 15px; color: #111;}
.welcome-card small {color: #6b7280; font-size: 13px;}
.welcome-foot {color: #6b7280; font-size: 13px; margin: 20px 0 0;}

/* ---- /admin 管理页（沿用窄列，但不要对话页的满屏高度） ---- */
.admin-col {max-width: 860px !important; margin: 0 auto !important; padding: 16px 20px !important;}
.chunks {max-height: 40vh; overflow-y: auto;}

/* ---- /dashboard 运营看板：指标卡 + 横条图 + 明细表（纯 CSS，不引图表库） ---- */
.dash {display: flex; flex-direction: column; gap: 16px; color: #202020;}
.kpi-row {display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;}
.kpi {background: #f7f8fa; border: 1px solid #ececf0; border-radius: 12px; padding: 12px 14px;
      display: flex; flex-direction: column; gap: 4px;}
.kpi b {font-size: 24px; font-weight: 700; line-height: 1.1;}
.kpi small {color: #6b7280; font-size: 12px;}
.dash-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 12px;}
.dash-card {border: 1px solid #ececf0; border-radius: 12px; padding: 14px 16px;}
.dash-card h4 {margin: 0 0 10px; font-size: 14px; font-weight: 600; color: #111;}
.bar-row {display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 13px;}
.bar-label {flex: 0 0 52px; color: #6b7280;}
.bar-track {flex: 1 1 auto; height: 14px; background: #f0f1f4; border-radius: 7px; overflow: hidden;}
.bar-fill {display: block; height: 100%; background: #2f6bff; border-radius: 7px;}
.bar-value {flex: 0 0 34px; text-align: right; color: #374151;}
.dash-table {width: 100%; border-collapse: collapse; font-size: 13px;}
.dash-table th, .dash-table td {text-align: left; padding: 6px 8px; border-bottom: 1px solid #f0f1f4;}
.dash-table th {color: #6b7280; font-weight: 500;}
"""

# 创建一个Gradio Blocks应用（左侧栏对话记录 + 对话主区；知识库管理是 /admin 独立页面）
with gr.Blocks(title="电商智能客服") as demo:
    with gr.Row(equal_height=False, elem_classes="app-row"):
        # ===== 左侧栏：对标 ChatGPT（新建对话 + 会话列表 + 底部管理入口） =====
        with gr.Column(scale=0, min_width=260, elem_classes="side-col"):
            gr.Markdown("### 电商智能客服", elem_classes="side-title")
            # 【声明的属性】new_chat：新建对话（旧对话留在左侧记录里）
            new_chat = gr.Button("＋ 新对话", elem_classes="new-chat")

            # 【声明的属性】conv_state：侧栏会话列表 [(标题, session_id), ...]（数据库现状），
            # 也是 gr.render 的渲染源——库一变，行就重建。
            conv_state = gr.State(_conv_choices())
            # 【声明的属性】open_sid / del_sid：行点击、行删除的中转态。
            # 行按钮在 render 里创建，那时 chatbot/session_state 还没建出来，
            # 所以行里只负责把 session_id 写进 State，真正的处理在页面底部绑定。
            open_sid = gr.State(None)
            del_sid = gr.State(None)

            # 会话列表：一行一条，鼠标移上来才显示右侧删除键（对标 ChatGPT）
            with gr.Column(elem_classes="conv-rows"):
                @gr.render(inputs=[conv_state, open_sid])
                def _render_conversations(convs, opened):
                    for title, sid in convs:
                        with gr.Row(elem_classes="conv-row active" if sid == opened else "conv-row"):
                            item = gr.Button(title, elem_classes="conv-item")
                            kill = gr.Button("×", elem_classes="conv-del")
                            item.click(fn=lambda s=sid: s, inputs=None, outputs=[open_sid])
                            kill.click(fn=lambda s=sid: s, inputs=None, outputs=[del_sid])

            # 管理入口放侧栏底部（对标 ChatGPT 左下角的设置项）
            admin_link = gr.Button("⚙ 知识库管理", elem_classes="side-foot")
            dashboard_link = gr.Button("📊 运营看板", elem_classes="side-foot")

        # ===== 对话主区（客户） =====
        with gr.Column(scale=1, elem_classes="chat-col"):
            # 【声明的属性】conv_pageload：对话页的加载触发器（空 HTML，只有 gr.HTML 有 load 事件）。
            # 侧栏选项是构建时算好的，页面一刷新/重进就会退回启动那一刻的快照（看起来像记录没了）。
            conv_pageload = gr.HTML("", container=False)

            # 【声明的属性】session_state：会话状态组件。gr.State 每个浏览器会话独立，
            # 用于存储当前会话 ID（实现不同用户对话历史的隔离），初始为空字符串。
            session_state = gr.State(value="")

            # 【声明的属性】welcome：空对话时的欢迎面板（有对话后隐藏，让位给聊天窗口）
            welcome = gr.HTML(WELCOME_HTML, elem_classes="welcome")

            # 【声明的属性】chatbot：聊天窗口组件，展示
            # [{"role": "user"|"assistant", "content": "..."}, ...] 形式的对话记录（Gradio 只支持
            # messages 格式）。layout="bubble" 即 ChatGPT 那种用户右、助手左的对话流。
            # 空对话时它是隐藏的，由上面的欢迎面板占位。
            chatbot = gr.Chatbot(
                show_label=False,
                layout="bubble",
                buttons=["copy", "like", "dislike"],
                height="72vh",
                visible=False,
                elem_classes="chat-window",
            )

            # 【声明的属性】rating：本次对话的满意度（1-5 分）。
            # 默认隐藏，由 sync_rating 跟着对话状态开关——没有问答就不给打分，免得污染运营数据。
            rating = gr.Radio(choices=[1, 2, 3, 4, 5], label=RATE_LABEL, visible=False,
                              elem_classes="rate-row")

            # 输入条：圆角胶囊 + 右侧圆形发送键
            with gr.Row(elem_classes="composer", equal_height=True):
                # 【声明的属性】user_input：问题输入框
                user_input = gr.Textbox(
                    placeholder="给 电商智能客服 发送消息…", show_label=False, container=False,
                    scale=9, lines=1, max_lines=8, autofocus=True
                )
                # 【声明的属性】user_submit：发送按钮（圆形 ↑，对标 ChatGPT）
                user_submit = gr.Button("↑", scale=0, min_width=40, elem_classes="send-btn")

    # ---- 对话页事件绑定 ----
    conv_pageload.load(fn=refresh_conversations, inputs=None, outputs=[conv_state])
    # 满意度控件跟着对话走：新建 / 切换 / 删除 / 每轮回答后都重算一次（有回答才出现）
    _sync = dict(fn=sync_rating, inputs=[chatbot, session_state], outputs=[rating])
    new_chat.click(fn=new_conversation, inputs=None, outputs=[chatbot, session_state, welcome]).then(
        **_sync)
    open_sid.change(fn=open_conversation, inputs=[open_sid, chatbot, session_state],
                    outputs=[chatbot, session_state, welcome]).then(**_sync)
    del_sid.change(fn=delete_conversation, inputs=[del_sid, chatbot, session_state],
                   outputs=[conv_state, chatbot, session_state, welcome]).then(**_sync)
    # 点数字即打分（用 input 而不是 change：change 也会被上面程序化的回显触发）
    rating.input(fn=rate_conversation, inputs=[session_state, rating], outputs=[rating])
    for _send in (user_input.submit, user_submit.click):
        _send(fn=submit_show, inputs=[user_input, chatbot],
              outputs=[user_input, chatbot, welcome]).then(
            fn=llm_reply, inputs=[chatbot, session_state], outputs=[chatbot, session_state]
        ).then(  # 回答结束后把这一轮存进左侧记录
            fn=save_conversation, inputs=[chatbot, session_state], outputs=[conv_state]
        ).then(**_sync)
    admin_link.click(fn=None, js="() => {window.location.href = '/admin'}")
    dashboard_link.click(fn=None, js="() => {window.location.href = '/dashboard'}")

# ===== 知识库管理页（独立路由，从左侧「管理」进入） =====
with demo.route("知识库管理", "/admin", show_in_navbar=False):
    with gr.Column(elem_classes="admin-col"):
        with gr.Row():
            gr.Markdown("### 知识库管理")
            back_btn = gr.Button("← 返回对话", size="sm", scale=0)

        # 导入说明：把文档放进 chroma/knowledge/ 也行，上传只是复制一份进去
        gr.Markdown(
            "**导入**：上传 `doc/docx/csv/txt/pdf/md` 文档，或直接把文件放进 `chroma/knowledge/` 后点刷新。\n\n"
            "**索引**：上传后自动在后台建索引（大文档几十秒），启动时也会预热；建好后提问直接命中。"
        )
        # 【声明的属性】admin_upload_file：文件上传组件，限定可上传的文档类型
        admin_upload_file = gr.File(label="上传文档", file_types=['.doc', '.docx', '.csv', '.txt', '.pdf', '.md'])
        # 【声明的属性】admin_upload_status：上传结果回显（只读文本框，不可交互）
        admin_upload_status = gr.Textbox(label="上传状态", interactive=False)

        with gr.Row():
            gr.Markdown("#### 现有文档")
            # 【声明的属性】admin_refresh_btn：手动刷新文档列表按钮
            admin_refresh_btn = gr.Button("刷新", size="sm", scale=0)
        # 【声明的属性】admin_doc_table：文档概览表（文件名 / 大小 / 索引状态）。
        # 传函数而不是字符串：字符串是构建时算好的，函数每次页面加载都重新读磁盘。
        admin_doc_table = gr.Markdown(doc_table)

        # 【声明的属性】admin_doc：删除与分块查看共用的文档选择器
        admin_doc = gr.Dropdown(choices=llm.list_documents(), label="选择文档")
        # 【声明的属性】admin_pageload：页面加载触发器（空 HTML，只有 gr.HTML 有 load 事件）。
        # 用它把下拉选项也刷成磁盘现状：隐藏(visible="hidden")的组件收不到 load 事件。
        admin_pageload = gr.HTML("", container=False)
        with gr.Row():
            # 【声明的属性】admin_confirm：删除前的确认勾选框（防止误删）
            admin_confirm = gr.Checkbox(label="确认删除所选文档", value=False)
            # 【声明的属性】admin_view_btn：触发查看分块的按钮
            admin_view_btn = gr.Button("查看分块", size="sm", scale=0)
            # 【声明的属性】admin_delete_btn：删除按钮，variant="stop" 表示红色危险操作
            admin_delete_btn = gr.Button("删除所选文档", variant="stop", size="sm", scale=0)
        # 【声明的属性】admin_delete_status：删除结果回显（只读文本框）
        admin_delete_status = gr.Textbox(label="删除状态", interactive=False)

        # 【声明的属性】admin_chunk_output：分块结果的 Markdown 展示区
        admin_chunk_output = gr.Markdown(elem_classes="chunks")

    back_btn.click(fn=None, js="() => {window.location.href = '/'}")
    # ---- 知识库管理事件绑定 ----
    # 打开/刷新本页时从磁盘重读文档列表：组件的值/选项是应用启动时算好的，
    # 上传/删除只改当前会话，不重读的话刷新页面会退回启动那一刻的快照（看起来像没保存、没法查看）。
    # 只有 gr.HTML 有页面加载(load)事件，借一个空 HTML 当触发器。
    admin_pageload.load(fn=refresh_docs, inputs=None, outputs=[admin_doc, admin_doc_table])
    admin_upload_file.upload(fn=do_upload, inputs=[admin_upload_file],
                             outputs=[admin_upload_status, admin_doc, admin_doc_table])
    admin_refresh_btn.click(fn=refresh_docs, inputs=None, outputs=[admin_doc, admin_doc_table])
    admin_delete_btn.click(fn=do_delete, inputs=[admin_doc, admin_confirm],
                           outputs=[admin_delete_status, admin_doc, admin_doc_table])
    admin_view_btn.click(fn=do_view_chunks, inputs=[admin_doc], outputs=[admin_chunk_output])

# ===== 运营数据看板（独立路由，从左侧「看板」进入） =====
with demo.route("运营看板", "/dashboard", show_in_navbar=False):
    with gr.Column(elem_classes="admin-col"):
        with gr.Row():
            gr.Markdown("### 运营数据看板")
            dash_back = gr.Button("← 返回对话", size="sm", scale=0)
        # 传函数而不是字符串：每次打开本页都重新从库里算
        gr.HTML(dashboard_html)
    dash_back.click(fn=None, js="() => {window.location.href = '/'}")

# 启动 Gradio 应用（只在直接运行时启动，便于被测试脚本 import）
if __name__ == "__main__":
    # 启动即预热：建索引要打很多次 embedding（一个 7MB PDF 上千个块，几十秒），
    # 放到后台线程里做——用户提问只查已建好的库，不再让第一条消息替建索引一起等。
    threading.Thread(target=warm_up_knowledge, daemon=True, name="kb-warmup").start()
    demo.launch(css=CSS)
