# 电商智能客服系统

> 基于 RAG（检索增强生成）+ 意图路由的电商智能客服，支持多格式文档解析、精准知识检索、多意图分派、内容安全合规与知识库可视化管理。

## ✨ 核心功能

- **多意图路由**：轻量意图分类（小模型分类 + 关键词规则兜底），把用户消息分派到业务咨询 / 工具调用 / 闲聊 / 转人工四条链路。
- **RAG 问答（有据可依）**：业务咨询优先走文档知识库检索，**有依据就用文档作答并附溯源**；知识库里没有依据时改用**联网搜索**兜底，并标注结果来源。
- **工具调用**：办理类/查询类意图走工具调用，内置查天气、查时间两个演示工具。
- **内容安全合规**：两级校验——先敏感词匹配，未命中再交大模型二次判断，违规内容直接拦截。
- **ChatGPT 式界面**：左侧浅灰会话栏（新对话 / 历史会话 / 底部管理入口），中间窄对话列 + 底部圆角输入条（圆形发送键），用户气泡靠右、助手纯文本靠左。
- **知识库可视化管理**：上传、列表、删除文档，并可查看文档切分后的分块。

## 🏗️ 请求链路图

下图是**一次用户提问**从进入到返回答案的完整链路：

```mermaid
flowchart TD
    A[用户提问<br/>Gradio 对话页] --> B["llm_reply<br/>main.py"]
    B --> C["安全合规校验<br/>safety_check"]
    C -- "命中敏感词 / 违规" --> C1["返回安全检查提示"]
    C -- "安全" --> D["意图分类<br/>classify_intent"]
    D --> D1{"意图类型"}
    D1 -- "human 人工请求" --> E["转人工提示"]
    D1 -- "chat 闲聊" --> F["闲聊链<br/>简短友好回应"]
    D1 -- "service 办理/查询" --> G["工具调用链<br/>bind_tools"]
    G --> G1["get_weather / get_time"]
    D1 -- "business_qa 业务咨询" --> H["RAG 链"]
    H --> H1["解析默认知识库<br/>_resolve_collection"]
    H1 --> H2["预检索<br/>retriever.invoke"]
    H2 -- "知识库无依据" --> H3["联网搜索兜底<br/>标注来源"]
    H2 -- "有文档" --> H4["create_stuff_documents_chain<br/>基于文档生成回答"]
    H4 --> H5["返回答案 + 溯源"]
    C1 --> Z["流式输出到对话框"]
    E --> Z
    F --> Z
    G1 --> Z
    H3 --> Z
    H5 --> Z
```

**意图分类**（`classify_intent`）内部流程：

```mermaid
flowchart LR
    Q[用户消息] --> R[关键词规则]
    R -- "命中 人工/工具词" --> O1[直接返回意图]
    R -- "未命中" --> M[小模型分类]
    M -- "输出合法标签" --> O2[返回意图]
    M -- "失败/非法" --> R2[规则兜底]
    R2 --> O3[返回意图 / 默认 business_qa]
```

**知识库入库链路**（上传 → 索引）：

```mermaid
flowchart LR
    UP[上传文档] --> CP[复制到 chroma/knowledge]
    CP --> L["启动预热 / 上传后<br/>后台建索引"]
    L --> LD["MyCustomLoader<br/>加载 + 切分 chunk=500 overlap=200"]
    LD --> IDX["Chroma 向量化 + BM25 混合索引"]
    IDX --> RT["生成 EnsembleRetriever"]
```

## 🧩 系统分层架构

```mermaid
flowchart TB
    subgraph L1["展示层 Presentation"]
        UI["Gradio UI<br/>main.py"]
    end

    subgraph L2["服务层 Service"]
        ROUTER["CombineClient 意图路由"]
        SAFETY["safety_check 安全校验"]
        INTENT["intent_classifier 意图分类"]
        TOOLS["tools 工具调用"]
        KB["MyKnowledge 知识库管理 / 检索"]
        LOADER["MyCustomLoader 文档加载切分"]
    end

    subgraph L3["数据 / 模型层 Data & Model"]
        CHROMA[("Chroma 向量库")]
        SQLITE[("SQLite 记录管理器")]
        FILES[("文档文件 chroma/knowledge")]
        API["DashScope API<br/>LLM / Embedding"]
    end

    UI --> ROUTER
    UI --> KB
    ROUTER --> SAFETY
    ROUTER --> INTENT
    ROUTER --> TOOLS
    ROUTER --> KB
    KB --> LOADER
    KB --> CHROMA
    KB --> SQLITE
    KB --> FILES
    ROUTER --> API
    KB --> API
```

> 箭头方向表示**调用关系**：展示层触发服务层，服务层访问数据 / 模型层；`ROUTER`（意图路由）与 `KB`（知识库）都会调用 DashScope API。

## 📁 模块结构

| 文件 | 职责 |
|---|---|
| `main.py` | Gradio 入口：左侧栏（知识库管理 / 分块查看）+ 对话区布局与事件绑定 |
| `combine_client.py` | `CombineClient`，意图路由核心：`stream` → 闲聊 / 工具 / RAG 三条链 |
| `intent_classifier.py` | 意图分类：`Intent` 枚举 + 关键词规则 + 小模型分类 + 兜底 |
| `safety_check.py` | 安全合规校验：敏感词匹配（`sensitive_words.txt`）+ 大模型二次判断 |
| `knowledge.py` | `MyKnowledge`：上传/列表/删除/分块查看，检索器构建与混合索引 |
| `custom_loader.py` | `MyCustomLoader`：多格式文档加载 + 递归字符切分 |
| `tools.py` | 工具定义：`get_weather`、`get_time` |
| `models.py` | 模型客户端：LLM / Embedding / Rerank |
| `logger.py` | loguru 日志配置 |
| `chat_store.py` | 对话记录持久化：`db/conversations.db`（SQLite），重启后可列出并恢复 |
| `sensitive_words.txt` | 可编辑的敏感词表（改后无需重启） |

## 🔧 技术栈

- **LLM 框架**：LangChain 生态（`langchain-core` / `langchain-classic` / `langchain-community` / `langchain-openai` / `langchain-text-splitters`）
- **模型平台**：阿里云 DashScope（百炼），LLM 使用 `qwen3.8-max` / `deepseek-r1` / `deepseek-v3`，Embedding 使用 `text-embedding-v3`
- **向量数据库**：Chroma（持久化目录 `chroma/<md5>`），配合 BM25 稀疏检索组成混合检索
- **检索增强**：`EnsembleRetriever`（向量 + BM25 混合召回）+ 阿里 `TextReRank` 重排（top_n=3），按重排分阈值决定是否转联网兜底
- **文档处理**：`unstructured`（文件类型检测）+ 各类型 Loader（`Docx2txtLoader` 等）+ `RecursiveCharacterTextSplitter`
- **Web 界面**：Gradio
- **日志**：loguru

## 🚀 快速开始

### 方式一：Docker（推荐，一条命令）

```bash
cp .env.example .env      # 填入 DASHSCOPE_API_KEY
docker compose up -d --build
# 打开 http://127.0.0.1:7860 ；日志：docker compose logs -f
# 向量库 / 会话库 / 日志已通过 volume 挂载到宿主机，重建容器不丢数据
```

容器带 `HEALTHCHECK`（`curl /` 200 才算健康），`restart: unless-stopped` 保证异常退出自动拉起。

### 方式二：标准命令脚本

```bash
# Linux / macOS
./scripts/run.sh

# Windows PowerShell
.\scripts\run.ps1
```

脚本负责：建 `.venv` → 装依赖（含 `setuptools<81`，否则 langchain 的 `pkg_resources` 会缺失）→ 检查 `.env` → 启动 `python main.py`。

### 方式三：手动

```bash
pip install "setuptools<81" -r requirements.txt
copy .env.example .env      # 填入 DASHSCOPE_API_KEY
python main.py
```

启动后浏览器打开 `http://127.0.0.1:7860`，界面分两栏：

- **左侧栏**：`＋ 新对话`；历史会话一行一条，鼠标悬停显示删除键，点击即恢复（含模型多轮记忆）；底部「⚙ 知识库管理」。
- **对话区（客户）**：空对话时展示欢迎面板；提问后自动做安全校验 → 意图分类 → 路由回答；每轮回答支持复制 / 点赞 / 点踩。

## 📊 评测

```bash
python eval/evaluate.py     # 检索指标（Hit@1 / Hit@3 / MRR / 拒答率）+ 端到端指标
python eval/ablation.py     # vector / hybrid / production 三种检索方案对比
```

评测集：`eval/data/retrieval_set.json`（22 题库内 + 4 题库外），文档在 `eval/data/docs`，
库外问题用的留出文档在 `eval/data/holdout_docs`（评测时临时入库、跑完删除）。
结果写入 `eval/reports/`，指标口径见该目录 README。

## 📌 依赖说明

- `.docx` 文档加载依赖 `docx2txt`，若报 `No module named 'docx2txt'`，请执行：
  ```bash
  pip install docx2txt
  ```
- 运行前需设置环境变量 `DASHSCOPE_API_KEY`（可在 `.env` 中配置，由 `models.py` 自动加载）。

## 🧪 可独立运行的自测

```bash
python intent_classifier.py   # 意图分类（规则路径，不调大模型）
python safety_check.py        # 安全校验（敏感词匹配，不调大模型）
python tools.py               # 工具调用
python test_chat_ui.py        # 对话页 messages 格式：上屏 / 流式填充 / 溯源（不调大模型）
python "功能模块测试用例/知识库文档加载.py"   # 文档加载与切分（覆盖 txt/md/pdf/docx）
```
