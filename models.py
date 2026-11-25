import os
from pathlib import Path
from typing import Optional, Sequence

import dashscope
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_openai import ChatOpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings

# 尝试从 .env 文件加载环境变量（如果存在）
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[INFO] 已从 {env_path} 加载环境变量")
    else:
        print(f"[INFO] 未找到 .env 文件，使用系统环境变量")
except ImportError:
    print("[INFO] 未安装 python-dotenv，仅使用系统环境变量")
    print("       如需使用 .env 文件，请运行: pip install python-dotenv")

ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
ALI_QWEN_MAX_MODEL = "qwen3.8-max"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"
# ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v3"
ALI_TONGYI_EMBEDDING_MODEL = "qwen3.7-text-embedding"
ALI_TONGYI_RERANK_MODEL = "qwen3.7-text-rerank"

def get_lc_model_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL
                        , model=ALI_QWEN_MAX_MODEL, temperature=0.7, max_tokens=8000, search=False, verbose=False, debug=False):
    """
        通过LangChain获得指定平台和模型的客户端，设定的默认平台和模型为阿里百炼qwen-max-latest
        也可以通过传入api_key，base_url，model三个参数来覆盖默认值
        search=True 时打开百炼内置联网搜索（回答带实时信息，不需要额外依赖或搜索 API Key）
        verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    """
    function_name = inspect.currentframe().f_code.co_name
    
    # 首先检查 API Key 是否存在
    if not api_key:
        error_msg = (
            f"\n{'='*60}\n"
            f"❌ 错误：环境变量 {ALI_TONGYI_API_KEY_OS_VAR_NAME} 未设置！\n\n"
            f"请按照以下步骤修复：\n"
            f"1. 检查 .env 文件是否存在于项目根目录\n"
            f"2. 确认 .env 文件中包含正确的 API Key\n"
            f"   格式：DASHSCOPE_API_KEY=sk-xxxxx（不要重复 sk- 前缀）\n"
            f"3. 如果使用系统环境变量，请重启 PyCharm\n"
            f"4. 当前工作目录: {os.getcwd()}\n"
            f"{'='*60}\n"
        )
        print(error_msg)
        raise ValueError(error_msg)
    
    if (verbose):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature}")
    if (debug):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature}")
        print(f"{function_name}-API Key 长度: {len(api_key)}, 前8位: {api_key[:8]}...")
        # 检查 API Key 格式
        if api_key.startswith('sk-sk-'):
            print(f"⚠️  警告：API Key 可能重复了 'sk-' 前缀！")
            print(f"   当前值: {api_key[:15]}...")
            print(f"   建议修改为: {api_key[3:]}")
    
    # qwen3 系列默认"思考"，思考 token 也算进 max_tokens：意图分类（16）、安全判断（32）
    # 这种小预算会被思考吃光、返回空内容，路由和安全判断因此静默失效。
    # 这里统一关掉思考，回答更稳、更快，也省额度。换纯推理模型（如 deepseek-r1）时去掉。
    extra_body = {"enable_thinking": False}
    if search:
        extra_body["enable_search"] = True  # 百炼内置联网搜索
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=temperature,
                      max_tokens=max_tokens, extra_body=extra_body)

def get_ali_embeddings():
    """通过LangChain获得一个阿里通义千问嵌入模型的实例"""
    api_key = os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    if not api_key:
        raise ValueError(f"环境变量 {ALI_TONGYI_API_KEY_OS_VAR_NAME} 未设置！请检查环境变量配置")

    # 修复批次上限：langchain_community 的 DashScopeEmbeddings 里 BATCH_SIZE 字典
    # 未收录新模型时会默认按 25 条一批发送，而 qwen3.7-text-embedding 单批上限是 20，
    # 导致 400 报错「batch size ... should not be larger than 20」。这里按上限 20 登记，
    # 建索引的 embedding 往返次数比一批 10 条少一半（大文档建索引快一倍）。
    from langchain_community.embeddings.dashscope import BATCH_SIZE
    BATCH_SIZE.setdefault(ALI_TONGYI_EMBEDDING_MODEL, 20)

    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING_MODEL, dashscope_api_key=api_key
    )

class AliyunRerank(BaseDocumentCompressor):
    """基于阿里百炼重排序(Rerank) API 的文档重排压缩器。

    不使用 langchain_community 的 DashScopeRerank：其 validate_environment 会
    无条件把 model 覆盖成已下线的 gte-rerank，导致接口返回 output=None，
    从而报 'NoneType' object has no attribute 'results'。
    """
    model: str = "qwen3.7-text-rerank"
    top_n: int = 3
    dashscope_api_key: Optional[str] = None
    # 重排策略指令（qwen3 系列支持），引导模型按"问答检索"打分，可设为 None 关闭
    instruct: Optional[str] = "Given a customer service question, retrieve the passage that directly answers it."

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        contents = [doc.page_content for doc in documents]
        kwargs = {"instruct": self.instruct} if self.instruct else {}
        resp = dashscope.TextReRank.call(
            model=self.model,
            query=query,
            documents=contents,
            top_n=self.top_n,
            return_documents=False,
            api_key=self.dashscope_api_key,
            **kwargs,
        )

        # 校验响应，避免 output 为 None 时抛出晦涩的 NoneType 错误
        if getattr(resp, "status_code", None) != 200:
            raise ValueError(
                f"重排序接口调用失败: code={getattr(resp, 'code', '')} "
                f"message={getattr(resp, 'message', '')}"
            )
        output = getattr(resp, "output", None)
        if output is None or not getattr(output, "results", None):
            raise ValueError(
                f"重排序接口返回空结果: {getattr(resp, 'message', '') or 'output 为空'}"
            )

        ranked = []
        for r in output.results:
            doc = documents[r.index]
            new_doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            new_doc.metadata["relevance_score"] = r.relevance_score
            ranked.append(new_doc)
        return ranked


def get_ali_rerank(top_n=10):
    '''获得阿里百炼重排序模型实例（自定义封装，规避 DashScopeRerank 的模型覆盖 bug）。'''
    api_key = os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    if not api_key:
        raise ValueError(f"环境变量 {ALI_TONGYI_API_KEY_OS_VAR_NAME} 未设置！请检查环境变量配置")
    return AliyunRerank(model=ALI_TONGYI_RERANK_MODEL, top_n=top_n, dashscope_api_key=api_key)


