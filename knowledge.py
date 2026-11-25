import hashlib
import os
import shutil
import threading
from typing import Optional

from langchain_classic.indexes import SQLRecordManager
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.document_loaders import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index

from custom_loader import MyCustomLoader
from models import get_ali_embeddings, get_ali_rerank
from loguru import logger

# 【声明的属性】KNOWLEDGE_DIR：知识库原始文档存放目录（模块级配置常量）。
# 上传的文档会被复制到该目录；load_knowledge 遍历该目录为每个文件建立向量索引。
KNOWLEDGE_DIR = './chroma/knowledge/'

# 【声明的属性】RECALL_K：混合召回候选数（模块级配置常量）。
# 重排前先召回足够多的候选（向量检索与 BM25 各召回 K 条），
# 让重排阶段有机会从更多候选中挑出最相关的块。
RECALL_K = 10

# 【声明的属性】embedding_model：全局嵌入模型实例（模块级单例）。
# 在导入时即初始化，供 MyKnowledge 向量化文档使用，避免重复创建连接。
embedding_model = get_ali_embeddings()

# 【声明的属性】_build_lock：建索引的串行锁。
# 各集合的 SQLRecordManager 记录都写同一个 db/record_manager_cache.db，
# 后台预热线程和上传后的建索引线程若同时写会互相锁库，这里串行化。
_build_lock = threading.Lock()

class MyKnowledge:
    """
    知识库管理模块
    """
    # 【声明的属性】__embeddings：向量化模型实例（类级共享），用于把文本转成向量。
    __embeddings = embedding_model
    logger.info(f"当前嵌入模型: {__embeddings.model}")

    # 【声明的属性】__retrievers：检索器缓存表（类级字典）。
    # 键为知识库名称的 md5，值为构建好的混合检索器；避免每次提问重复构建索引。
    __retrievers = {}

    def load_knowledge(self, build=True):
        """
        加载知识库目录下的所有文档，并为每个新文档建立检索索引。

        流程：遍历 KNOWLEDGE_DIR → 每个文件取 md5 作为集合名 → 若缓存中无该集合，
        则用 MyCustomLoader 加载文档并 create_indexes 建立混合检索索引。

        建索引要打很多次 embedding（一个 7MB 的 PDF 上千个块，几十秒），
        所以提问链路用 build=False 只取已建好的，建索引交给启动预热 / 上传后的后台线程。

        :param build: True=缺索引就现建（后台线程用）；False=只用已建好的，不阻塞提问。
        :return: 知识库文件名列表（list[str]）；无可用（build=True 且目录为空）时返回 ["无可用知识库"]。
        """
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

        collections = []
        for file in os.listdir(KNOWLEDGE_DIR):
            # 得到知识库的路径（跳过子目录，避免把目录当文档去加载）
            file_path = os.path.join(KNOWLEDGE_DIR, file)
            if not os.path.isfile(file_path):
                continue

            # 简单化处理知识库名称，由文件名md5编码得到
            collection_name = get_md5(file)

            if collection_name not in self.__retrievers:
                if not build:
                    continue  # 还没建好：本次提问先不用它
                try:
                    # SQLRecordManager 共用一个 sqlite 文件，建索引串行做，避免并发写锁
                    with _build_lock:
                        if collection_name in self.__retrievers:
                            continue
                        logger.info(f"开始建索引: {file}")
                        loader = MyCustomLoader(file_path)
                        self.__retrievers[collection_name] = create_indexes(
                            collection_name, loader, self.__embeddings
                        )
                        logger.info(f"建索引完成: {file}")
                except Exception as e:
                    # 单个文档建索引失败（不支持的格式、损坏文件等）不能拖垮整个知识库
                    logger.exception(f"知识库 [{file}] 建索引失败，已跳过: {str(e)}")
                    continue

            collections.append(file)
        if not collections and build:
            collections = ["无可用知识库"]
        logger.debug(f"可用知识库: {collections}")
        return collections

    # 获取知识库检索器
    def get_retrievers(self, collection):
        """
        获取指定知识库的最终检索器（混合召回 + 重排）。

        底层是 EnsembleRetriever（稠密向量 + BM25），外层用 ContextualCompressionRetriever
        包装阿里重排器（top_n=3），对召回结果精排。

        :param collection: 知识库名称（str，即原始文件名）。
        :return: 构建好的检索器；集合名不存在于缓存时返回 None。
        """
        collection_name = get_md5(collection)
        if collection_name not in self.__retrievers:
            logger.debug(f"知识库尚未建好索引: {collection}")
            return None

        retriever = self.__retrievers[collection_name] # 获取检索器
        # 创建压缩检索器
        # compression_retriever = ContextualCompressionRetriever(
        #     base_compressor=LLMChainFilter.from_llm(self.__llm), #对检索结果做二次过滤/压缩 后执行
        #     base_retriever=RePhraseQueryRetriever.from_llm(retriever, self.__llm) #底层检索器，负责真正去查文档 先执行
        # )
        # final_retriever = compression_retriever

        # 1) 底层检索：混合召回（稠密向量 + BM25）。
        #    不再套 RePhraseQueryRetriever 做查询改写：它每次提问都要多跑一次大模型，
        #    而且每个知识库各跑一次，首字延迟全耗在这；相关性交给下面的重排兜住。
        base_retriever = retriever

        # 2) 重排：对召回候选集按相关性打分，取 top_n，并把 relevance_score 写入 metadata。
        #    不再使用 LLMChainFilter 做二次过滤——它依赖 LLM 输出 YES/NO，极不稳定，
        #    会把相关文档也过滤掉，导致"未检索到相关文档"。
        #    相关性判断改由重排分 + 阈值完成（见 combine_client._rag_stream）。
        rerank_compressor = get_ali_rerank(top_n=3)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=rerank_compressor,
            base_retriever=base_retriever,
        )

        logger.debug(f"最终检索器为: {final_retriever}")
        return final_retriever

    def list_documents(self):
        """
        轻量列出知识库目录下的文档文件名（只读目录，不建立索引）。

        :return: 按文件名排序的列表（list[str]）；目录不存在时返回空列表。
        """
        if not os.path.exists(KNOWLEDGE_DIR):
            return []
        files = [
            f for f in os.listdir(KNOWLEDGE_DIR)
            if os.path.isfile(os.path.join(KNOWLEDGE_DIR, f))
        ]
        return sorted(files)

    def describe_documents(self):
        """
        知识库文档概览：文件名、大小、是否已建索引（chroma/<md5> 目录是否存在）。

        :return: list[dict]，每项含 name/size/indexed；目录不存在时返回空列表。
        """
        docs = []
        for name in self.list_documents():
            size = os.path.getsize(os.path.join(KNOWLEDGE_DIR, name))
            docs.append({
                "name": name,
                "size": f"{size / 1024:.0f} KB" if size < 1024 * 1024 else f"{size / 1048576:.1f} MB",
                "indexed": os.path.isdir(os.path.join('./chroma', get_md5(name))),
            })
        return docs

    def upload_document(self, temp_file):
        """
        上传文档到知识库（仅保存文件，不立即建索引）。
        索引会在客户提问时由 _resolve_collections -> load_knowledge() 惰性建立。

        :param temp_file: 上传的临时文件路径（str）；未选择时为 None。
        :return: 元组 (状态消息, 文档列表)；状态消息如「已上传: xxx」「文件已存在，跳过」。
        """
        if temp_file is None:
            return "未选择文件", self.list_documents()
        file_name = os.path.basename(temp_file)
        file_path = os.path.join(KNOWLEDGE_DIR, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            shutil.copy(temp_file, file_path)
            logger.info(f"已上传文档: {file_name}")
            return f"已上传: {file_name}", self.list_documents()
        return f"文件已存在，跳过: {file_name}", self.list_documents()

    def delete_document(self, file_name):
        """
        删除指定文档，并清理其向量索引目录（chroma/<md5>）与检索器缓存。

        :param file_name: 要删除的文档名称（str）。
        :return: 元组 (状态消息, 文档列表)。
        """
        if not file_name:
            return "未选择要删除的文档", self.list_documents()

        file_path = os.path.join(KNOWLEDGE_DIR, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已删除文档文件: {file_path}")

        collection_name = get_md5(file_name)
        chroma_dir = os.path.join('./chroma', collection_name)
        if os.path.isdir(chroma_dir):
            shutil.rmtree(chroma_dir, ignore_errors=True)
            logger.info(f"已删除向量索引目录: {chroma_dir}")

        self.__retrievers.pop(collection_name, None)
        return f"已删除: {file_name}", self.list_documents()

    def get_chunks(self, file_name):
        """
        查看某个文档的分块结果。

        :param file_name: 文档名称（str）。
        :return: list[dict]，每项含「序号/内容/元数据」三个键；
                 未选择文档、文档不存在或加载失败（如缺少 docx2txt）时返回空列表。
        """
        if not file_name:
            return []
        file_path = os.path.join(KNOWLEDGE_DIR, file_name)
        if not os.path.exists(file_path):
            logger.warning(f"文档不存在: {file_path}")
            return []
        try:
            documents = MyCustomLoader(file_path).load()
        except Exception as e:
            logger.exception(f"分块加载失败: {str(e)}")
            return []

        chunks = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "") if doc.metadata else ""
            page = doc.metadata.get("page")
            meta = os.path.basename(source) if source else "未知文档"
            if page is not None:
                meta += f"（第 {page} 页）"
            chunks.append({"序号": i, "内容": doc.page_content, "元数据": meta})
        return chunks

# 创建索引
def create_indexes(collection_name: str, loader: BaseLoader, embedding_function: Optional[Embeddings] = None):
    """
    为一个知识库文档建立索引，并返回混合检索器（向量 + BM25）。

    流程：初始化 Chroma 向量库 → 初始化 SQL 记录管理器（增量索引/去重）→
    加载切分文档 → index 入库 → 构建 EnsembleRetriever（稠密向量 + 稀疏 BM25）。

    :param collection_name: 集合名（str，通常为文件名 md5），用于隔离不同知识库的向量集合。
    :param loader: 文档加载器（BaseLoader），负责加载并切分文档。
    :param embedding_function: 向量化函数（Embeddings）；None 时 Chroma 使用默认嵌入。
    :return: 混合检索器（EnsembleRetriever；文档为空时退回单向量检索器）。
    """
    # 初始化Chroma数据库
    db = Chroma(
        collection_name=collection_name,  # 集合名称，用于标识和隔离不同的向量数据集合
        embedding_function=embedding_function,  # 嵌入函数，负责将文本转换为向量表示
        persist_directory=os.path.join('./chroma', collection_name)  # 持久化存储路径，向量数据将保存到该目录下的本地文件中
    )

    # 初始化记录管理器:管理与文档相关的元数据和检索信息，并将这些数据存储在一个SQL数据库中
    record_manager = SQLRecordManager(
        f"chromadb/{collection_name}", db_url="sqlite:///db/record_manager_cache.db"
    )
    """在文档被索引到Chroma数据库之前，SQLRecordManager会管理这些文档的索引记录。它确保文档的索引状态被正确记录，避免重复索引或遗漏。"""
    logger.info(f"record_manager: {record_manager}")
    # 初始化数据库表结构
    record_manager.create_schema()
    """
    在你开始加载和索引文档之前，调用create_schema() 方法，确保用于存储文档元数据和索引信息的数据库表已经存在。
    如果是第一次运行这个脚本，create_schema()方法会自动创建这些表。对于后续运行，如果表已经存在，则该方法不会重复创建，而是直接通过。
    """

    # 记录管理器可能还留着该集合的历史记录，而向量库目录已被删掉（删文档后重新上传同一文件）。
    # 这时 index() 会把所有块判为"已索引"而整批跳过，向量库实际是空的，检索退化成只有 BM25。
    # 只在"向量库为空"时清记录，其余情况保留记录走增量索引——否则每次重启都要重新向量化整个库。
    if not db.get(limit=1)["ids"]:
        stale_keys = record_manager.list_keys()  # 记录管理器按 namespace 隔离，这里即本集合的全部记录
        if stale_keys:
            logger.warning(f"向量库为空但存在 {len(stale_keys)} 条历史索引记录，清理后重新索引: {collection_name}")
            record_manager.delete_keys(stale_keys)

    logger.info("准备进行文件的加载切分....")
    documents = loader.load()
    logger.info(f"文档切分数量: {len(documents)}....")

    # 将加载的文档索引到数据库中
    r = index(documents, record_manager, db,
              cleanup="full", # 索引结束后，删除向量库里不再属于当前来源的旧文档
              source_id_key="source") # 用文档 metadata["source"] 判断哪些文档属于同一个来源
    logger.info(f"文档索引结果为: {r}")
    """
    num_added: 0 表示没有新文档被添加到数据库中。
    num_updated: 0 表示没有文档被更新。
    num_skipped: 8 表示有8个文档被跳过，没有被索引到数据库中。 提高索引效率
    num_deleted: 0 表示没有文档被删除。
    """
    '''混合检索，将稀疏检索器（如BM25）与密集检索器（如嵌入相似性）相结合。
    稀疏检索器擅长根据关键字查找相关文档，而密集检索器擅长根据语义相似性查找相关文档。'''
    
    # 检查文档是否为空，避免 BM25Retriever 报错
    if not documents or len(documents) == 0:
        logger.warning(f"知识库 [{collection_name}] 文档为空，仅使用向量检索")
        ensemble_retriever = db.as_retriever(search_kwargs={"k": RECALL_K}) # 创建一个向量检索器
    else:
        ensemble_retriever = EnsembleRetriever( # 创建一个混合检索器
            # 召回足够多的候选，交给重排做精排
            retrievers=[
                db.as_retriever(search_kwargs={"k": RECALL_K}),# 向量检索器：基于语义相似度召回，擅长捕捉同义词和语义相近的内容
                BM25Retriever.from_documents(documents, k=RECALL_K),# 关键词检索器：基于 BM25 算法进行精确关键词匹配，擅长命中专有名词和特定术语
            ]
        )
    
    # logger.info(f"ensemble_retriever: {ensemble_retriever}")
    return ensemble_retriever

def get_md5(input_string):
    """
    计算字符串的 md5 十六进制哈希值（用作知识库集合名，避免文件名中的特殊字符干扰）。

    :param input_string: 输入字符串（str）。
    :return: 该字符串的 32 位小写 md5 十六进制哈希（str）。
    """
    # 创建一个 md5 哈希对象
    hash_md5 = hashlib.md5()
    # 需要确保输入字符串是字节串，因此如果它是字符串，则需要编码为字节串
    hash_md5.update(input_string.encode('utf-8'))
    # 获取十六进制的哈希值
    return hash_md5.hexdigest()
