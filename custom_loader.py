import re
from langchain_community.document_loaders import CSVLoader, TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader, \
    Docx2txtLoader
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.file_utils.filetype import FileType, detect_filetype
from loguru import logger
class QATextSplitter(RecursiveCharacterTextSplitter):
    """问答类文档分割器：优先按"编号问题"切分，保持"问题+答案"完整。

    匹配行首的编号问题（如 "1." "2、" "1）"）或"问："式问题，在这些位置切开；
    若切出的单个问答仍超过 chunk_size，则退回父类的递归字符切分兜底。
    """

    # 【声明的属性】_QA_BOUNDARY：问答边界正则（编译后的 Pattern，类级共享）。
    # 用零宽前瞻匹配"行首编号问题"（如 "1." "2、" "1）"）或"问："式问题的起始位置，
    # 作为 split 的分隔点，从而保证每个切分块是一个完整的"问题+答案"。
    _QA_BOUNDARY = re.compile(r'(?m)^(?=\s*(?:\d+\s*[.、)）:：]|问\s*[：:])\s*\S)')

    def split_text(self, text: str) -> list:
        """
        按问答边界切分文本：优先保证每个块是一个完整问答对，超长时退回递归切分。

        :param text: 待切分的文本（str）。
        :return: 切分后的文本块列表（list[str]）。
        """
        parts = self._QA_BOUNDARY.split(text)
        chunks = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self._chunk_size:
                chunks.append(part)
            else:
                # 单个问答过长时，退回父类的递归切分
                chunks.extend(super().split_text(part))
        return chunks


class MyCustomLoader(BaseLoader):
    """
    文档加载和分割模块
    """
    # 【声明的属性】file_type：文件类型 → (加载器类, 加载参数) 的映射表（工厂模式）。
    # 键为 unstructured 检测出的 FileType，值为 (Loader 类, 初始化参数 dict)，
    # 用于在 __init__ 中根据检测结果动态创建对应的文档加载器。
    # 显式指定 UTF-8：autodetect_encoding 在当前 langchain/charset_normalizer 组合下会抛
    # TypeError（FileEncoding 多了 mime_type），txt/md/csv 全都读不出来。
    # shortcut: 只认 UTF-8，GBK 文本会加载失败，遇到再补降级读取
    file_type = {
        FileType.CSV: (CSVLoader, {'encoding': 'utf-8'}),
        FileType.TXT: (TextLoader, {'encoding': 'utf-8'}),
        FileType.DOC: (UnstructuredWordDocumentLoader, {}),
        FileType.DOCX: (Docx2txtLoader, {}),
        FileType.PDF: (PyPDFLoader, {}),
        # md 也按纯文本读：UnstructuredMarkdownLoader 会吃掉列表序号（"1." 变 ""），
        # 按编号问答切分就失效了，而且慢得多。
        FileType.MD: (TextLoader, {'encoding': 'utf-8'})
    }

    # 初始化方法：检测文件类型、创建对应的加载器与问答感知分割器
    def __init__(self, file_path: str):
        """
        初始化文档加载器。

        :param file_path: 待加载文档的路径（str）。
                          根据文件类型从 file_type 表中选择对应 Loader；
                          文件类型不支持时抛出 ValueError。
        """
        self.file_path = file_path
        detected_type = detect_filetype(file_path)
        logger.info(f"检测到的文件类型: {detected_type}")
        
        if detected_type not in self.file_type:
            raise ValueError(f"不支持的文件类型: {detected_type}")
            
        loader_class, params = self.file_type[detected_type]
        logger.info(f"本文档[{file_path}]需使用文档加载器: {loader_class}")
        self.loader: BaseLoader = loader_class(file_path, **params)# 使用工厂模式创建加载器
        # 问答感知文本分割：优先按"编号问题"切分（每个块=一个完整问答对），
        # 单个问答过长时退回递归字符切分兜底。chunk_size 作为兜底阈值。
        self.text_splitter = QATextSplitter(
            separators=["\n\n", "\n", " ", ""], # 分隔符列表：指定切分的优先级顺序。程序会优先尝试用 "\n\n"（段落）切分，不行则依次尝试 "\n"（行）、" "（空格），最后才用 ""（空字符串/逐字）强制切分
            chunk_size=300,  # 文本块大小：限制每个切分后的片段（Chunk）最大包含的字符数（由 length_function 决定计算方式）
            chunk_overlap=100,  # 重叠字符数：相邻两个文本块之间重复保留的字符数量，用于防止关键信息在切分点丢失，保持上下文连贯
            length_function=len,  # 长度计算函数：用于计算文本长度的函数，这里使用 Python 内置的 len() 函数按字符数计算（也可替换为 tiktoken 等按 Token 计算的函数）
        )

    def lazy_load(self):
        """
        惰性加载并切分文档（返回加载切分后的文档）。

        :return: 加载并切分后的 Document 列表（或生成器）。
        """
        # 文档的切分加载
        return self.loader.load_and_split(self.text_splitter)

    def load(self):
        """
        加载文档（带异常降级）：先尝试「加载+切分」，失败则降级为「仅加载不切分」。

        :return: Document 列表；彻底失败时返回空列表 []（不向上抛异常）。
        """
        # 加载
        try:
            documents = self.lazy_load()
            # 确保返回的是列表而不是生成器
            if not isinstance(documents, list):
                documents = list(documents)
            logger.info(f"成功加载文档，数量: {len(documents)}")
            if len(documents) > 0:
                logger.debug(f"第一个文档片段预览: {documents[0].page_content[:200]}")
            return documents
        except Exception as e:
            logger.exception(f"文档加载失败: {str(e)}")
            # 尝试不使用分割器直接加载
            logger.warning("尝试不使用分割器直接加载文档...")
            try:
                raw_documents = self.loader.load()
                logger.info(f"直接加载成功，原始文档数量: {len(raw_documents)}")
                return raw_documents
            except Exception as e2:
                logger.exception(f"直接加载也失败: {str(e2)}")
                return []
