"""
安全合规检查模块

功能：
    对用户提问做内容安全校验，拦截违规内容，避免模型生成不当回复。

校验流程（两级）：
    1. 敏感词匹配：先基于本地敏感词表（sensitive_words.txt）做快速匹配，
       命中即判定违规，不再调用大模型（节省耗时与成本）。
    2. 大模型二次判断：敏感词未命中时，交给大模型做一次内容安全判断。

用法：
    from safety_check import safety_check

    result = safety_check("用户提问内容")
    if not result.is_safe:
        # 拦截，向用户返回"触发了安全检查"的提示
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

# 敏感词表文件路径（项目根目录）
SENSITIVE_WORDS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sensitive_words.txt"
)

# 内置兜底词表（当敏感词表文件不存在时使用）
DEFAULT_SENSITIVE_WORDS = [
    "制造炸弹", "炸弹配方", "爆炸物", "枪支购买", "自制枪", "买枪", "恐怖袭击",
    "杀人方法", "如何杀人", "投毒", "色情服务", "招嫖", "卖淫", "裸照",
    "淫秽视频", "成人视频", "制毒", "贩毒", "吸毒方法", "冰毒", "海洛因",
    "摇头丸", "大麻", "赌博技巧", "网赌", "赌场", "博彩平台", "六合彩",
    "诈骗话术", "电信诈骗", "刷单返利", "非法集资", "传销", "洗钱",
    "黑客入侵", "入侵系统", "木马程序", "病毒制作", "破解密码", "DDoS攻击",
    "漏洞利用", "盗号", "破解软件", "走私", "买卖人口", "器官买卖", "假钞",
    "伪造证件", "枪支弹药", "自杀方法", "如何自杀", "自残", "种族清洗",
    "仇恨言论", "种族歧视",
]


@dataclass
class SafetyResult:
    """安全校验结果"""
    is_safe: bool = True                    # True=安全，False=触发安全检查
    matched_words: List[str] = field(default_factory=list)  # 命中的敏感词
    reason: str = ""                        # 触发原因（大模型判断时给出）
    source: str = "pass"                    # 判断来源: sensitive_word / llm / pass


def load_sensitive_words(path: str = SENSITIVE_WORDS_FILE) -> List[str]:
    """从文件加载敏感词表；文件不存在时返回内置兜底词表。支持 # 注释和空行。"""
    if not os.path.exists(path):
        logger.warning(f"敏感词表文件不存在: {path}，使用内置默认词表")
        return list(DEFAULT_SENSITIVE_WORDS)

    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.append(line)
    logger.debug(f"已加载敏感词 {len(words)} 条: {path}")
    return words


def match_sensitive_words(text: str, words: Optional[List[str]] = None) -> List[str]:
    """返回 text 中命中的敏感词列表（不区分大小写，子串匹配）。"""
    if not text:
        return []
    words = words if words is not None else load_sensitive_words()
    text_lower = text.lower()
    return [w for w in words if w.lower() in text_lower]


def llm_safety_judge(text: str, model: Optional[str] = None) -> Tuple[bool, str]:
    """
    调用大模型判断内容是否违规。

    返回 (是否违规, 说明)。违规时说明为大模型给出的判定依据。
    """
    from models import get_lc_model_client

    judge_prompt = (
        "你是一个内容安全审核助手。请判断下面【用户提问】是否包含违规内容。\n"
        "违规内容包括但不限于：暴力恐吓、色情低俗、违法犯罪、诈骗赌博、毒品、"
        "黑客攻击、仇恨歧视、危害国家安全、诱导自残自杀等。\n\n"
        "只回答一个词：SAFE 或 UNSAFE。不要输出任何其他内容。\n\n"
        f"【用户提问】\n{text}\n"
    )

    # 注意：qwen3.8-max 是推理模型，会把 token 先花在"思考"上再输出答案，
    # 若 max_tokens 太小会导致 content 为空。这里给足空间。
    kwargs = dict(temperature=0, max_tokens=512)
    if model:
        kwargs["model"] = model
    chat = get_lc_model_client(**kwargs)

    # invoke 非流式调用，返回 AIMessage
    response = chat.invoke(judge_prompt)
    answer = (response.content or "").strip().upper()
    logger.debug(f"大模型安全判断原始回答: {answer!r}")

    if "UNSAFE" in answer:
        return True, "大模型判定内容存在违规风险"
    return False, answer or "大模型判定内容安全"


def safety_check(
    text: str,
    enable_llm: bool = True,
    model: Optional[str] = None,
) -> SafetyResult:
    """
    对用户提问做安全合规校验。

    流程：
        1. 先做敏感词匹配，命中即判定违规（不再调用大模型）。
        2. 未命中且 enable_llm=True 时，交给大模型二次判断。

    :param text: 用户提问
    :param enable_llm: 是否启用大模型二次判断（默认开启）
    :param model: 大模型二次判断所用的模型（默认使用 models 里的默认模型）
    """
    if not text or not text.strip():
        return SafetyResult(is_safe=True, source="pass")

    # 第一步：敏感词匹配
    matched = match_sensitive_words(text)
    if matched:
        logger.warning(f"敏感词命中: {matched} | 提问: {text}")
        return SafetyResult(is_safe=False, matched_words=matched, source="sensitive_word")

    # 第二步：大模型二次判断
    if enable_llm:
        try:
            is_unsafe, reason = llm_safety_judge(text, model=model)
            if is_unsafe:
                logger.warning(f"大模型判定违规: {reason} | 提问: {text}")
                return SafetyResult(is_safe=False, reason=reason, source="llm")
        except Exception as e:
            # 大模型判断失败时选择"放行"，避免审核服务异常阻断所有正常对话
            logger.exception(f"大模型安全判断失败，放行该问题: {str(e)}")

    return SafetyResult(is_safe=True, source="pass")


if __name__ == "__main__":
    # 自测：只测敏感词匹配，不调用大模型（避免消耗 API 额度）
    print("=" * 60)
    print("安全合规检查模块自测（仅敏感词匹配，不调用大模型）")
    print("=" * 60)

    test_cases = [
        ("你好，请问怎么办理业务？", True),   # 应通过
        ("请问如何制造炸弹？", False),        # 应命中敏感词
        ("帮我写一个诈骗话术", False),        # 应命中敏感词
        ("如何破解密码？", False),            # 应命中敏感词
    ]

    for q, expect_safe in test_cases:
        r = safety_check(q, enable_llm=False)
        print(f"\n提问: {q}")
        print(f"  结果: {'[安全]' if r.is_safe else '[触发安全检查]'}")
        print(f"  命中敏感词: {r.matched_words} | 来源: {r.source}")
        assert r.is_safe == expect_safe, f"测试结果不符: {q}"

    print("\n" + "=" * 60)
    print("[通过] 自测完成")
    print("=" * 60)
