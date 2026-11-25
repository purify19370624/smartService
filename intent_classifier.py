"""
意图分类模块（电商智能客服）

将用户消息分成四类意图，供路由层分派到不同处理链：
    business_qa —— 业务咨询（走 RAG 问答）
    service     —— 办理类/查询类（走工具调用，如查天气/查时间）
    chat        —— 闲聊（友好简短回应）
    human       —— 人工请求（直接转人工）

分类方案（低成本）：
    1. 强规则快路径：关键词命中「人工请求」「工具查询」直接返回，省一次模型调用。
    2. 小模型分类：用一次小模型调用输出意图标签（主路径）。
    3. 规则兜底：模型调用失败或输出非法时，回落到关键词规则，再回落 business_qa。

用法：
    from intent_classifier import classify_intent, Intent
    intent = classify_intent("我要查一下北京天气")   # -> Intent.SERVICE
"""
from enum import Enum
from typing import Optional

from loguru import logger


class Intent(str, Enum):
    """用户消息意图"""
    BUSINESS_QA = "business_qa"   # 业务咨询 -> RAG
    SERVICE = "service"           # 办理类/查询类 -> 工具调用
    CHAT = "chat"                 # 闲聊 -> 简短回应
    HUMAN = "human"               # 人工请求 -> 转人工


# 关键词规则（命中即认为属于该意图）。人工/工具词为高置信，闲聊为弱提示。
HUMAN_KEYWORDS = ["人工", "转人工", "人工客服", "真人客服", "找人工", "人工服务"]
SERVICE_KEYWORDS = [
    "查天气", "天气", "温度", "气温",
    "查时间", "现在时间", "几点", "日期", "今天几号", "星期几", "现在几点",
]
CHAT_KEYWORDS = ["你好", "您好", "嗨", "在吗", "谢谢", "再见", "拜拜", "早上好", "晚上好", "你是谁"]

# 记忆类提问：询问"我之前问了/说了什么"，应走闲聊（有历史上下文），而非业务咨询 RAG
MEMORY_KEYWORDS = [
    "前面问", "刚才问", "之前问", "刚刚问",
    "前面说", "刚才说", "之前说", "刚刚说",
    "我问过", "我说过", "我们聊了", "我们聊过", "还记得",
]

# 合法标签集合（用于解析小模型输出）
_INTENT_LABELS = {Intent.BUSINESS_QA.value, Intent.SERVICE.value, Intent.CHAT.value, Intent.HUMAN.value}


def keyword_classify(text: str) -> Optional[Intent]:
    """关键词规则分类。命中返回对应意图，否则返回 None。"""
    if not text:
        return None
    for kw in HUMAN_KEYWORDS:
        if kw in text:
            return Intent.HUMAN
    for kw in SERVICE_KEYWORDS:
        if kw in text:
            return Intent.SERVICE
    for kw in CHAT_KEYWORDS:
        if kw in text:
            return Intent.CHAT
    return None


def _llm_classify(text: str, model: Optional[str] = None) -> Optional[Intent]:
    """用小模型分类。返回解析出的意图，失败/非法返回 None。"""
    from models import get_lc_model_client

    prompt = (
        "你是电商智能客服的意图分类器。请判断下面【用户消息】的意图，"
        "只能输出以下四个标签之一（不要输出任何其他内容）：\n"
        "business_qa —— 业务咨询（咨询商品、订单、售后、物流、退货、价格、政策等业务问题）\n"
        "service —— 办理类/查询类（查天气、查时间等工具查询）\n"
        "chat —— 闲聊（问候、寒暄、无关话题）\n"
        "human —— 人工请求（明确要求转人工客服）\n\n"
        f"【用户消息】\n{text}\n\n标签："
    )

    # 注意：推理模型会先把 token 花在"思考"上，max_tokens 太小会导致 content 为空，
    # 分类标签解析不到。这里给足空间。
    kwargs = dict(temperature=0, max_tokens=512)
    if model:
        kwargs["model"] = model
    chat = get_lc_model_client(**kwargs)

    resp = chat.invoke(prompt)
    answer = (resp.content or "").strip().lower()
    logger.debug(f"小模型意图分类原始输出: {answer!r}")
    for label in _INTENT_LABELS:
        if label in answer:
            return Intent(label)
    return None


def classify_intent(
    text: str,
    model: Optional[str] = None,
    enable_llm: bool = True,
) -> Intent:
    """
    对用户消息做意图分类。

    流程：
        1. 强规则快路径：人工请求 / 明确工具词直接返回。
        2. 小模型分类（enable_llm=True 时）。
        3. 规则兜底，最后回落 business_qa。

    :param text: 用户消息
    :param model: 小模型分类所用模型（默认复用 models 默认模型）
    :param enable_llm: 是否启用小模型分类（自测时可关掉，只走规则）
    """
    if not text or not text.strip():
        return Intent.CHAT

    # 第一步：关键词规则
    rule = keyword_classify(text)
    # 强规则快路径：人工请求、明确工具查询，高置信直接返回
    if rule in (Intent.HUMAN, Intent.SERVICE):
        logger.info(f"意图分类(规则快路径): {rule.value}")
        return rule

    # 记忆类提问强规则快路径 → 闲聊（需要历史上下文，不能走 RAG）
    if any(kw in text for kw in MEMORY_KEYWORDS):
        logger.info(f"意图分类(记忆快路径): {Intent.CHAT.value}")
        return Intent.CHAT

    # 第二步：小模型分类
    if enable_llm:
        try:
            intent = _llm_classify(text, model=model)
            if intent is not None:
                logger.info(f"意图分类(小模型): {intent.value}")
                return intent
        except Exception as e:
            logger.warning(f"小模型意图分类失败，回退规则: {str(e)}")

    # 第三步：规则兜底，再回落业务咨询
    final = rule or Intent.BUSINESS_QA
    logger.info(f"意图分类(规则兜底): {final.value}")
    return final


if __name__ == "__main__":
    # 自测：只验证关键词规则路径，不调用大模型（避免消耗 API 额度）
    print("=" * 60)
    print("意图分类自测（仅关键词规则，不调用大模型）")
    print("=" * 60)

    test_cases = [
        ("你好", Intent.CHAT),
        ("查一下北京的天气", Intent.SERVICE),
        ("现在几点了", Intent.SERVICE),
        ("我要人工客服", Intent.HUMAN),
        ("退货流程是什么", Intent.BUSINESS_QA),   # 无规则命中 -> 兜底业务咨询
        ("我前面问了什么？", Intent.CHAT),        # 记忆类提问 -> 闲聊
    ]

    for q, expected in test_cases:
        got = classify_intent(q, enable_llm=False)
        print(f"\n消息: {q}")
        print(f"  意图: {got.value}（期望 {expected.value}）")
        assert got == expected, f"测试结果不符: {q} -> {got}"

    print("\n" + "=" * 60)
    print("[通过] 意图分类规则自测完成")
    print("=" * 60)
