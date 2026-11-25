"""
工具模块（电商智能客服）

为「办理类/查询类」意图提供简单的工具调用能力。当前提供两个演示工具：
    get_weather —— 查询城市天气（模拟数据，不真正查询）
    get_time    —— 查询当前日期时间

后续如需接入真实业务数据（订单、物流等），只需在此新增 @tool 工具，
并把 TOOLS 列表加入新工具即可，路由层无需改动。
"""
from datetime import datetime

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。参数 city 为城市名，例如「北京」。"""
    # 模拟返回，不真正查询天气数据
    return f"{city} 今天晴，气温 18~26℃，空气质量良，适合出行。"


@tool
def get_time() -> str:
    """查询当前日期和时间。"""
    now = datetime.now()
    weekday = "一二三四五六日"[now.weekday()]
    return f"现在是 {now:%Y年%m月%d日 %H:%M:%S} 星期{weekday}"


# 可供路由层绑定的工具列表
TOOLS = [get_weather, get_time]


if __name__ == "__main__":
    # 自测：直接调用两个工具，验证可用
    print("=" * 60)
    print("工具自测")
    print("=" * 60)
    print("get_weather('杭州') ->", get_weather.invoke({"city": "杭州"}))
    print("get_time()         ->", get_time.invoke({}))
    print("=" * 60)
    print("[通过] 工具自测完成")
