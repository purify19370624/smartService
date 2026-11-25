# 评测报告

本目录存放 `python eval/evaluate.py` 与 `python eval/ablation.py` 的产出。

| 文件 | 内容 |
|---|---|
| `latest.md` / `latest.json` | 最近一次检索指标 + 端到端问答指标（含逐题明细） |
| `ablation.md` / `ablation.json` | 同一评测集上 vector / hybrid / production 三种检索方案的对比 |

指标口径：

- **Hit@1 / Hit@3**：库内问题里，前 1 / 前 3 个召回片段中至少有一个来自参考答案所在文档的比例。
- **MRR**：第一条来自正确文档的片段排名倒数，取平均。
- **Reject Rate**（检索侧）：库外问题中，最高重排分低于 `RERANK_SCORE_THRESHOLD` 的比例。
- **Grounded Rate / Keyword Hit / No-hallucination Rate**（端到端侧）：是否有知识库依据、回答是否含参考答案关键词、库外问题是否拒答或转联网。
- **First-token latency**：从提问到第一段文字返回的平均秒数。

复现：

```bash
cp .env.example .env    # 填入 DASHSCOPE_API_KEY
python eval/evaluate.py     # 检索 + 端到端（端到端会真实调用模型，约 1-2 分钟/10 题）
python eval/ablation.py     # 消融对比
```

> 评测会向 `chroma/knowledge/` 复制 `eval/data/docs` 与 `eval/data/holdout_docs` 下的文档，
> 跑完自动删除留出文档；若你的知识库目录里已有同名文件，会被覆盖。
