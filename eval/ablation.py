"""
消融对比实验：同一评测集上比较「优化前 / 优化中 / 优化后」的检索指标

三种检索器（都用知识库里同一份评测文档）：
    vector    基线：纯向量相似度检索（无关键词召回、无重排）
    hybrid    向量 + BM25 混合召回（无重排）
    production 混合召回 + 重排 top_n=3，按重排分取前 3（线上实际链路）

运行：python eval/ablation.py         # 会调用 embedding / rerank 接口
产物：eval/reports/ablation.json / ablation.md
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 没有可用 Key 时给出明确提示，而不是跑一半挂在 401 上
if not os.getenv("DASHSCOPE_API_KEY"):
    sys.exit("[跳过] 未设置 DASHSCOPE_API_KEY：消融实验需要真实 embedding / rerank 接口，"
             "请先在 .env 中配置后运行 python eval/ablation.py")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

DATA = EVAL_DIR / "data" / "retrieval_set.json"
DOCS = EVAL_DIR / "data" / "docs"
REPORTS = EVAL_DIR / "reports"


def rank_and_metrics(sources, expected):
    """sources 为按排名给出的来源文件列表；返回 (命中排名, hit@1, hit@3, mrr)。"""
    rank = next((i + 1 for i, s in enumerate(sources) if s == expected), 0)
    return rank, int(rank == 1), int(0 < rank <= 3), (1 / rank if rank else 0.0)


def evaluate(name, retrieve, cases):
    rows, h1, h3, mrr = [], 0, 0, 0.0
    for c in cases:
        sources = retrieve(c["question"])
        rank, a, b, m = rank_and_metrics(sources, c["source"])
        h1, h3, mrr = h1 + a, h3 + b, mrr + m
        rows.append({"id": c["id"], "question": c["question"], "expected": c["source"],
                     "found": sources[:3], "rank": rank})
    n = len(cases)
    return {"name": name, "n": n, "hit@1": round(h1 / n, 4), "hit@3": round(h3 / n, 4),
            "mrr": round(mrr / n, 4), "rows": rows}


def main():
    import knowledge
    from knowledge import MyKnowledge

    cases = [c for c in json.loads(DATA.read_text(encoding="utf-8-sig")) if c["source"]]
    for f in DOCS.iterdir():
        shutil.copy(f, Path(knowledge.KNOWLEDGE_DIR) / f.name)
    kb = MyKnowledge()
    kb.load_knowledge()

    from langchain_community.vectorstores import Chroma
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever
    from custom_loader import MyCustomLoader

    # 集合名是文件名的 md5（knowledge.load_knowledge 的命名规则），不是文件名
    files = {knowledge.get_md5(f.name): f.name for f in DOCS.iterdir() if f.is_file()}
    emb = MyKnowledge._MyKnowledge__embeddings
    stores = {g: Chroma(collection_name=g, embedding_function=emb,
                        persist_directory=os.path.join('./chroma', g)) for g in files}
    chunks = {g: MyCustomLoader(os.path.join(knowledge.KNOWLEDGE_DIR, name)).load()
              for g, name in files.items()}  # 供 BM25 使用，与入库时同一份切分

    def sources_of(docs, n=3):
        return [os.path.basename(d.metadata.get("source", "")) for d in docs[:n]]

    def rrf(ranked_lists, k=60, n=3):
        """倒数排名融合：各召回列表按位置加权合并，再取前 n（跨集合比较分数的通用做法）。"""
        score = {}
        for docs in ranked_lists:
            for pos, d in enumerate(docs):
                key = d.page_content[:200]
                score.setdefault(key, [0.0, d])
                score[key][0] += 1.0 / (k + pos + 1)
        return [d for _, d in sorted(score.values(), key=lambda x: -x[0])][:n]

    def vector_only(q):
        # 每库取 20 条候选，按余弦相似度合并后取前 3（只有这一种方案有可比分数）
        cand = []
        for g in files:
            cand.extend(stores[g].similarity_search_with_relevance_scores(q, k=knowledge.RECALL_K))
        cand.sort(key=lambda x: -x[1])
        return sources_of([d for d, _ in cand])

    def hybrid_only(q):
        lists = []
        for g in files:
            ens = EnsembleRetriever(
                retrievers=[stores[g].as_retriever(search_kwargs={"k": knowledge.RECALL_K}),
                            BM25Retriever.from_documents(chunks[g], k=knowledge.RECALL_K)])
            lists.append(ens.invoke(q))
        return sources_of(rrf(lists))

    def production(q):
        out = []
        for name in files.values():   # get_retrievers 接收文件名，内部再算 md5
            r = kb.get_retrievers(name)
            if r is not None:
                out.extend(r.invoke(q))
        out.sort(key=lambda d: d.metadata.get("relevance_score", 0.0), reverse=True)
        return sources_of(out)

    variants = [evaluate("vector", vector_only, cases),
                evaluate("hybrid", hybrid_only, cases),
                evaluate("production", production, cases)]

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "ablation.json").write_text(json.dumps(variants, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 检索消融对比（同一评测集，库内问题）", "",
             "| 方案 | 样本数 | Hit@1 | Hit@3 | MRR |", "|---|---|---|---|---|"]
    lines += [f"| {v['name']} | {v['n']} | {v['hit@1']} | {v['hit@3']} | {v['mrr']} |" for v in variants]
    lines += ["", "## 逐题命中排名（数字越小越好，- 表示未命中所属文档）", "",
              "| 问题 | 期望来源 | vector | hybrid | production |", "|---|---|---|---|---|"]
    for i, c in enumerate(variants[0]["rows"]):
        lines.append(f"| {c['question']} | {c['expected']} | {c['rank'] or '-'} | "
                     f"{variants[1]['rows'][i]['rank'] or '-'} | {variants[2]['rows'][i]['rank'] or '-'} |")
    (REPORTS / "ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"报告已写入: {REPORTS / 'ablation.md'}")


if __name__ == "__main__":
    main()
