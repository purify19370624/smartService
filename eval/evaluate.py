"""
评测模块：检索命中率与端到端问答指标

运行（项目根目录）：
    python eval/evaluate.py               # 全部：检索 + 端到端问答（需要 DASHSCOPE_API_KEY）
    python eval/evaluate.py --only retr   # 只跑检索

产物：
    eval/reports/latest.json   机器可读结果（脚本据此生成对比表）
    eval/reports/latest.md     人读报告

设计要点：
    - 「知识外问题」测试时把 eval/data/holdout_docs 上传进知识库，跑完再删除，
      因此单次运行即可同时度量「库里有的能命中」和「库里没有的不硬答」。
    - 端到端走的是线上真实链路 CombineClient.stream()，与 Gradio 页面一致。
    - 指标定义写在各 _method 的注释里，便于答辩解释。
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # knowledge.py 用的是相对路径 ./chroma

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

DATA = EVAL_DIR / "data" / "retrieval_set.json"
DOCS = EVAL_DIR / "data" / "docs"
HOLDOUT_DOCS = EVAL_DIR / "data" / "holdout_docs"
REPORTS = EVAL_DIR / "reports"

# 知识外问题的三种「正确」反应
REFUSAL_MARKERS = ("没有找到", "无法回答", "抱歉", "没有相关", "未找到")
WEB_PREFIX = "（知识库没有找到依据"


def load_cases():
    return json.loads(DATA.read_text(encoding="utf-8"))


def install_docs(kb):
    """把评测文档复制进知识库目录（不覆盖用户已有同名文件由调用方负责）。"""
    import shutil
    for src_dir in (DOCS, HOLDOUT_DOCS):
        for f in sorted(src_dir.iterdir()):
            if f.is_file():
                shutil.copy(f, Path(kb_dir()) / f.name)


def kb_dir():
    import knowledge
    return knowledge.KNOWLEDGE_DIR


def cleanup_holdout(kb):
    for f in HOLDOUT_DOCS.iterdir():
        kb.delete_document(f.name)


# ---------------- 检索指标 ----------------

def eval_retrieval(kb, cases):
    """
    指标（n = 库内问题数）：
        Hit@1 / Hit@3 : 命中的前 1 / 前 3 个片段中，至少有一个来自参考答案所在文档的比例
        MRR           : 第一条来自正确文档的片段的倒数排名均值
        Reject        : 库外问题中，最高重排分低于阈值的比例（越高说明越少硬答）
    """
    import combine_client as cc

    in_kb = [c for c in cases if c["source"]]
    out_kb = [c for c in cases if not c["source"]]
    collections = [f.name for f in DOCS.iterdir() if f.is_file()]
    retrievers = {g: kb.get_retrievers(g) for g in collections}
    rows, hit1, hit3, mrr = [], 0, 0, 0.0
    for c in in_kb:
        t0 = time.perf_counter()
        docs = []
        for g in collections:
            r = retrievers.get(g)
            if r is not None:
                docs.extend(r.invoke(c["question"]))
        docs.sort(key=lambda d: d.metadata.get("relevance_score", 0.0), reverse=True)
        docs = docs[:3]
        latency = time.perf_counter() - t0
        sources = [os.path.basename(d.metadata.get("source", "")) for d in docs]
        rank = next((i + 1 for i, s in enumerate(sources) if s == c["source"]), 0)
        hit1 += rank == 1
        hit3 += rank > 0
        mrr += 1 / rank if rank else 0
        rows.append({"id": c["id"], "question": c["question"], "expected": c["source"],
                     "rank": rank, "top": sources, "latency_s": round(latency, 2)})

    rejects = 0
    for c in out_kb:
        docs = []
        for g in collections:
            r = retrievers.get(g)
            if r is not None:
                docs.extend(r.invoke(c["question"]))
        top = max((d.metadata.get("relevance_score", 0.0) for d in docs), default=0.0)
        ok = top < cc.RERANK_SCORE_THRESHOLD
        rejects += ok
        rows.append({"id": c["id"], "question": c["question"], "expected": "(库外)",
                     "rank": 0, "top": ["<无>"], "reject_ok": ok,
                     "top_score": round(top, 4), "latency_s": None})

    n = max(len(in_kb), 1)
    summary = {
        "n_in_kb": len(in_kb),
        "n_out_kb": len(out_kb),
        "hit_at_1": round(hit1 / n, 4),
        "hit_at_3": round(hit3 / n, 4),
        "mrr": round(mrr / n, 4),
        "reject_rate": round(rejects / max(len(out_kb), 1), 4),
        "recall_k": cc and __import__("knowledge").RECALL_K,
    }
    return summary, rows


# ---------------- 端到端问答 ----------------

def eval_e2e(llm, cases):
    """
    指标：
        answer_rate : 库内问题拿到了知识库依据（返回了 context）的比例
        keyword_hit : 回答中包含参考答案关键词的比例（粗粒度正确性）
        reject_rate : 库外问题「拒答或转联网」的比例（越低越糟，硬答=幻觉）
        first_token : 首字延迟（秒）均值
    """
    rows = []
    for c in cases:
        sid = f"eval-{c['id']}-{int(time.time() * 1000)}"
        t0 = time.perf_counter()
        first = None
        answer, has_ctx, web = "", False, False
        try:
            for chunk in llm.stream(c["question"], session_id=sid):
                if "context" in chunk:
                    has_ctx = True
                text = chunk.get("answer")
                if text:
                    if first is None:
                        first = time.perf_counter() - t0
                    answer += text
                    web = web or text.startswith(WEB_PREFIX)
        except Exception as e:  # 单个样本失败不影响整体
            answer = f"<error: {e}>"
        finally:
            llm.restore_history(sid, [])

        kws = c["answer_keywords"]
        kw_ok = (not kws) or all(k in answer for k in kws)
        rejected = web or any(m in answer for m in REFUSAL_MARKERS)
        ok = has_ctx if c["source"] else rejected
        rows.append({
            "id": c["id"], "question": c["question"], "in_kb": bool(c["source"]),
            "grounded": has_ctx, "web_fallback": web, "keyword_ok": kw_ok,
            "passed": ok, "first_token_s": round(first, 2) if first else None,
            "answer": answer[:300],
        })

    in_kb = [r for r in rows if r["in_kb"]]
    out_kb = [r for r in rows if not r["in_kb"]]
    ft = [r["first_token_s"] for r in rows if r["first_token_s"]]
    summary = {
        "n_in_kb": len(in_kb),
        "n_out_kb": len(out_kb),
        "grounded_rate": round(sum(r["grounded"] for r in in_kb) / max(len(in_kb), 1), 4),
        "keyword_hit_rate": round(sum(r["keyword_ok"] for r in in_kb) / max(len(in_kb), 1), 4),
        "no_hallucination_rate": round(sum(r["passed"] for r in out_kb) / max(len(out_kb), 1), 4),
        "pass_rate": round(sum(r["passed"] for r in rows) / max(len(rows), 1), 4),
        "first_token_avg_s": round(sum(ft) / len(ft), 2) if ft else None,
    }
    return summary, rows


# ---------------- 报告 ----------------

def write_report(result):
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (REPORTS / "latest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / f"results-{stamp}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 评测结果（{result['timestamp']}）", ""]
    for section, title in (("retrieval", "检索指标"), ("e2e", "端到端问答")):
        if section not in result:
            continue
        s = result[section]["summary"]
        lines += [f"## {title}", "", "| 指标 | 数值 |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in s.items()]
        lines.append("")
        rows = result[section]["rows"]
        if section == "retrieval":
            lines += ["| 问题 | 期望来源 | 命中排名 | Top3 来源 |", "|---|---|---|---|"]
            lines += [f"| {r['question']} | {r['expected']} | {r['rank'] or '-'} | {', '.join(r['top'])} |" for r in rows]
        else:
            lines += ["| 问题 | 库内 | 有依据 | 联网兜底 | 通过 | 首字延迟(s) |", "|---|---|---|---|---|---|"]
            lines += [f"| {r['question']} | {'是' if r['in_kb'] else '否'} | {r['grounded']} | {r['web_fallback']} | {r['passed']} | {r['first_token_s']} |" for r in rows]
        lines.append("")
    (REPORTS / "latest.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["retr", "e2e"], default=None)
    args = parser.parse_args()

    cases = load_cases()
    from knowledge import MyKnowledge
    kb = MyKnowledge()
    install_docs(kb)
    kb.load_knowledge()  # 建好索引（含留出文档，用到时再删掉它们）

    started = datetime.now().isoformat(timespec="seconds")
    result = {"timestamp": started, "threshold": None}

    if args.only in (None, "retr"):
        print("[1/2] 检索指标 ...")
        summary, rows = eval_retrieval(kb, cases)
        result["retrieval"] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 端到端必须在「库外文档已移除」的状态下跑，否则知识外问题会被误判为有依据
    cleanup_holdout(kb)

    if args.only in (None, "e2e"):
        print("[2/2] 端到端问答（会真实调用模型，耗时较长）...")
        from combine_client import CombineClient, RERANK_SCORE_THRESHOLD
        result["threshold"] = RERANK_SCORE_THRESHOLD
        summary, rows = eval_e2e(CombineClient(), cases)
        result["e2e"] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    write_report(result)
    print(f"报告已写入: {REPORTS / 'latest.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
