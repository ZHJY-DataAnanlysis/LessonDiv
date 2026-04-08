# tune_topk.py
# tune_topk.py
"""
一次性网格搜索 top_k_initial / top_k_final
要求：
  SegmentTeachingRAG.__init__(..., top_k_initial, top_k_final)
  TeachingRAGOrchestrator.make_segment_rag(k_i, k_f) -> SegmentTeachingRAG
"""

import os
import json
import csv
import time
from LLMtest_evaluate8_RAG import batch_process_with_eval
from RAGOrchestrator import TeachingRAGOrchestrator

# ---------------- 配置 ----------------
TOPK_GRID = [(i, f) for i in [6]      # 召回篇数
                     for f in [10]]  # 最终篇数

MODEL_NAME   = "Qwen3"
INPUT_DIR    = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\测试集\测试集2_删去环节名称并打乱顺序"
GT_JSON      = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\教学环节txt_json\teaching_plan.json"
OUT_BASE     = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\测试结果\topk_tune_results_test"

os.makedirs(OUT_BASE, exist_ok=True)

# ---------------- 主循环 ----------------
def run_all_topk():
    records = []
    for k_i, k_f in TOPK_GRID:
        print(f"\n====== 正在测试  initial={k_i}, final={k_f}  ======")

        # 新建 orchestrator，并用工厂方法注入本次 topk
        orch = TeachingRAGOrchestrator(
            encoder_url="http://10.154.22.11:9000",
            reranker_url="http://10.154.22.11:9001"
        )
        orch.rags["segment"] = orch.make_segment_rag(k_i, k_f)
        print("构造后实例 top_k =", orch.rags["segment"].top_k_initial, orch.rags["segment"].top_k_final)
        # 运行评估
        out_dir = os.path.join(OUT_BASE, f"i{k_i}_f{k_f}")
        batch_process_with_eval(INPUT_DIR, out_dir, GT_JSON,orch)

        # 读取平均指标
        summary_path = os.path.join(out_dir, f"{MODEL_NAME}_eval.json")
        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                avg = json.load(f)["summary"]["average_metrics"]
            rec = {"initial": k_i, "final": k_f, **avg}
            records.append(rec)
            print(rec)          # 实时查看
        else:
            print("⚠️ 未生成 eval.json，可能评估脚本异常")

    # 写 CSV
    if records:
        csv_path = os.path.join(OUT_BASE, "topk_results.csv")
        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"\n✅ 全部结果已汇总到 {csv_path}")
    else:
        print("\n❌ 没有任何有效记录，请检查环境与脚本")

# ---------------- 入口 ----------------
if __name__ == "__main__":
    run_all_topk()
