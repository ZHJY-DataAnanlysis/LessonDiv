# ablation_run.py
import os, json, time, itertools, shutil
from typing import Dict, List
from LLMtest_evaluate8_RAG import batch_process_with_eval
from RAGOrchestrator import TeachingRAGOrchestrator

MODELS = {
    #"DeepSeek-R1-Distill-Qwen-7B":"/home/lhr/7Bmodel/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen3": "/home/lhr/7Bmodel/Qwen3-4B",
    #"InternLM2-5-7B-Chat-1M": "/home/lhr/7Bmodel/internlm2_5-7b-chat-1m",
}


# ========== 1. 三种消融配置 ==========
ABLATION_CFG = {
    "full":      {"use_vector": True,  "use_rerank": True},
    "-rerank":   {"use_vector": True,  "use_rerank": False},
    "-vector":   {"use_vector": False, "use_rerank": True},
}

# ========== 2. 公共路径 ==========
INPUT_DIR   = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\测试集\测试集2_删去环节名称并打乱顺序"
GT_JSON     = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\教学环节txt_json\teaching_plan.json"
BASE_OUT    = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\测试结果\ablation_results"

# ========== 3. 运行函数 ==========
def run_ablation():
    summary_all: Dict[str, Dict] = {}
    for ablation, flags in ABLATION_CFG.items():
        print(f"\n========== 开始消融实验：{ablation} ==========")
        for model_name, model_path in MODELS.items():      # ← 新增模型循环
            print(f"\n--- 模型：{model_name} ---")
            out_dir = os.path.join(BASE_OUT, ablation, model_name)
            os.makedirs(out_dir, exist_ok=True)

            # 构造带参 orchestrator
            orch = TeachingRAGOrchestrator(
                encoder_url="http://10.154.22.11:9000",
                reranker_url="http://10.154.22.11:9001"
            )
            orch.rags["segment"] = orch.make_segment_rag(
                12, 10,                         # 固定 TOPK
                use_vector=flags["use_vector"],
                use_rerank=flags["use_rerank"]
            )

            # 跑评估
            batch_process_with_eval(INPUT_DIR, out_dir, GT_JSON, orch)

            # 收集结果
            summary_path = os.path.join(out_dir, f"{model_name}_eval.json")
            if os.path.exists(summary_path):
                with open(summary_path, encoding="utf-8") as f:
                    summary_all[f"{ablation}_{model_name}"] = json.load(f)["summary"]

    # 写总表
    with open(os.path.join(BASE_OUT, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_all, f, ensure_ascii=False, indent=2)
    print("\n========== 消融实验全部完成 ==========")
    for k, v in summary_all.items():
        print(f"{k:8s}  综合评分: {v.get('average_metrics', {}).get('综合评分', 'N/A')}")

# ========== 4. 动态生成消融版 SegmentTeachingRAG ==========
def patch_segment_rag(flags: Dict[str, bool]):
    """
    根据 flags 在本地生成一份临时的 SegmentTeachingRAG.py
    仅改动 retrieve() 逻辑，其他不变
    """
    # 读原文件
    with open("SegmentTeachingRAG.py", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 找到 retrieve 函数的起止行
    start, end = -1, -1
    for i, l in enumerate(lines):
        if l.strip().startswith("def retrieve("):
            start = i
        if start != -1 and l.strip() == "" and l == lines[-1]:
            end = len(lines)
            break
    if end == -1:
        end = len(lines)

    # 生成新的 retrieve
    new_retrieve = _build_retrieve_function(flags)
    patched = lines[:start] + [new_retrieve] + lines[end:]

    # 写回（覆盖）
    with open("SegmentTeachingRAG.py", "w", encoding="utf-8") as f:
        f.writelines(patched)

# ========== 5. 生成 retrieve 代码字符串 ==========
def _build_retrieve_function(flags: Dict[str, bool]) -> str:
    use_v = flags["use_vector"]
    use_r = flags["use_rerank"]

    code = f"""
    def retrieve(self, query: str) -> str:
        # ====== 消融实验代码自动生成 ======
        use_vector = {use_v}
        use_rerank = {use_r}

        # 1. 向量召回（或随机召回）
        if use_vector:
            query_emb = self._encode(query)
            doc_embs  = [self._encode(d["content"]) for d in self.docs]
            sims = [np.dot(query_emb, e) for e in doc_embs]
            cand_idx = np.argsort(sims)[-self.top_k_initial:][::-1]
            candidates = [self.docs[i] for i in cand_idx]
        else:
            # 随机召回
            cand_idx = np.random.choice(len(self.docs), size=min(self.top_k_initial, len(self.docs)), replace=False)
            candidates = [self.docs[i] for i in cand_idx]

        # 2. rerank（或跳过）
        passages = [d["content"] for d in candidates]
        if use_rerank:
            scores = self._rerank(query, passages)
            reranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
            final = reranked[:self.top_k_final]
        else:
            final = [(p, 0.0) for p in passages[:self.top_k_final]]

        # 3. 结果格式化
        results = []
        for content, _ in final:
            title = re.search(r'^【(.+?)】', content)
            title = title.group(1) if title else ""
            first = content.split("。")[0] + "。"
            results.append(f"{{title}}：{{first}}")
        return "\\n".join(results)
"""
    return code

# ========== 6. 入口 ==========
if __name__ == "__main__":
    run_ablation()