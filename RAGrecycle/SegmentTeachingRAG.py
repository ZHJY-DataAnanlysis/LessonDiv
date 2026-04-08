# SegmentTeachingRAG.py
import os, re, json, httpx, numpy as np
from typing import List
import time

class SegmentTeachingRAG:
    """
    环节级知识库 RAG
    使用本地 txt 分片 + 服务器 bge + rerank
    接口与 HybridTeachingRAG 完全一致
    """
    def __init__(self,
                 kb_dir: str,
                 encoder_url: str = "http://10.154.22.11:9000",
                 reranker_url: str = "http://10.154.22.11:9001",
                 top_k_initial: int = 10,
                 top_k_final: int = 5,

                #######################################消融循环函数专用
                 use_vector: bool = True,  # 新增
                 use_rerank: bool = True  # 新增
                 #######################################消融循环函数专用

                 ):
        self.kb_dir = kb_dir
        self.encoder_url = encoder_url.rstrip("/")
        self.reranker_url = reranker_url.rstrip("/")
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        # 2. 加这两行，方便外部脚本直接改
        self._k_init = top_k_initial
        self._k_final = top_k_final

        #######################################消融循环函数专用
        self.use_vector = use_vector  # 新增
        self.use_rerank = use_rerank  # 新增
        #######################################消融循环函数专用


        # 1. 本地环节级 txt -> [(title, content)]
        self.docs = self._load_segment_docs()
        self.client = httpx.Client(timeout=10)

    # ---------------- 读本地环节级 txt ----------------
    def _load_segment_docs(self) -> List[dict]:
        docs = []
        for fn in os.listdir(self.kb_dir):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(self.kb_dir, fn)
            with open(path, encoding="utf-8") as f:
                text = f.read().strip()
            # 用【】或空行切
            for block in re.split(r'\n【|\n\n', text):
                if not block.strip():
                    continue
                title = re.search(r'^(.+?)】', '【' + block)
                title = title.group(1) if title else "未命名"
                docs.append({"title": title, "content": block.strip()})
        if not docs:
            raise RuntimeError("环节知识库为空！")
        return docs

    # ---------------- 服务器 embedding ----------------
    def _encode(self, text: str) -> List[float]:
        resp = self.client.post(f"{self.encoder_url}/encode",
                                params={"text": text})
        resp.raise_for_status()
        return resp.json()["embedding"]

    # ---------------- rerank ----------------
    def _rerank(self, query: str, passages: List[str]) -> List[float]:
        resp = self.client.post(f"{self.reranker_url}/v1/rerank",
                                json={"query": query, "documents": passages})
        resp.raise_for_status()
        return resp.json()["scores"]

    # ---------------- 主检索 ----------------




    def retrieve(self, query: str) -> str:
        # ====== 消融实验代码自动生成 ======
        #use_vector = True#原代码
        #use_rerank = True#原代码

        #######################################消融循环函数专用
        use_vector = self.use_vector  # 新增
        use_rerank = self.use_rerank  # 新增
        #######################################消融循环函数专用



        print(f"实际使用TOPK: initial={self.top_k_initial}, final={self.top_k_final}")
        print(f"消融参数:use_vector={use_vector}, use_rerank={use_rerank}")
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

            #####################################RAG内容为【教学环节名称】+对应所有内容
            #results.append(f"【{title}】\n{content}")#
            #####################################RAG内容为【教学环节名称】+对应所有内容


            #####################################RAG内容为【教学环节名称】+首句
            first = content.split("。")[0] + "。"
            results.append(f"{title}：{first}")
            #####################################RAG内容为【教学环节名称】+首句

            #################################################侦察RAG内容
            # print("检索到的片段：")
            # for content, score in final:
            #     print("---")
            #     print(content)
            #     print("score:", score)
            #################################################侦察RAG内容


            #################################################侦察RAG内容
            print("RAG筛选结果：", results)
            #################################################侦察RAG内容

        return "\n".join(results)


# def select_retrieval_strategy(model_name: str, candidates: List[dict]) -> str:
#     """
#     基于模型特性动态选择检索策略
#     """
#     # 模型注意力配置特征
#     model_configs = {
#         "Qwen3-4B": {
#             "num_heads": 32,
#             "head_dim": 128,
#             "max_seq_len": 8192,
#             "prefers_full": True
#         },
#         "DeepSeek-R1-Distill-Qwen-7B": {
#             "num_heads": 24,
#             "head_dim": 128,
#             "max_seq_len": 4096,
#             "prefers_full": False
#         },
#         "Internlm2_5-7b-chat-1m": {
#             "num_heads": 16,
#             "head_dim": 256,
#             "max_seq_len": 8192,
#             "prefers_full": False
#         }
#     }
#
#     config = model_configs.get(model_name)
#     if not config:
#         raise ValueError(f"未知模型: {model_name}")
#
#     # 计算注意力密度
#     attention_density = (config["num_heads"] * config["head_dim"]) / config["max_seq_len"]
#
#     # 决策规则：Qwen3-4B使用完整内容，其他使用首句
#     if model_name == "Qwen3-4B":
#         return _format_full_content(candidates)
#     else:
#         return _format_first_sentence(candidates)
#
#
# def _format_full_content(candidates):
#     """完整内容格式化"""
#     return "\n".join([f"【{d['title']}】\n{d['content']}" for d in candidates])
#
#
# def _format_first_sentence(candidates):
#     """首句内容格式化"""
#     return "\n".join([
#         f"{d['title']}：{d['content'].split('。')[0]}。"
#         for d in candidates
#     ])
