# HybridTeachingRAG.py
import os
import json
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import numpy as np
import httpx  # 用于非 OpenAI 格式接口

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridTeachingRAG:
    def __init__(
        self,
        encoder_url: str = "http://10.154.22.11:9000",
        reranker_url: str = "http://10.154.22.11:9001",
        knowledge_base_path: str = "knowledge_base",
        params: Dict[str, Any] = None
    ):
        self.encoder_url = encoder_url.rstrip("/")
        self.reranker_url = reranker_url.rstrip("/")
        self.encoder_client = httpx.Client(timeout=None)
        self.reranker_client = httpx.Client(timeout=None)

        self.params = {
            "top_k_initial": 10,
            "top_k_final": 3,
            **(params or {})
        }

        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.doc_embeddings = self._precompute_embeddings()

    # ---------------- 知识库加载（支持文件夹） ----------------
    def _load_knowledge_base(self, path: str) -> List[str]:
        docs = []
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                docs = [f.read().strip()]
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            docs.append(content)
        else:
            raise FileNotFoundError(f"知识库路径无效: {path}")
        if not docs:
            raise ValueError("知识库为空或未找到任何文档")
        return docs

    # ---------------- 嵌入接口（兼容 /encode） ----------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def _get_embedding_with_retry(self, text: str) -> List[float]:
        response = self.encoder_client.post(
            f"{self.encoder_url}/encode",
            params={"text": text},  # 注意是 params
            timeout=None
        )
        response.raise_for_status()
        return response.json()["embedding"]

    # ---------------- 预计算嵌入 ----------------
    def _precompute_embeddings(self) -> List[List[float]]:
        embeddings = []
        for i, doc in enumerate(self.knowledge_base):
            try:
                emb = self._get_embedding_with_retry(doc)
                embeddings.append(emb)
                logger.info(f"文档 {i+1}/{len(self.knowledge_base)} 嵌入完成")
            except Exception as e:
                logger.error(f"文档 {i+1} 嵌入失败: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

    # ---------------- rerank 接口（兼容 /v1/rerank） ----------------
    def _rerank(self, query: str, passages: List[str]) -> List[float]:
        try:
            response = self.reranker_client.post(
                f"{self.reranker_url}/v1/rerank",
                json={"query": query, "documents": passages},
                timeout=None
            )
            response.raise_for_status()
            return response.json()["scores"]
        except Exception as e:
            logger.error(f"rerank 失败: {e}")
            return [0.0] * len(passages)

    # ---------------- 余弦相似度 ----------------
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    # ---------------- 主检索 ----------------
    def retrieve(self, query: str) -> List[str]:
        query_emb = self._get_embedding_with_retry(query)
        sim_scores = [self._cosine_similarity(query_emb, doc_emb) for doc_emb in self.doc_embeddings]

        candidates = sorted(
            zip(self.knowledge_base, sim_scores),
            key=lambda x: x[1],
            reverse=True
        )[:self.params["top_k_initial"]]

        if len(candidates) > 1:
            passages = [doc for doc, _ in candidates]
            scores = self._rerank(query, passages)
            final = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)[:self.params["top_k_final"]]
            return [doc for doc, _ in final]

        return [doc for doc, _ in candidates][:self.params["top_k_final"]]