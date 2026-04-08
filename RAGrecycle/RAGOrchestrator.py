# RAGOrchestrator.py
from typing import Literal
import logging
import time
from HybridTeachingRAG import HybridTeachingRAG
from SegmentTeachingRAG import SegmentTeachingRAG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeachingRAGOrchestrator:
    def __init__(
        self,
        encoder_url: str = "http://10.154.22.11:9000",
        reranker_url: str = "http://10.154.22.11:9001",
        knowledge_base_path: str = "knowledge_base",
        timeout: int = 300
    ):
        self.rags = {
            # "hybrid": HybridTeachingRAG(
            #     encoder_url=encoder_url.rstrip("/"),
            #     reranker_url=reranker_url.rstrip("/"),
            #     knowledge_base_path=knowledge_base_path,
            #     params={
            #         "top_k_initial": 5,
            #         "top_k_final": 3,
            #         "embedding_timeout": timeout,
            #         "reranker_timeout": timeout,
            #         "embedding_timeout": None,
            #         "reranker_timeout": None  # ← 改为 None
            #     }
            # )

            "segment": SegmentTeachingRAG(
                kb_dir=r"D:\研究生\项目组\科研\小模型+RAG暑期实验\RAG知识库_3",  # 环节级 txt 目录
                encoder_url="http://10.154.22.11:9000",
                reranker_url="http://10.154.22.11:9001",
                #top_k_initial=10,  # 可再调小
                #top_k_final=6,

            )

        }
        logger.info("RAG系统初始化完成")



    ####################################################################topk循环函数专用
    def make_segment_rag(self, k_i, k_f,use_vector: bool = True,use_rerank: bool = True):
        return SegmentTeachingRAG(
            kb_dir=r"D:\研究生\项目组\科研\小模型+RAG暑期实验\RAG知识库_3",
            encoder_url="http://10.154.22.11:9000",
            reranker_url="http://10.154.22.11:9001",
            top_k_initial=k_i,     #TOPK循环专用
            top_k_final=k_f,       #TOPK循环专用
            use_vector=use_vector, #消融循环专用
            use_rerank=use_rerank  #消融循环专用
        )

    ####################################################################topk循环函数专用



    def process(self, query: str, mode: str = "segment"):
        try:
            start_time = time.time()
            result = self.rags[mode].retrieve(query)
            logger.info(f"RAG完成，耗时 {time.time()-start_time:.2f}秒")
            return result
        except KeyError:
            raise ValueError(f"不支持的RAG模式: {mode}")

