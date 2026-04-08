from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import Document


class HierarchicalTeachingRAG:
    def __init__(self, encoder_client, data_path: str):
        self.encoder_client = encoder_client
        documents = SimpleDirectoryReader(data_path).load_data()

        # 分层节点解析
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[1024, 512, 256]
        )
        nodes = node_parser.get_nodes_from_documents(documents)

        # 创建分层索引
        self.index = VectorStoreIndex(nodes)

    def _encode(self, text: str) -> List[float]:
        response = self.encoder_client.embeddings.create(
            input=text,
            model="bge-small-zh-v1.5"
        )
        return response.data[0].embedding

    def retrieve(self, query: str) -> List[Document]:
        # 分层检索
        return self.index.as_retriever(
            similarity_top_k=3,
            level="section"  # 可以指定检索层级: course/section/subsection
        ).retrieve(query)