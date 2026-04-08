from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document


class PedagogyEnhancedRAG:
    def __init__(self, encoder_client, data_path: str):
        self.encoder_client = encoder_client
        documents = SimpleDirectoryReader(data_path).load_data()

        # 使用教学环节感知的分割器
        self.parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"【.+?】",
        )

        nodes = self.parser.get_nodes_from_documents(documents)
        self.index = VectorStoreIndex(nodes)

    def _encode(self, text: str) -> List[float]:
        response = self.encoder_client.embeddings.create(
            input=text,
            model="bge-small-zh-v1.5"
        )
        return response.data[0].embedding

    def retrieve(self, query: str) -> List[Document]:
        # 增强教学相关查询
        enhanced_query = f"教学相关内容: {query}"
        return self.index.as_retriever().retrieve(enhanced_query)