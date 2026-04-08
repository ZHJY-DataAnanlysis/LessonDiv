import re
from typing import List
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core.schema import Document


class FlowAwareRAG:
    def __init__(self, data_path: str):
        documents = SimpleDirectoryReader(data_path).load_data()
        self.index = KnowledgeGraphIndex.from_documents(
            documents,
            kg_triplet_extract_fn=self._extract_relations
        )

    def _extract_relations(self, text: str) -> List[tuple]:
        relations = []
        sections = re.findall(r"【(.+?)】", text)
        for i in range(len(sections) - 1):
            relations.append((sections[i], "precedes", sections[i + 1]))
        return relations

    def retrieve(self, query: str) -> List[Document]:
        # 提取查询中的教学环节关键词
        section_keywords = re.findall(r"【(.+?)】", query)
        if not section_keywords:
            return []

        # 根据教学流程检索
        return self.index.as_retriever().retrieve(
            f"教学流程关系查询: {', '.join(section_keywords)}"
        )