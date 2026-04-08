import os
import re
from docx import Document
from tqdm import tqdm


class LessonPlanProcessor:
    def __init__(self, use_api=False):
        self.use_api = use_api  # 是否使用API（默认False，优先本地规则）
        if self.use_api:
            from openai import OpenAI
            self.client = OpenAI(base_url="https://api.deepseek.com/v1", api_key="your-api-key")

    def extract_text_from_docx(self, docx_path):
        """从Word提取纯文本，过滤表格/图片"""
        doc = Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            if self._has_table_or_image(para):
                continue
            text = para.text.strip()
            if text:
                full_text.append(text)
        return '\n'.join(full_text)

    def _has_table_or_image(self, paragraph):
        """检查段落是否包含表格或图片"""
        return (paragraph._element.xpath('.//w:tbl') or
                paragraph._element.xpath('.//pic:pic'))

    def _extract_with_rules(self, text):
        """规则提取：智能匹配教学环节"""
        # 1. 找到"教学过程"或类似的关键词
        start_keywords = ["教学过程", "教学步骤", "课堂活动", "教学环节"]
        start_pattern = re.compile(rf'({"|".join(start_keywords)})[:：]?\s*(\n|$)')
        match = start_pattern.search(text)

        if not match:
            # 如果没有关键词，尝试从第一个编号开始（如 "一、" 或 "1."）
            numbered_pattern = re.compile(r'(^|\n)([一二三四1-9]+)[、.)]')
            match = numbered_pattern.search(text)
            if not match:
                return None  # 规则无法处理，需调用API

        start_pos = match.start() if match else 0
        content = text[start_pos:]

        # 2. 提取教学环节（带编号或标题的段落）
        # 匹配 "一、导入" 或 "1. 讲解" 等
        section_pattern = re.compile(r'(\n|^)([一二三四1-9]+)[、.)]\s*(.*?)(?=\n|$)')
        sections = section_pattern.findall(content)

        if not sections:
            # 如果没有编号，尝试按自然段落分割
            sections = [(m.start(), m.group()) for m in re.finditer(r'\n\s*\S+', content)]
            if not sections:
                return None

        # 3. 格式化为【教学环节】+内容
        result = []
        for _, num, title in sections:
            if title.strip():
                result.append(f"【{title.strip()}】")

        # 如果没有提取到标题，则直接返回内容
        return '\n'.join(result) if result else content

    def process_lesson_plan(self, text):
        """处理教案文本（优先规则，失败则调用API）"""
        processed = self._extract_with_rules(text)
        if processed or not self.use_api:
            return processed

        # 调用DeepSeek API（仅当规则失败且启用API时）
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                    "role": "user",
                    "content": f"从以下教案中提取教学环节，格式为【标题】+内容：\n{text}"
                }],
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {e}")
            return None

    def process_directory(self, input_dir, output_dir):
        """批量处理Word文档"""
        os.makedirs(output_dir, exist_ok=True)
        docx_files = [f for f in os.listdir(input_dir) if f.endswith('.docx')]

        for filename in tqdm(docx_files, desc="处理教案"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")

            try:
                text = self.extract_text_from_docx(input_path)
                processed = self.process_lesson_plan(text)
                if processed:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(processed)
            except Exception as e:
                print(f"处理失败 {filename}: {e}")


if __name__ == "__main__":
    processor = LessonPlanProcessor(use_api=False)  # 默认禁用API
    input_dir = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\数据集\几万条word教案"
    output_dir = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\数据集\txt数据集"
    processor.process_directory(input_dir, output_dir)