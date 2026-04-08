import os
import re
from docx import Document
from tqdm import tqdm
from openai import OpenAI

# 初始化DeepSeek官方API客户端
client = OpenAI(base_url="https://api.deepseek.com/v1", api_key="sk-22ecf33f52f84f8ab94932c509eeda02")  # 替换为你的实际API key


class LessonPlanProcessor:
    def __init__(self):
        # 不再需要本地模型路径和模板设置
        pass

    def extract_text_from_docx(self, docx_path):
        """从Word文档中提取纯文本，过滤表格和图片"""
        doc = Document(docx_path)
        full_text = []

        for para in doc.paragraphs:
            if self._contains_image_or_table(para):
                continue
            text = para.text.strip()
            if text:
                full_text.append(text)

        return '\n'.join(full_text)

    def _contains_image_or_table(self, paragraph):
        """检查段落是否包含图片或表格"""
        if paragraph._element.xpath('.//pic:pic'):
            return True
        if paragraph._element.xpath('.//w:tbl'):
            return True
        return False

    def process_lesson_plan(self, text):
        """使用DeepSeek API处理教案文本"""
        user_prompt = """
        我有很多条教案word，需要针对教学环节内容转成txt，txt格式为【教学环节】+对应内容。
        提示：
        1. 大多数文档从"教学过程："开始，之后的内容才是需要的
        2. 少部分文档直接就是教学过程的内容
        3. 部分文档可能不叫"教学过程"，而是类似"课堂活动"、"教学步骤"等其他名称

        处理要求：
        - 从教学内容主体开始（跳过教学目的等前置内容）
        - 智能识别教学环节标题（可能有一、二、三或1. 2. 3.等编号，也可能没有明确标题）
        - 为每个环节添加【】标记（如【课堂导入】）
        - 严格过滤表格、图片等非文本内容
        - 只需要纯文本输出

        现在请处理以下教案内容：
        """ + text

        messages = [
            {
                "role": "system",
                "content": "你是一个专业教学文档处理助手，请严格按照用户要求处理教案"
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",  # 使用DeepSeek官方模型
                messages=messages,
                temperature=0.2,
                #max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return None

    def post_process_output(self, text):
        """后处理模型输出"""
        if not text:
            return ""

        # 标准化环节标记
        text = re.sub(r'(【|\[)?(.+?)(】|\])?', lambda m: f"【{m.group(2)}】" if m.group(2) else "", text)

        # 移除模型可能添加的解释性文字
        text = re.sub(r'Assistant:.*?(?=【|$)', '', text, flags=re.DOTALL)

        # 整理空行和格式
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def process_directory(self, input_dir, output_dir):
        """批量处理目录中的所有Word文档"""
        os.makedirs(output_dir, exist_ok=True)
        docx_files = [f for f in os.listdir(input_dir) if f.endswith('.docx')]

        for filename in tqdm(docx_files, desc="处理教案文件中"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")

            try:
                raw_text = self.extract_text_from_docx(input_path)
                processed_text = self.process_lesson_plan(raw_text)
                final_text = self.post_process_output(processed_text)

                if final_text:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(final_text)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    processor = LessonPlanProcessor()

    # 配置路径
    input_directory = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\数据集\几万条word教案"
    output_directory = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\数据集\txt数据集_1"

    # 开始处理
    processor.process_directory(input_directory, output_directory)
    print("所有教案文件处理完成！")