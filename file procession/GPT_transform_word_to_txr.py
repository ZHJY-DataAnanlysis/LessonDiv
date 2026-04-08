# -*- coding: utf-8 -*-
import os
from openai import OpenAI  # 新版导入方式
from docx import Document
import time

# 初始化客户端 - 新版API
client = OpenAI(api_key="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" )  # 替换为你的实际API key


def extract_with_gpt(text: str) -> str:
    """
    使用GPT智能提取教学环节内容
    """
    prompt = f"""
    我有很多条教案word，需要针对教学环节内容转成txt，txt格式为【教学环节】+对应内容。提示：也就是大多数文档中的教学过程开始，教学过程：...之后的内容，少部分文档是直接就是教学过程的内容，部分文档或许不叫教学过程，可能叫其他，但他是类似教学过程那部分内容。前面说到，从教学过程开始（大部分文档），那些教学目的什么的就不要放在txt里面了，你就从教学内容后面开始，捕捉他们的教学环节标题，有些会有一、二、三、四、这样的，有些没有，这个就要看你聪不聪明了，你成功捕捉到教学环节标题，然后你自己给它加上【】，同时，对于含有表格、图片等格式的word，你需要把那些给我过滤掉，因为我只需要纯文本。输出所有的txt给我。同时我需要你给每个txt命名从序号00001开始。
    请从以下教案文本中提取教学环节内容，要求：
1. 从"教学过程"或类似章节开始提取
2. 识别所有教学环节小节（如一、二、三或1. 2. 3.等格式）
3. 每个小节用【】标注标题
4. 只返回处理后的内容，不要解释

示例输出格式：
【创设情境】
这里是情境内容...
【探究新知】
这里是新知内容...

待处理文本：
{text}"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的教案处理助手，能精准提取教学环节内容。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"GPT处理出错: {str(e)}")
        return None


def process_docx_with_gpt(docx_path: str, output_dir: str) -> None:
    """
    使用GPT处理单个Word文档
    """
    try:
        doc = Document(docx_path)
        full_text = '\n'.join([para.text for para in doc.paragraphs])

        # 使用GPT提取内容
        extracted_content = extract_with_gpt(full_text)
        if not extracted_content:
            print(f"GPT未返回有效内容: {docx_path}")
            return

        # 保存结果
        base_name = os.path.splitext(os.path.basename(docx_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_content)

        print(f"成功处理: {docx_path}")
        time.sleep(1)  # 避免API速率限制

    except Exception as e:
        print(f"处理文件 {docx_path} 时出错: {str(e)}")


def batch_process_with_gpt(input_dir: str, output_dir: str) -> None:
    """
    批量处理所有Word文档（GPT版本）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    docx_files = [f for f in os.listdir(input_dir) if f.endswith('.docx')]
    total_files = len(docx_files)

    print(f"找到 {total_files} 个Word文档待处理")

    for i, filename in enumerate(docx_files, 1):
        docx_path = os.path.join(input_dir, filename)
        print(f"正在处理文件 {i}/{total_files}: {filename}")
        process_docx_with_gpt(docx_path, output_dir)

    print("所有文件处理完成")


if __name__ == "__main__":
    # 配置输入输出目录
    input_directory = "D:\研究生\项目组\科研\小模型+RAG暑期实验\数据集\几万条word教案"  # 存放Word文档的目录
    output_directory = "D:\研究生\项目组\科研\小模型+RAG暑期实验\数据集\GPTtest数据集"  # 输出TXT文件的目录

    # 执行批量处理
    batch_process_with_gpt(input_directory, output_directory)