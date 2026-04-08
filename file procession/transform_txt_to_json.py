import os
import json
import re
from collections import OrderedDict


def parse_teaching_segments(content):
    """解析用【】标注的教学环节"""
    pattern = re.compile(r'【(.*?)】')  # 匹配【标题】格式
    segments = OrderedDict()
    last_pos = 0
    last_title = "前置内容"

    for match in pattern.finditer(content):
        title = match.group(1)
        start_pos = match.start()
        segments[last_title] = content[last_pos:start_pos].strip()
        last_title = title
        last_pos = match.end()

    # 添加最后一个环节的内容
    segments[last_title] = content[last_pos:].strip()

    # 过滤空内容
    return {k: v for k, v in segments.items() if v}


def process_files_to_json(input_folder, output_json):
    """处理文件夹中的所有TXT文件"""
    result = {}

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析教学环节
            segments = parse_teaching_segments(content)
            if segments:
                result[filename] = segments

    # 保存为JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"成功处理 {len(result)} 个文件，结果已保存到 {output_json}")


# 使用示例
if __name__ == "__main__":
    input_folder = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\教学环节txt_json"  # 存放TXT文件的文件夹
    output_json = "teaching_plan.json"  # 输出JSON文件

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"请将TXT文件放入 {input_folder} 文件夹后重新运行程序")
    else:
        process_files_to_json(input_folder, output_json)