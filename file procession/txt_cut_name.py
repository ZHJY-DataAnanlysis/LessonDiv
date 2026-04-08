import os
import re
from pathlib import Path


def process_teaching_materials(input_folder, output_folder):
    """
    处理教学材料文件：
    1. 删除【】标注的教学环节名称
    2. 保留每个环节的完整教学内容
    3. **保持原始顺序**（不进行打乱）
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 使用正则表达式分割文本，保留【】标注和内容
            segments = re.split(r'(【.+?】)', content)

            # 提取环节内容和对应的原始环节名称
            original_segments = []
            for i in range(1, len(segments), 2):
                segment_name = segments[i]
                segment_content = segments[i + 1].strip()
                original_segments.append((segment_name, segment_content))

            # 不再打乱顺序，直接按原顺序拼接内容
            ordered_content = '\n\n'.join([content for _, content in original_segments])

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ordered_content)

            print(f"处理完成: {filename}")


if __name__ == "__main__":
    input_folder = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\教学环节txt"  # 替换为你的原始文件文件夹路径
    output_folder = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\测试集1_删去环节名称"  # 替换为你想要的输出文件夹路径

    process_teaching_materials(input_folder, output_folder)