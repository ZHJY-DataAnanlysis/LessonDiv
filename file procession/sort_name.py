import os

# 设置目标文件夹路径
folder_path = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\汇总"  # ← 修改为你自己的文件夹路径

# 获取所有 .doc 和 .docx 文件
files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.doc', '.docx'))]

# 排序方式：按文件名排序（也可以改为按修改时间）
files.sort()  # 如果想按修改时间排序：key=lambda x: os.path.getmtime(os.path.join(folder_path, x))

# 批量重命名
for idx, filename in enumerate(files, start=1):
    old_path = os.path.join(folder_path, filename)

    # 防止重复添加序号
    base_name = filename
    if base_name[:3].isdigit() and base_name[3] == '_':
        print(f"跳过已加序号: {filename}")
        continue

    new_name = f"{idx:02d}_{filename}"
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    print(f"重命名: {filename} → {new_name}")