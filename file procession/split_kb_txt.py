# split_kb_txt.py
import re, os, uuid

# ====== 你只需改这两行 ======
IN_DIR  = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\完整教案\教学环节txt"
OUT_DIR = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\RAG知识库_2"
# ===========================

os.makedirs(OUT_DIR, exist_ok=True)

for fn in os.listdir(IN_DIR):
    if not fn.endswith(".txt"):
        continue
    with open(os.path.join(IN_DIR, fn), encoding="utf-8") as f:
        text = f.read()

    for block in re.split(r'\n【|\n\n', text):
        block = block.strip()
        if not block:
            continue
        title = re.search(r'^(.+?)】', '【' + block)
        title = title.group(1) if title else "未命名"
        safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)
        doc_id = uuid.uuid4().hex[:8]
        out_path = os.path.join(OUT_DIR, f"{doc_id}_{safe_title}.txt")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(block)

print("✅ 已切分完成，共生成", len(os.listdir(OUT_DIR)), "个环节级 txt")