# split_kb_txt_1.py
import re, os, uuid

# ====== 只需改这两行 ======
IN_DIR  = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\完整教案\教学环节txt"
OUT_DIR = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\RAG知识库_3"
# ==========================

os.makedirs(OUT_DIR, exist_ok=True)

for fn in os.listdir(IN_DIR):
    if not fn.endswith(".txt"):
        continue

    with open(os.path.join(IN_DIR, fn), encoding="utf-8") as f:
        text = f.read()

    # 用正向预查保留【作为分隔点，且保留【符号
    # split 之后，第 0 段是无效内容，从第 1 段开始
    segments = re.split(r'(?=【)', text)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        title_match = re.match(r'【([^】]+)】', seg)
        if not title_match:          # 没有标题的段落直接跳过
            continue
        title = title_match.group(1)
        safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)
        doc_id = uuid.uuid4().hex[:8]
        out_path = os.path.join(OUT_DIR, f"{doc_id}_{safe_title}.txt")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(seg)

print("✅ 已切分完成，共生成", len(os.listdir(OUT_DIR)), "个环节级 txt")