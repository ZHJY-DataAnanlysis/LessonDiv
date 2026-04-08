#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trans_micro_jsonl.py   ——最终修复后缀问题
"""
import json, re
from pathlib import Path
import pandas as pd

# ===== 0. 路径 =====
TXT_DIR    = Path(r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\完整教案\教学环节txt")
EXCEL_FILE = Path(r"第一批2023年11月12日-小学、初中数学教学资源标签数据.xlsx")
OUTPUT_FILE= Path("macro_template.jsonl")

# ===== 1. 读 Excel：去掉“数据类型-”前缀 + 文件扩展名 =====
df = pd.read_excel(EXCEL_FILE, sheet_name="Sheet1").dropna(subset=["数据类型-数据"])
# 1. 去前缀  2. 去扩展名
df["real_name"] = (df["数据类型-数据"]
                   .str.replace(r"^\w+-", "", regex=True)   # 去掉前缀
                   .str.replace(r"\.[^.]+$", "", regex=True) # 去掉 .docx/.pptx 等
                  )

excel_map = {row["real_name"]: {
                "教学模式": str(row.get("教学模式", "")),
                "课型": str(row.get("课型", "")),
                "核心概念": str(row.get("核心概念", ""))
            } for _, row in df.iterrows()}

# ===== 2. 读 txt：去掉「数字+_教案-」前缀 =====
gold_files = {f.stem: f for f in TXT_DIR.glob("*.txt")}
sorted_stems = sorted(gold_files.keys(), key=lambda x: int(re.match(r"(\d+)", x).group(1)))

records = []
for stem in sorted_stems:
    real_name = re.sub(r"^\d+_教案-", "", stem)   # 与 excel_map 保持同名
    txt_file  = gold_files[stem]

    text  = txt_file.read_text(encoding="utf-8")
    steps = re.findall(r'【([^】]+)】', text)

    attr  = excel_map.get(real_name, {})
    records.append({
        "教学模式": attr.get("教学模式", ""),
        "课型": attr.get("课型", ""),
        "核心概念": attr.get("核心概念", ""),
        "教学环节及其顺序": steps
    })

# ===== 3. 写 jsonl =====
with OUTPUT_FILE.open("w", encoding="utf-8") as fw:
    for rec in records:
        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"完成！共 {len(records)} 条 → {OUTPUT_FILE.resolve()}")