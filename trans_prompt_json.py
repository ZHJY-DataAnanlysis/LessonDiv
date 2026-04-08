#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifier_train.py   ——去掉「教案-」前缀版
"""
import json, re
from pathlib import Path
import pandas as pd

# ===== 0. 路径 =====
TXT_DIR    = Path(r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\测试集\测试集2_删去环节名称并打乱顺序")
EXCEL_FILE = Path(r"第一批2023年11月12日-小学、初中数学教学资源标签数据.xlsx")
OUTPUT_FILE= Path("classifier_train.jsonl")

# ===== 1. 读 Excel：key = 去掉「数据类型-」前缀 + 扩展名 =====
df = pd.read_excel(EXCEL_FILE, sheet_name="Sheet1").dropna(subset=["数据类型-数据"])
df["real_name"] = (df["数据类型-数据"]
                   .str.replace(r"^\w+-", "", regex=True)   # 去前缀
                   .str.replace(r"\.[^.]+$", "", regex=True)) # 去扩展名

excel_map = {row["real_name"]: f"{row.get('教学模式', '')}|{row.get('课型', '')}|{row.get('核心概念', '')}"
             for _, row in df.iterrows()}

# ===== 2. 遍历 txt：去掉「数字+_教案-」和「教案-」前缀 =====
records = []
for txt_file in sorted(TXT_DIR.glob("*.txt")):
    stem = txt_file.stem
    # 先去掉 01_ 再干掉「教案-」
    real_name = re.sub(r"^\d+_教案-", "", stem)
    label = excel_map.get(real_name, None)

    # 调试用：随机抽查 3 条
    if len(records) < 3:
        print(f"[调试] stem={stem}  →  real_name={real_name}  →  label={label}")

    if label is None:
        label = "||"
    text = txt_file.read_text(encoding="utf-8").strip()
    records.append({"text": text, "label": label})

# ===== 3. 写 jsonl =====
with OUTPUT_FILE.open("w", encoding="utf-8") as fw:
    for rec in records:
        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"完成！共 {len(records)} 条 → {OUTPUT_FILE.resolve()}")