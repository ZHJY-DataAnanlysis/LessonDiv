# -*- coding: utf-8 -*-
"""
统计文件夹内 .doc/.docx/.txt
1. 排除 0 字后输出最少 / 最多字数文件
2. 区间统计 + 柱状图
3. 统计含有【】的文档个数及列表
"""
import os
import re
from pathlib import Path
import logging
import win32com.client as win32
from docx import Document
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm

# =============== ① 改成你的文件夹 ===============
FOLDER = r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\完整教案\原文档txt"
# =================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


# ---------------- 工具 ----------------
def safe_com_close(app, doc=None):
    try:
        if doc:
            doc.Close(False)
        if app:
            app.Quit()
    except:
        pass


def chinese_chars(txt: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', txt))


def has_brackets(txt: str) -> bool:
    """判断是否包含【】"""
    return bool(re.search(r'【.*?】', txt, re.DOTALL))


def read_txt(fp: Path) -> str:
    try:
        return fp.read_text(encoding='utf-8', errors='ignore')
    except:
        return fp.read_text(encoding='gbk', errors='ignore')


def read_doc(fp: Path) -> str:
    word = None
    doc = None
    try:
        word = win32.Dispatch("Word.Application")
        doc = word.Documents.Open(str(fp), Encoding='utf-8')
        return doc.Content.Text
    except Exception as e:
        logging.error("读 .doc 失败：%s  %s", fp, e)
        return ""
    finally:
        safe_com_close(word, doc)


def read_docx(fp: Path) -> str:
    doc = Document(fp)
    return ''.join(p.text for p in doc.paragraphs)


def count_and_brackets(fp: Path) -> tuple[int, bool]:
    suffix = fp.suffix.lower()
    text = ""
    if suffix == '.txt':
        text = read_txt(fp)
    elif suffix == '.docx':
        text = read_docx(fp)
    elif suffix == '.doc':
        text = read_doc(fp)
    return chinese_chars(text), has_brackets(text)


# ---------------- 主流程 ----------------
def main():
    folder = Path(FOLDER)
    if not folder.is_dir():
        logging.error("路径不存在：%s", folder)
        return

    files = [fp for fp in folder.rglob("*") if fp.suffix.lower() in {'.txt', '.doc', '.docx'}]
    if not files:
        logging.warning("未找到任何 .txt/.doc/.docx 文件！")
        return

    records = []          # [(fp, cnt, has_bracket), ...]
    bracket_files = []    # 保存含有【】的文件路径
    for fp in tqdm(files, desc="进度", unit="个", ncols=80):
        cnt, flag = count_and_brackets(fp)
        records.append((fp, cnt, flag))
        if flag:
            bracket_files.append(fp)

    # ****** 排除 0 字 ******
    non_zero = [(fp, cnt, flag) for fp, cnt, flag in records if cnt > 0]
    if not non_zero:
        print("所有文档字数均为 0，无有效统计！")
        return

    non_zero.sort(key=lambda x: x[1])
    min_fp, min_cnt, _ = non_zero[0]
    max_fp, max_cnt, _ = non_zero[-1]

    print("\n========== 最少（>0） / 最多 ==========")
    print(f"最少：{min_cnt:5d} 字  -->  {min_fp}")
    print(f"最多：{max_cnt:5d} 字  -->  {max_fp}")

    # ========== 区间统计（含 0） ==========
    bins = list(range(900, 3101, 200))
    counts = {b: 0 for b in bins}
    for _, cnt, _ in records:
        for b in bins:
            if b <= cnt < b + 200:
                counts[b] += 1
                break

    print("\n========== 区间统计（含 0） ==========")
    for b in bins:
        print(f"{b:4d}~{b+199:4d} 字：{counts[b]:3d} 个")

    # ========== 【】统计 ==========
    print(f"\n含有【】的文档共 {len(bracket_files)} 个")
    if len(bracket_files) <= 20:  # 不多时直接列出
        for fp in bracket_files:
            print("  ", fp)
    else:
        print("  文件较多，仅显示前 20 个：")
        for fp in bracket_files[:20]:
            print("  ", fp)

    # ========== 柱状图 ==========
    rcParams['font.family'] = 'SimHei'
    plt.figure(figsize=(10, 5))
    x = [f"{b}-{b+199}" for b in bins]
    y = [counts[b] for b in bins]
    bars = plt.bar(x, y, color='steelblue')
    plt.title("文档字数分布", fontsize=14)
    plt.xlabel("字数区间")
    plt.ylabel("文件个数")
    plt.xticks(rotation=45)
    for bar, num in zip(bars, y):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(num), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()