# train_teachingmode_classifier.py

import jsonlines
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ---------- 1. 读数据 ----------
records = []
with jsonlines.open(r"D:\Python\PythonProject\LLM_division_of_teaching_links\分类模型\无环节乱序训练\classifier_train.jsonl") as reader:
    for obj in reader:
        records.append(obj)
df = pd.DataFrame(records)

# ---------- 2. 只保留“教学模式”单标签 ----------
df['label'] = df['label'].str.split('|').str[0]          # 第一段:str[0]--teachingmode 第二段:str[1]--lessontype 第三段:str[2]--coreconcept
df = df[~df['label'].isna()].copy()

# 去掉样本数<2的稀有类别
vc   = df['label'].value_counts()
keep = vc[vc >= 2].index
df   = df[df['label'].isin(keep)].reset_index(drop=True)

print(f"可用样本数：{len(df)}，类别数：{df['label'].nunique()}")
print(df['label'].value_counts())

# # ---------- 3. 构建 pipeline ----------
# pipe = Pipeline([
#         ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
#         ('clf',  LogisticRegression(max_iter=1000, n_jobs=-1))
# ])

#          # 方案1：k=1 近邻
# pipe = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
#     ('clf',  KNeighborsClassifier(n_neighbors=1, metric='cosine'))
# ])

from sklearn.tree import DecisionTreeClassifier             # 方案2：决策树
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf',  DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42))
])

# ---------- 4. 全量训练 ----------
X, y = df['text'], df['label']
pipe.fit(X, y)

# ---------- 5. 背诵验证 ----------
pred = pipe.predict(X)
acc  = (pred == y).mean()
print(f"训练集准确率（背诵度）：{acc:.4f}")

# ---------- 6. 保存 ----------
joblib.dump(pipe, r"D:\Python\PythonProject\LLM_division_of_teaching_links\分类模型\无环节乱序训练\决策树\teachingmode_classifier_cls.pkl")
print("模型已保存为 teachingmode_classifier_cls.pkl")