#train_classifier.py
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib, pandas as pd

df = pd.read_json("lesson_type_train1.jsonl", lines=True)
pipe = make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1,2)),
                     LogisticRegression(max_iter=1000))
X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, test_size=0.1, random_state=42)
pipe.fit(X_train, y_train)
print("acc:", pipe.score(X_test, y_test))   # 一般 >0.92
joblib.dump(pipe, "lesson_type_cls.pkl")