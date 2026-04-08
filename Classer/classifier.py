import joblib
pipe = joblib.load(r"D:\Python\PythonProject\LLM_division_of_teaching_links\分类模型\无环节乱序训练\k邻近\coreconcept_classifier_cls.pkl")  # 或 teachingmode...
t = input('\n输入新教案正文：')
print(pipe.predict([t])[0])
