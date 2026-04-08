import requests
r = requests.post(
    "http://10.154.22.11:9001/v1/rerank",
    json={"query": "分数加法", "documents": ["3/8+2/8=5/8", "分数加法规则"]},
    timeout=10
)
print(r.status_code, r.json())