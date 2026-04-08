from openai import OpenAI

# 初始化客户端
client = OpenAI(api_key="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")  # 替换为你的实际 Key

# 调用 GPT-4o 模型
response = client.chat.completions.create(
    model="gpt-4o",  # 指定模型
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "请解释什么是相对论。"}
    ],
    temperature=0.7,  # 控制随机性（0-2，越高回答越多样）
)

# 输出回复
print(response.choices[0].message.content)