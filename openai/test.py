# from openai import OpenAI
# client = OpenAI()

# response = client.responses.create(
#     model="gpt-5.4",
#     input="Write a one-sentence bedtime story about a unicorn."
# )

# print(response.output_text)
# test.py
import os
from openai import OpenAI

# 设置代理环境变量
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["DEEPSEEK_API_KEY"] = ""

# import google.generativeai as genai

# # 配置 API Key
# genai.configure(api_key="AIzaSyDF6S_r6DpWPki8rGvk7jVoHPS2wiqeyVA")

# # 选择模型（免费层可用）
# model = genai.GenerativeModel('gemini-2.0-flash')

# # 生成内容
# response = model.generate_content("Write a one-sentence bedtime story about a unicorn.")
# print(response.text)


# client = OpenAI(
#     api_key=os.environ.get('DEEPSEEK_API_KEY'),
#     base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)
client = OpenAI(api_key="sk-2b06a53548f3453b917655e4d06f3efb", base_url="https://api.deepseek.com")

# Turn 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

# Turn 2
messages.append({'role': 'assistant', 'content': content})
messages.append({'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"})
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)
print(response.choices[0].message.content)