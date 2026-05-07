import os
import langchain

langchain.debug = True

# 1. 网络与镜像配置 (确保能秒下那几 MB 的配置文件)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
proxy_url = "http://127.0.0.1:7890"
os.environ["http_proxy"] = proxy_url
os.environ["https_proxy"] = proxy_url
os.environ["HTTP_PROXY"] = proxy_url
os.environ["HTTPS_PROXY"] = proxy_url

# 2. 🚨 必须填入你的真实 API KEY
HF_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 3. 关键修改：显式使用 Endpoint (云端API模式)，绝对不下载模型权重
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct", # 换成当前免费节点支持的顶级模型
    task="text-generation",
    temperature=0.7,
    max_new_tokens=1024,
    huggingfacehub_api_token=HF_TOKEN 
)

# 将底层的云端 LLM 包装成支持对话格式的 Chat 模型
chat_model = ChatHuggingFace(llm=llm)

# 4. 创建 Agent
agent = create_agent(
    model=chat_model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# 5. 执行
print("开始请求云端 API...")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)

print("\n=== 最终结果 ===")
print(result["messages"][-1].content)