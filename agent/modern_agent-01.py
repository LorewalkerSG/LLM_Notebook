# In[]

from dotenv import load_dotenv
from ast import literal_eval

assert load_dotenv()


from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results = 3)]


# In[]
from langchain_huggingface import  HuggingFaceEndpoint,ChatHuggingFace
from langchain_classic import hub
# from langchain_classic.agents import AgentExecutor, create_react_agent
from langsmith import Client

# 初始化 LangSmith 客户端
client = Client()

# 直接使用底层的 client，并传入安全授权参数
prompt = client.pull_prompt(
    "hwchase17/react", 
    dangerously_pull_public_prompt=True
)
print(prompt.template)



# In[]

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)

# 第二步：将定义好的 endpoint 作为 `llm` 参数传给 ChatHuggingFace
llm = ChatHuggingFace(llm=endpoint)
# agent = create_react_agent(llm,tools,prompt)
# agent_executor = AgentExecutor(agent=agent,tools = tools)
# agent_executor.invoke({'input':'what is the hometown of the current Australia open winner?'})



# In[]
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model = llm,tools = tools)
message = agent.invoke({"messages":[{'role':'user','content':'what is the hometown of the current Australia open winner?'}]})

from rich.pretty import pprint
 
pprint(message)
# %%
