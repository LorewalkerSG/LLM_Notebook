# In[]
from dotenv import load_dotenv
load_dotenv()



from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel
)
import os
os.environ["LANGCHAIN_PROJECT"] = "lcel_test"


from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
output_parser = StrOutputParser()
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# 第一步：定义基础的 Hugging Face 模型端点
# 注意这里使用的是 repo_id，并且你需要配置好 Hugging Face Token
endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)

# 第二步：将定义好的 endpoint 作为 `llm` 参数传给 ChatHuggingFace
llm = ChatHuggingFace(llm=endpoint)
chain = ({"topic":RunnablePassthrough()}
         |prompt
         |llm
         |output_parser)

chain.invoke("ice cream")


# RunnablePassthrough : 允许将输入数据直接传递而不做任何修改，通常与RunnablePassthrough 一起使用
# 占位符，在需要时填入数据 
# In[]
chain = RunnablePassthrough()| RunnablePassthrough() |RunnablePassthrough()

chain.invoke("hello")
# In[]
chain = RunnablePassthrough() | RunnableLambda(lambda x : x.upper())
chain.invoke("hello")



os.environ["LANGCHAIN_PROJECT"] = "JSON_TEST"
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts.chat  import SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatHuggingFace(llm = endpoint,model_kwargs={'response_format':{"type":"json_object"}})
json_parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system",'''I want you to extract the person name, age and a description from the following text. Here is the JSON object, output:
     {{
     "name":string,
     "age":int,
     "description":string
     }}
     '''),
     ("human","{input}")
])

chain = ({"input":RunnablePassthrough()}
         | prompt
         | llm
         | json_parser
         )
print(prompt[0])
print(prompt[1])

# In[]
result = chain.invoke("John is 20 years old. He is a student at the University of California, Berkeley. He is a vert smart student.")

# In[]
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(

)
vectorstore = FAISS.from_texts(texts = ["Cats love tuna"],
                               embedding = embeddings,
                               )

retriever = vectorstore.as_retriever()
retriever.invoke("What do cats like to eat?")


template = """Answer the question based only on the following context:
{context}

Question:{question}
"""

prompt = ChatPromptTemplate.from_template(template = template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context":retriever|format_docs,"question":RunnablePassthrough()}
    | prompt
    | ChatHuggingFace(llm=endpoint)
    | StrOutputParser()
)
rag_chain.invoke("What do cats like to eat?")


# In[]
# TOOLS

import numpy as np
from langchain_core.tools import Tool
from langchain_core.tools import tool
@tool
def add(num1:float,num2:float)->float:
    """
    add two numbers
    """
    return num1 + num2

@tool
def subtract(num1:float,num2:float)->float:
    """
    Subtract two numbers
    """
    return num1-num2

@tool
def multiply(num1:float,num2:float)->float:
    """
    Multiply two numbers
    """
    return num1 * num2

@tool
def divide(numerator:float, denominator:float)->float:
    """
    divides the numerator by the denominator.
    """
    return numerator/denominator

@tool
def power(base:float,exponent:float)->float:
    """
    Take the base to the exponent power, base^exponent
    """
    return base ** exponent

@tool
def exp(x):
    """
    Calculate the natural exponential $e^x$
    """
    return np.exp(x)

tools = [add,subtract,multiply,divide,power,exp]
# from langchain.agents import create_agent
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,

)

llm = ChatHuggingFace(llm=endpoint,temperature = 0,streaming = True)

# In[]
system_template="""
You are a helpful math assistant that uses calculation functions to solve complex math problems step by step.
"""
human_template = "{input}"
prompt = ChatPromptTemplate.from_messages(
    [SystemMessagePromptTemplate.from_template(system_template),
     MessagesPlaceholder(variable_name="chat_history",optional = True),
     HumanMessagePromptTemplate.from_template(input_variables = ["input"],template = human_template),
     MessagesPlaceholder(variable_name="agent_scratchpad")]
)
agent = create_tool_calling_agent(llm = llm,tools = tools,prompt= prompt)
# response = agent.invoke({"message":[{"role":"user","content":"What is the result of directive of sigmoid(2.5)?"}]})

# print(response)

# from langchain.agents import AgentExecutor

executor = AgentExecutor(agent = agent, tools = tools , verbose = True)
executor.invoke({"input":"What is the result of directive of sigmoid(2.5)? Note that I want to know the directive of sigmoid(2.5) rather sigmoid(2.5)."})
# %%
