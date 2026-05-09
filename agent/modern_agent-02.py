# # In[]

# from dotenv import load_dotenv
# from ast import literal_eval

# assert load_dotenv()

# from langchain_community.tools.tavily_search import TavilySearchResults

# tools  = [TavilySearchResults(max_results = 3)]

# # In[]
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# endpoint = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-72B-Instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
# )

# # 第二步：将定义好的 endpoint 作为 `llm` 参数传给 ChatHuggingFace
# llm = ChatHuggingFace(llm=endpoint)

# # In[]

# # 定义状态

# import operator
# from typing import Annotated,List,Tuple
# from typing_extensions import TypedDict

# class PlanExecute(TypedDict):
#     input:str
#     plan:List[str]
#     past_steps: Annotated[List[Tuple],operator.add]
#     response:str

# # planner
# from pydantic import BaseModel,Field


# class Plan(BaseModel):
#     """Plan to follow in future"""
#     steps:List[str] = Field(
#         description = "different steps to follow, should be in sorted order"
#     )


# from langchain_core.prompts import ChatPromptTemplate

# planner_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """For the given objective, come up with simple step by step plan.\
#             This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superflous steps.\
#             The result of final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",   
#         ),
#         ("placeholder","{messages}")
#     ]
# )

# planner = planner_prompt | llm.with_structured_output(Plan)
# planner.invoke(
#     {
#         "messages":[
#             ("user","What is the hometown of the current Australia Open men's singles winner?")
#         ]
#     }
# )
# # %%
# In[]

from dotenv import load_dotenv
from ast import literal_eval

assert load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults

tools  = [TavilySearchResults(max_results = 3)]

# In[]
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)

# 第二步：将定义好的 endpoint 作为 `llm` 参数传给 ChatHuggingFace
llm = ChatHuggingFace(llm=endpoint)
prompt = "You are a helpful assisstant."
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(llm,tools,prompt=prompt)
# In[]

# 定义状态

import operator
from typing import Annotated,List,Tuple
from typing_extensions import TypedDict

class PlanExecute(TypedDict):
    input:str
    plan:List[str]
    past_steps: Annotated[List[Tuple],operator.add]
    response:str

# planner
from pydantic import BaseModel,Field


class Plan(BaseModel):
    """Plan to follow in future"""
    steps:List[str] = Field(
        description = "different steps to follow, should be in sorted order"
    )


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser # <--- 引入 Parser

# 实例化 Pydantic 解析器
parser = PydanticOutputParser(pydantic_object=Plan)

# 修改 Prompt，在 System 提示词中加入 format_instructions 占位符
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with simple step by step plan.\
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superflous steps.\
            The result of final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.\n\n{format_instructions}""",   
        ),
        ("placeholder","{messages}")
    ]
)

# 将 Prompt, LLM 和 Parser 组合成 Chain (不再使用 llm.with_structured_output)
planner = planner_prompt | llm | parser

# 调用时，将 format_instructions 传入 prompt
result = planner.invoke(
    {
        "messages":[
            ("user","What is the hometown of the current Australia Open men's singles winner?")
        ],
        "format_instructions": parser.get_format_instructions() # <--- 传入解析器的指令
    }
)

print(result)


# In[]

# replan
from typing import Union
class Response(BaseModel):
    """response to user."""
    response:str

class Act(BaseModel):
    """Action to perform."""
    action:Union[Response,Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response."
    )
replanner_prompt = ChatPromptTemplate.from_template(
    """
        For the given objective, come up with simple step by step plan.\
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superflous steps.\
        The result of final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        Your objective was this:
        {input}
        
        Your original plan was this:
        {plan}

        You have currently done the following steps:
        {past_steps}

        Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that.
        Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
        {format_instructions}
        CRITICAL INSTRUCTION: You MUST output ONLY valid JSON. 
        Even if you have the final answer and want to respond to the user, you MUST wrap your response inside the JSON structure specified above. 
        Do NOT output any conversational text outside of the JSON block.
"""
)

replanner_parser = PydanticOutputParser(pydantic_object=Act)
replanner = replanner_prompt | llm | replanner_parser


# In[]
from typing import Literal
from langgraph.graph import END
async def execute_step(state:PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}"for i,step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages":[("user", task_formatted)]}
    )
    return {"past_steps":[(task,agent_response["messages"][-1].content),]}

# async def plan_step(state:PlanExecute):
#     plan = await planner.ainvoke({"messages":["user",state["input"]]})
#     return {"plan":plan.steps}
async def plan_step(state:PlanExecute):
    plan = await planner.ainvoke(
        {
            "messages": [("user", state["input"])], # 注意：这里用 tuple 包裹
            "format_instructions": parser.get_format_instructions() # <--- 必须加上这一行
        }
    )
    return {"plan": plan.steps}


# async def replan_step(state:PlanExecute):
#     output = await replanner.ainvoke(state)
#     if isinstance(output.action,Response):
#         return {"response":output.action.response}
#     else:
#         return {"plan":output.action.steps}
async def replan_step(state:PlanExecute):
    # 将 state 字典和 format_instructions 合并传给 replanner
    invoke_input = {
        **state,
        "format_instructions": replanner_parser.get_format_instructions()
    }
    
    output = await replanner.ainvoke(invoke_input)
    
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}
    
def should_end(state:PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


# In[]
from langgraph.graph import StateGraph, START
workflow = StateGraph(PlanExecute)

workflow.add_node("planner",plan_step)
workflow.add_node("agent",execute_step)
workflow.add_node("replan",replan_step)
workflow.add_edge(START,"planner")
workflow.add_edge("planner","agent")
workflow.add_edge("agent","replan")
workflow.add_conditional_edges("replan",
                               should_end,
                               ["agent",END])

                  
app = workflow.compile()

# In[]

config = {"recursion_limit":50}
inputs = {"input":"what is the hometown of the mens 2024 Australia open winner?"}
async for event in app.astream(inputs,config = config):
    for k,v in event.items():
        if k !="__end__":
            print(k,v)



# %%
