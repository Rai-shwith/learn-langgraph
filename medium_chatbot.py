from langchain_core.messages import BaseMessage,AIMessage,HumanMessage,SystemMessage
from langgraph.graph import START,StateGraph,END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Sequence,Annotated
from langgraph.graph.message import add_messages
from google.api_core.exceptions import ResourceExhausted
import os

load_dotenv()

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    
@tool
def add(a:int,b:int)->int:
    """This function adds two numbers"""
    return a+b

@tool
def subtract(a:int,b:int)->int:
    """This function subtracts two numbers (a-b)"""
    return a-b

@tool
def multiply(a:int,b:int)->int:
    """This function multiplies two numbers (a*b)"""
    return a*b

@tool
def divide(a:int,b:int)->int:
    """This function divides two numbers (a/b)"""
    return a/b


tools = [add,subtract,multiply,divide]


llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL")).bind_tools(tools)


def model_call(state:AgentState)->AgentState:
    """This function calls the AI"""
    messages = state["messages"]
    if not messages or not isinstance(messages[0],SystemMessage):
        system_prompt = "You are my AI assistant, Your name is Astra."
        messages= [SystemMessage(content=system_prompt)] + messages
        
    
    response = llm.invoke(messages)
    return {"messages":[response]}


def should_continue(state:AgentState):
    """Decides should continue or not"""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
    
    
graph = StateGraph(AgentState)
graph.add_node("agent_node",model_call)

tool_node = ToolNode(tools)
graph.add_node("tools",tool_node)

graph.add_edge(START,"agent_node")
graph.add_conditional_edges("agent_node",should_continue,{
    "end":END,
    "continue":"tools"
})
graph.add_edge("tools","agent_node")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()
            
inputs:AgentState = {"messages":[HumanMessage(content="add 4 and 5 then multiply the result with 2 and divide the result by 6 and subtract 3 from this what is the answer?")]}
print_stream(app.stream(inputs,stream_mode="values"))

