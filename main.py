from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from typing import TypedDict

load_dotenv()

class AgentState(TypedDict):
    message:list[HumanMessage]
    result:str
    
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def process(state:AgentState)->AgentState:
    """Interacts with the llm"""
    response = llm.invoke(state["message"])
    state['result']= response.content
    return state
    
graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()
query = input("Ask any question with gemini :")
res = agent.invoke({"message":[HumanMessage(content=query)]})
print(res['result'])




    
