from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
import os

load_dotenv()

class AgentState(TypedDict):
    messages: list[HumanMessage | AIMessage]

llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL"))

def process(state:AgentState)->AgentState:
    """This node will process the request"""
    response = llm.invoke(state["messages"])
    state['messages'].append(AIMessage(response.content))
    return state
    
graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)

agent = graph.compile()
conversation_history = []


print("_"*50)
print("_"*50)
print("="*20 , "Hello I am your AI","="*20)
prompt = input("YOU: ")
while prompt.lower() != "exit":
    conversation_history.append(HumanMessage(content=prompt))
    res = agent.invoke({"messages":conversation_history})
    print("AI: ",res['messages'][-1].content)
    print("="*50)
    conversation_history = res['messages']
    prompt = input("YOU :")


with open("memory.log","w") as file:
    file.write(f"{'='*30}AI chat history{'='*30}")
    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write("\nYou: "+message.content)
        elif isinstance(message,AIMessage):
            file.write("\nAI: "+message.content)
    print("Conversation Stored")

        