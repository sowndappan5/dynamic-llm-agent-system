import os
import importlib
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import Literal

# Import our new graph-based tool generator
import tool_generator1

# Import the auto-routing graph engine
import graph

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-20b",
)

class chat_output_schema(BaseModel):
    intent: Literal["BUILD", "CHAT"] = Field(description="The intent of the user")

def classify_intent(user_input: str) -> str:
    """Uses LLM to classify if the user wants to manage tools or ask a normal question."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", 'You are a router. Classify the user\'s input into one of two exact categories:\n"BUILD": The user is asking to create, build, make, modify, add, or delete a TOOL or CAPABILITY.\n"CHAT": The user is asking a normal question to be answered by the current capabilities.\n\nOutput ONLY the word BUILD or CHAT.'),
        ("human", "Input: {user_input}")
    ])
    
    chat_schema = llm.with_structured_output(chat_output_schema)
    chain = prompt | chat_schema
    result = chain.invoke({"user_input": user_input})
    return result.intent
    

def run_chat():
    print("=" * 60)
    print("Meta-Agent Interface Initialized!")
    print("Ask questions, or tell me to build/modify tools for you.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            if user_input.lower() in ('quit', 'exit'): break
            
            intent = classify_intent(user_input)

            if intent == 'BUILD':
                print("\n[Meta-Agent] Detected Tool Building Request. Initiating Build Pipeline...")
                
                # Step 1: Tool Generator creates/upkeeps tool files
                response = tool_generator1.tool_graph.invoke({"user_request": user_input})
                print(f"[Meta-Agent] Architecture result: {response}")
                print("[Meta-Agent] The universal node will execute this new tool automatically!")
                
            else:
                print("\n[Meta-Agent] Querying Graph...")
                # Use dynamically imported graph
                result = graph.app.invoke({"question": user_input, "category": "", "output": ""})
                print(f"[Node Path Traveled -> {result.get('category', 'unknown')}]")
                print(f"Agent: {result.get('output', 'No answer')}")
                continue
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_chat() "__main__":
    run_chat()