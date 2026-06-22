import os
import importlib
import sys
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import Literal

# Import our new graph-based tool generator
import tool_generator

# Import the auto-routing graph engine
import graph
import benchmark_logger

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
    print("AdaptBot Interface Initialized!")
    print("Ask questions, or tell me to build/modify tools for you.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        user_input = ""
        intent = "UNKNOWN"
        category = "N/A"
        tool_gen_time = None
        param_extract = "N/A"
        tool_exec = "N/A"
        start_time = time.time()
        
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            if user_input.lower() in ('quit', 'exit'): break
            
            intent = classify_intent(user_input)

            if intent == 'BUILD':
                print("\n[AdaptBot] Detected Tool Building Request. Initiating Build Pipeline...")
                build_start = time.time()
                response = tool_generator.tool_graph.invoke({"user_request": user_input})
                tool_gen_time = time.time() - build_start
                
                tool_name = response.get("tool_name", "Unknown")
                inner_intent = response.get("intent", intent)
                status = response.get("status", "pending")
                category = f"build_{inner_intent.lower()}"
                
                print(f"[AdaptBot] Architecture result: {response}")
                print("[AdaptBot] The universal node will execute this new tool automatically!")
                
                end_time = time.time()
                benchmark_logger.log_benchmark(
                    query=user_input,
                    intent=intent,
                    category=category,
                    tool_generation_time=tool_gen_time,
                    parameter_extraction_success="N/A",
                    tool_execution_success="N/A",
                    end_to_end_latency=end_time - start_time,
                    error_message="" if status == "success" else f"Build status: {status}"
                )
            else:
                print("\n[AdaptBot] Querying Graph...")
                result = graph.app.invoke({"question": user_input, "category": "", "output": ""})
                
                path = result.get("category", "general")
                category = path
                param_extract = result.get("parameter_extraction_success", "N/A")
                tool_exec = result.get("tool_execution_success", "N/A")
                err_msg = result.get("error_message", "")
                
                print(f"[Node Path Traveled -> {path}]")
                print(f"Agent: {result.get('output', 'No answer')}")
                
                end_time = time.time()
                benchmark_logger.log_benchmark(
                    query=user_input,
                    intent=intent,
                    category=category,
                    tool_generation_time=None,
                    parameter_extraction_success=param_extract,
                    tool_execution_success=tool_exec,
                    end_to_end_latency=end_time - start_time,
                    error_message=err_msg
                )
                
        except Exception as e:
            end_time = time.time()
            print(f"Error: {e}")
            if user_input:
                benchmark_logger.log_benchmark(
                    query=user_input,
                    intent=intent,
                    category=category,
                    tool_generation_time=tool_gen_time,
                    parameter_extraction_success=param_extract,
                    tool_execution_success=tool_exec,
                    end_to_end_latency=end_time - start_time,
                    error_message=str(e)
                )

if __name__ == "__main__":
    run_chat()