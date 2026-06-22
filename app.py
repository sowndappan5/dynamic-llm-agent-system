import os
import sys
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Update python execution path natively so dynamic file loading works cleanly
sys.path.append(os.path.abspath("."))

import tool_generator
import graph
import benchmark_logger

app = FastAPI(title="AdaptBot")

templates = Jinja2Templates(directory="templates")

# Re-use the LLM setup for intent classification
llm = ChatOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-20b",
)

class chat_output_schema(BaseModel):
    intent: Literal["BUILD", "CHAT"] = Field(description="The intent of the user")

class ChatRequest(BaseModel):
    message: str

def classify_intent(user_input: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", 'You are a router. Classify the user\'s input into one of two exact categories:\n"BUILD": The user is asking to create, build, make, modify, add, or delete a TOOL or CAPABILITY.\n"CHAT": The user is asking a normal question to be answered by the current capabilities.\n\nOutput ONLY the word BUILD or CHAT.'),
        ("human", "Input: {user_input}")
    ])
    chat_schema = llm.with_structured_output(chat_output_schema)
    result = (prompt | chat_schema).invoke({"user_input": user_input})
    return result.intent

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/chat")
async def chat(payload: ChatRequest):
    start_time = time.time()
    user_input = payload.message.strip()
    if not user_input:
        return {"tool_build": False, "response": "Empty message."}
        
    intent = "UNKNOWN"
    category = "N/A"
    tool_gen_time = None
    param_extract = "N/A"
    tool_exec = "N/A"
    err_msg = ""
    
    try:
        intent = classify_intent(user_input)
        if intent == "BUILD":
            build_start = time.time()
            # Pipe into the Auto-Builder Architecture
            b_response = tool_generator.tool_graph.invoke({"user_request": user_input})
            tool_gen_time = time.time() - build_start
            
            tool_name = b_response.get("tool_name", "Unknown")
            inner_intent = b_response.get("intent", intent)
            status = b_response.get("status", "pending")
            category = f"build_{inner_intent.lower()}"
            
            action_word = "Built"
            if inner_intent == "MODIFY": action_word = "Modified"
            elif inner_intent == "DELETE": action_word = "Deleted"
            
            msg = f"<strong>Task: {inner_intent}</strong><br/>{action_word}: <code>tools/{tool_name}.py</code><br/>Status: <em>{status}</em>"
            
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
            
            return {
                "tool_build": True, 
                "response": msg, 
                "path": "tool_generator"
            }
        else:
            # Query the highly-dynamic permanent graph node
            g_response = graph.app.invoke({"question": user_input, "category": "", "output": ""})
            path = g_response.get("category", "general")
            ans = g_response.get("output", "I encounter an error processing this.")
            category = path
            
            param_extract = g_response.get("parameter_extraction_success", "N/A")
            tool_exec = g_response.get("tool_execution_success", "N/A")
            err_msg = g_response.get("error_message", "")
            
            if path == "need_tool":
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
                return {
                    "tool_build": False,
                    "need_tool": True,
                    "response": "To answer this question I need to create a tool, please press the button if you like to create a tool.",
                    "path": "need_tool",
                    "original_question": user_input
                }
            
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
            
            return {
                "tool_build": False, 
                "response": ans, 
                "path": path
            }
            
    except Exception as e:
        end_time = time.time()
        err_str = str(e)
        benchmark_logger.log_benchmark(
            query=user_input,
            intent=intent,
            category=category,
            tool_generation_time=tool_gen_time,
            parameter_extraction_success=param_extract,
            tool_execution_success=tool_exec,
            end_to_end_latency=end_time - start_time,
            error_message=err_str
        )
        return {"tool_build": False, "response": f"Server Crash: {err_str}", "path": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
