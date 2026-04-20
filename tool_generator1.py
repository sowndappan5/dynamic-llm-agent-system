import os
import glob
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

from langgraph.graph import StateGraph, START, END

load_dotenv(find_dotenv())

# Define and create the tools directory
TOOLS_DIR = "tools"
os.makedirs(TOOLS_DIR, exist_ok=True)

llm = ChatOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-20b",
)

import json

REGISTRY_FILE = "tools_registry.json"

def get_existing_tools() -> list[str]:
    """Returns a list of tool names from the registry file."""
    if not os.path.exists(REGISTRY_FILE):
        return []
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)

def write_registry(tools: list[str]):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(tools, f)

def read_tool(tool_name: str) -> str:
    """Reads the current content of a specific tool file."""
    filepath = os.path.join(TOOLS_DIR, f"{tool_name}.py")
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_tool(tool_name: str, code: str):
    """Writes code to a specific tool file."""
    if "\\n" in code and "\n" not in code:
        code = code.replace("\\n", "\n").replace("\\t", "    ").replace('\\"', '"')
    
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    filepath = os.path.join(TOOLS_DIR, f"{tool_name}.py")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

class graph_schema(TypedDict):
    user_request: str
    tool_name: str
    intent: Literal["CREATE", "MODIFY", "DELETE"]
    status: Literal["pending", "success", "error"]

class decision_output_schema(BaseModel):
    tool_name: str = Field(description="The precise snake_case name of the tool")
    intent: Literal["CREATE", "MODIFY", "DELETE"] = Field(description="The intent")
    user_request: str = Field(description="A short summary of the user's request")

class tool_code_schema(BaseModel):
    tool_code: str = Field(description="The FULL python code for the tool")

def decision_tool(state: graph_schema) -> graph_schema:
    existing_tools = get_existing_tools()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Decide if a tool should be CREATE, MODIFY, or DELETE. \nCurrently existing tools: {existing_tools}\nExtract the tool name (snake_case) and summarize the request."),
        ("human", "{user_request}"),
    ])
    decision_schema = llm.with_structured_output(decision_output_schema)
    chain = prompt | decision_schema
    result = chain.invoke({
        'user_request': state['user_request'],
        'existing_tools': existing_tools
    })
    return {
        'tool_name': result.tool_name,
        'intent': result.intent,
        'user_request': result.user_request
    }

def condition(state: graph_schema) -> str:
    intent = state.get("intent")
    if intent == "CREATE": return "create_tool"
    elif intent == "MODIFY": return "modify_tool"
    elif intent == "DELETE": return "delete_tool"
    return "end"

def create_tool(state: graph_schema) -> graph_schema:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a tool creator agent. Output the FULL Python code for the new tool. CRITICAL RULE: High-level files MUST expose two things:\n1. A Pydantic `class ToolInputSchema(BaseModel):` outlining exactly what structured properties the tool needs (e.g. valid city strings, exact equations, etc.).\n2. An `def execute_tool(params: ToolInputSchema) -> str:` function as the entrypoint hook. The Universal Extractor handles conversation parameters for you. Do not use Markdown blocks!"),
        ("human", "Create a tool named {tool_name} based on this request: {user_request}"),
    ])
    create_schema = llm.with_structured_output(tool_code_schema)
    chain = prompt | create_schema
    result = chain.invoke({'tool_name': state['tool_name'], 'user_request': state['user_request']})
    write_tool(state['tool_name'], result.tool_code)
    
    tools = get_existing_tools()
    if state['tool_name'] not in tools:
        tools.append(state['tool_name'])
        write_registry(tools)
        
    return {'status': 'success'}

def modify_tool(state: graph_schema) -> graph_schema:
    current_code = read_tool(state['tool_name'])
    if not current_code: return {'status': 'error'}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a tool modifier agent. Output the ENTIRE updated Python code. CRITICAL RULE: Always ensure the tool continues to expose both a Pydantic `class ToolInputSchema(BaseModel):` and an `def execute_tool(params: ToolInputSchema) -> str:` interface. No markdown blocks."),
        ("human", "Current code for {tool_name}:\n{current_code}\n\nModify the tool based on this request: {user_request}"),
    ])
    modify_schema = llm.with_structured_output(tool_code_schema)
    chain = prompt | modify_schema
    result = chain.invoke({'current_code': current_code, 'tool_name': state['tool_name'], 'user_request': state['user_request']})
    write_tool(state['tool_name'], result.tool_code)
    return {'status': 'success'}

def delete_tool(state: graph_schema) -> graph_schema:
    filepath = os.path.join(TOOLS_DIR, f"{state['tool_name']}.py")
    if os.path.exists(filepath):
        os.remove(filepath)
        tools = get_existing_tools()
        if state['tool_name'] in tools:
            tools.remove(state['tool_name'])
            write_registry(tools)
        return {'status': 'success'}
    return {'status': 'error'}

tool_graph = StateGraph(graph_schema)
tool_graph.add_node("decision_tool", decision_tool)
tool_graph.add_node("create_tool", create_tool)
tool_graph.add_node("modify_tool", modify_tool)
tool_graph.add_node("delete_tool", delete_tool)

tool_graph.add_edge(START, "decision_tool")
tool_graph.add_conditional_edges("decision_tool", condition, {"create_tool": "create_tool", "modify_tool": "modify_tool", "delete_tool": "delete_tool", "end": END})
tool_graph.add_edge("create_tool", END)
tool_graph.add_edge("modify_tool", END)
tool_graph.add_edge("delete_tool", END)

tool_graph = tool_graph.compile()