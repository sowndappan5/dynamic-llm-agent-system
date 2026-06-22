import os
import sys
import json
import importlib
from typing import TypedDict, Optional
from pydantic import BaseModel, Field, create_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

sys.path.append(os.path.abspath("."))
REGISTRY_FILE = "tools_registry.json"

llm = ChatOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-20b",
)

class GraphState(TypedDict):
    question: str
    category: Optional[str]
    output: Optional[str]
    parameter_extraction_success: Optional[str]
    tool_execution_success: Optional[str]
    error_message: Optional[str]

# ----------------- Nodes -----------------
def classifier_node(state: GraphState) -> GraphState:
    """Reads the JSON registry live on every single user message!"""
    tools = []
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            tools = json.load(f)
            
    # Dynamically build the LLM's classification schema based on the live JSON
    DynamicCategorySchema = create_model(
        'llm_schema', 
        category=(str, Field(description=f"Must pick EXACTLY one category from this list: general, need_tool, {', '.join(tools)}. Choose 'need_tool' if the question requires a specific tool, function, or real-time data (like weather, calculator, search, clocks, API actions) that is NOT present in the existing tools list."))
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a router. Given a question, classify if it should be answered generally ('general'), requires a tool that doesn't exist yet ('need_tool'), or matches one of the existing tools: {', '.join(tools)}. Output exactly the matching category name."),
        ("human", "{question}"),
    ])
    category_res = (prompt | llm.with_structured_output(DynamicCategorySchema)).invoke({"question": state["question"]})
    return {"question": state["question"], "category": category_res.category}

def general_node(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer generic questions directly without tools."),
        ("human", "{question}")
    ])
    response = (prompt | llm).invoke({"question": state["question"]}).content
    return {"question": state["question"], "output": response}

def execute_tool_node(state: GraphState) -> GraphState:
    """The Single Universal Tool Node! Injects any Python file dynamically and runs it."""
    tool_name = state.get("category")
    param_extract = "N/A"
    tool_exec = "False"
    err_msg = ""
    try:
        # Dynamically import ONLY the exact tool file requested!
        module_name = f"tools.{tool_name}"
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
            
        if hasattr(module, 'execute_tool'):
            # The Universal Extractor
            if hasattr(module, 'ToolInputSchema'):
                schema = module.ToolInputSchema
                if len(schema.model_fields) == 0:
                    param_extract = "True"
                    try:
                        res = module.execute_tool(schema())
                        tool_exec = "True"
                    except Exception as ex:
                        res = f"Tool crash during execution: {ex}"
                        err_msg = str(ex)
                else:
                    try:
                        extractor = llm.with_structured_output(schema)
                        clean_params = extractor.invoke(f"Extract precisely exactly the required schema parameters for this tool from this conversational query:\n\nQuery: '{state['question']}'")
                        param_extract = "True"
                    except Exception as ex:
                        param_extract = "False"
                        raise Exception(f"Parameter extraction failed: {ex}")
                        
                    try:
                        res = module.execute_tool(clean_params)
                        tool_exec = "True"
                    except Exception as ex:
                        res = f"Tool crash during execution: {ex}"
                        err_msg = str(ex)
            else:
                try:
                    res = module.execute_tool(state["question"])
                    tool_exec = "True"
                except Exception as ex:
                    res = f"Tool crash during execution: {ex}"
                    err_msg = str(ex)
            output = str(res)
        else:
            output = f"Error: Tool script '{tool_name}' failed to define execute_tool()"
            err_msg = output
            
    except Exception as e:
        output = f"Tool crash: {e}"
        err_msg = str(e)
        if param_extract == "N/A":
            param_extract = "False"
        
    return {
        "question": state["question"],
        "category": tool_name,
        "output": output,
        "parameter_extraction_success": param_extract,
        "tool_execution_success": tool_exec,
        "error_message": err_msg
    }

def synthesizer_node(state: GraphState) -> GraphState:
    res = state.get("output", "")
    q = state.get("question", "")
    
    if str(res).startswith("Error:") or str(res).startswith("Tool crash:"):
        return {"question": q, "output": res}
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly AI Agent. Rewrite the tool's raw data into a polite, conversational sentence."),
        ("human", "User Question: {question}\nRaw Tool Data: {output}")
    ])
    final_answer = (prompt | llm).invoke({"question": q, "output": res}).content
    return {"question": q, "output": final_answer}

# ----------------- Static Graph Wiring Strategy -----------------
# This graph structure NEVER changes! It never needs to be recompiled!

workflow = StateGraph(GraphState)
workflow.add_node("classifier", classifier_node)
workflow.add_node("general", general_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "classifier")

# Route based on the dynamic classifier result
def route_classifier(state: GraphState) -> str:
    category = state.get("category", "general")
    if category == "general":
        return "general"
    elif category == "need_tool":
        return "end"
    return "execute_tool"

workflow.add_conditional_edges(
    "classifier",
    route_classifier,
    {
        "general": "general",
        "execute_tool": "execute_tool",
        "end": END
    }
)

# Wiring to finish
workflow.add_edge("general", END)
workflow.add_edge("execute_tool", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compile exactly once!
app = workflow.compile()