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
        category=(str, Field(description=f"Must pick EXACTLY one category from this list: general, {', '.join(tools)}"))
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a router. Given a question, output the category name exactly as one of the following: general, {', '.join(tools)}."),
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
                    res = module.execute_tool(schema())
                else:
                    extractor = llm.with_structured_output(schema)
                    clean_params = extractor.invoke(f"Extract precisely exactly the required schema parameters for this tool from this conversational query:\n\nQuery: '{state['question']}'")
                    res = module.execute_tool(clean_params)
            else:
                res = module.execute_tool(state["question"])
            output = str(res)
        else:
            output = f"Error: Tool script '{tool_name}' failed to define execute_tool()"
            
    except Exception as e:
        output = f"Tool crash: {e}"
        
    return {"question": state["question"], "category": tool_name, "output": output}

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
    return "execute_tool"

workflow.add_conditional_edges("classifier", route_classifier)

# Wiring to finish
workflow.add_edge("general", END)
workflow.add_edge("execute_tool", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compile exactly once!
app = workflow.compile()ral":
        return "general"
    return "execute_tool"

workflow.add_conditional_edges("classifier", route_classifier)

# Wiring to finish
workflow.add_edge("general", END)
workflow.add_edge("execute_tool", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compile exactly once!
app = workflow.compile()