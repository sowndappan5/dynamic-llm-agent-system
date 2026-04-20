import os
import sys
from flask import Flask, render_template, request, jsonify

# Update python execution path natively so dynamic file loading works cleanly
sys.path.append(os.path.abspath("."))

import tool_generator1
import graph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

app = Flask(__name__)

# Re-use the LLM setup for intent classification
llm = ChatOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-20b",
)

class chat_output_schema(BaseModel):
    intent: Literal["BUILD", "CHAT"] = Field(description="The intent of the user")

def classify_intent(user_input: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", 'You are a router. Classify the user\'s input into one of two exact categories:\n"BUILD": The user is asking to create, build, make, modify, add, or delete a TOOL or CAPABILITY.\n"CHAT": The user is asking a normal question to be answered by the current capabilities.\n\nOutput ONLY the word BUILD or CHAT.'),
        ("human", "Input: {user_input}")
    ])
    chat_schema = llm.with_structured_output(chat_output_schema)
    result = (prompt | chat_schema).invoke({"user_input": user_input})
    return result.intent

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"tool_build": False, "response": "Empty message."})
        
    try:
        intent = classify_intent(user_input)
        if intent == "BUILD":
            # Pipe into the Auto-Builder Architecture
            b_response = tool_generator1.tool_graph.invoke({"user_request": user_input})
            
            # Formulate the frontend success message
            tool_name = b_response.get("tool_name", "Unknown")
            inner_intent = b_response.get("intent", intent)
            status = b_response.get("status", "pending")
            
            action_word = "Built"
            if inner_intent == "MODIFY": action_word = "Modified"
            elif inner_intent == "DELETE": action_word = "Deleted"
            
            msg = f"<strong>Task: {inner_intent}</strong><br/>{action_word}: <code>tools/{tool_name}.py</code><br/>Status: <em>{status}</em>"
            
            return jsonify({
                "tool_build": True, 
                "response": msg, 
                "path": "tool_generator"
            })
        else:
            # Query the highly-dynamic permanent graph node
            g_response = graph.app.invoke({"question": user_input, "category": "", "output": ""})
            path = g_response.get("category", "general")
            ans = g_response.get("output", "I encounter an error processing this.")
            
            return jsonify({
                "tool_build": False, 
                "response": ans, 
                "path": path
            })
            
    except Exception as e:
        return jsonify({"tool_build": False, "response": f"Server Crash: {str(e)}", "path": "error"})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
