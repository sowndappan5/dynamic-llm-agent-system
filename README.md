# Meta Agent

Meta Agent is a dynamic AI agent system that can answer normal questions, route tool-based requests through a static LangGraph workflow, and generate or modify its own Python tools at runtime.

The core idea is simple: the graph stays permanent, but the toolset does not. New tools are written as individual Python files in `tools/`, registered in `tools_registry.json`, and loaded dynamically when the agent needs them.

## What It Does

- Serves a web chat interface with Flask
- Supports a CLI chat loop for local testing
- Uses LangGraph for static orchestration
- Uses an OpenAI-compatible model endpoint via Groq
- Dynamically creates, updates, and deletes tool files
- Extracts structured parameters from natural language before tool execution

## Architecture Overview

This project follows the architecture described in [Architecture_Roadmap.md](./Architecture_Roadmap.md).

### 1. Intent split

User input is first classified into one of two paths:

- `BUILD`: create, modify, or delete a tool
- `CHAT`: answer with current capabilities

### 2. Tool management graph

`tool_generator1.py` handles dynamic tool lifecycle management:

- decides whether the request is `CREATE`, `MODIFY`, or `DELETE`
- writes each tool as an isolated file inside `tools/`
- updates `tools_registry.json`

This avoids maintaining one large monolithic tool file.

### 3. Static execution graph

`graph.py` defines a permanently compiled LangGraph with four stable nodes:

- `classifier_node`
- `general_node`
- `execute_tool_node`
- `synthesizer_node`

Instead of recompiling the graph whenever a new tool is added, the graph reads the registry at runtime and dynamically imports the requested tool module.

### 4. Universal structured extraction

Each generated tool is expected to expose:

```python
class ToolInputSchema(BaseModel):
    ...

def execute_tool(params: ToolInputSchema) -> str:
    ...
```

Before execution, the system uses the tool's `ToolInputSchema` to extract clean parameters from conversational user input. This keeps tools simple and avoids making every tool responsible for parsing raw prompts.

## Repository Structure

```text
Meta Agent/
|- app.py                    # Flask web server
|- bot.py                    # CLI chat interface
|- graph.py                  # Static LangGraph execution engine
|- tool_generator1.py        # Dynamic tool creation/modification/deletion graph
|- tools/
|  |- clock.py               # Example tool
|  |- weather.py             # Example tool
|- templates/
|  |- index.html             # Chat UI
|- static/
|  |- style.css              # UI styling
|- tools_registry.json       # Runtime tool registry
|- Architecture_Roadmap.md   # Architecture memory log and design rationale
```

## Current Flow

1. User sends a message through the web app or CLI.
2. The system classifies the message as `BUILD` or `CHAT`.
3. If it is a build request, the tool generation graph creates, modifies, or deletes a tool file.
4. If it is a normal query, the static LangGraph routes either to a general response or to a dynamically imported tool.
5. Tool output is rewritten into a conversational response before being returned.

## Tech Stack

- Python
- Flask
- LangGraph
- LangChain
- Pydantic
- `langchain-openai`
- `python-dotenv`
- OpenAI-compatible Groq endpoint using `openai/gpt-oss-20b`

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Meta Agent"
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If you do not already have a `requirements.txt`, install the core packages manually:

```bash
pip install flask langgraph langchain langchain-openai pydantic python-dotenv requests
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Running the Project

### Run the web app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

### Run the CLI interface

```bash
python bot.py
```

## Example Prompts

Normal chat:

- `What time is it?`
- `What's the weather in Chennai?`

Tool-building requests:

- `Build a random number generator tool`
- `Modify the weather tool to accept country codes`
- `Delete the clock tool`

## Why This Project Is Interesting

Most agent systems either:

- hardcode tools ahead of time, or
- require app restarts and graph rewiring when capabilities change

This project takes a different approach:

- the graph stays fixed
- the tool inventory stays dynamic
- the agent can expand its own capabilities through file-based tool generation

That makes it a practical prototype for self-extending agent systems built on top of static orchestration frameworks.

## Notes

- The dynamic tools are stored as Python files, so generated code should be reviewed before production use.
- `tools_registry.json` is the live source of truth for available tools.
- The architecture roadmap includes future ideas for a true JIT execution engine beyond LangGraph's compile-time model.

## License

Add your preferred license here before publishing to GitHub.
