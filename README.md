# Build an AI Agent from Scratch

Companion code repository for Manning Publications' [*Build an AI Agent from Scratch*](https://www.manning.com/books/build-an-ai-agent-from-scratch).

## Structure

```
scratch_agents/          # Final package (complete through CH10)
  types.py              # Message, ToolCall, ToolResult, Event, ContentItem
  context.py            # ExecutionContext, AgentResult, PendingToolCall, ToolConfirmation
  llm.py                # LlmRequest, LlmResponse, LlmClient
  agent.py              # Agent (ReAct loop)
  rag.py                # Embeddings, chunking, vector search
  callbacks.py          # approval_callback, search_compressor
  planning.py           # Task, create_tasks, reflection
  skills.py             # SkillInfo, discover_skills, generate_skills_prompt
  transfer.py           # create_transfer_tool
  remote.py             # RemoteAgent (A2A)
  a2a_server.py         # MathAgentExecutor
  tools/                # Tool modules
  memory/               # Session, long-term memory, context optimization
  workflows/            # Sequential, Parallel, Loop
  eval/                 # GAIA benchmark, evaluation prompts

notebooks/              # Chapter notebooks
  ch02/                 # LLM API Basics
  ch03/                 # Tools and Function Calling
  ch04/                 # ReAct Agent (+ chapter snapshot code)
  ch05/                 # RAG and File Tools (+ chapter snapshot code)
  ch06/                 # Memory Systems (+ chapter snapshot code)
  ch07/                 # Planning and Reflection
  ch08/                 # Code Execution (+ chapter snapshot code)
  ch09/                 # Multi-Agent Systems (+ chapter snapshot code)
  ch10/                 # Evaluation
```

## Setup

Use Python 3.13 or later. Start from the repository root. `uv sync` installs
`scratch_agents` as a package as well as its dependencies.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --locked

# Set up API keys in .env
cp .env.example .env
# Edit .env and add your API keys

# Launch Jupyter Lab
uv run jupyter lab
```

## API Keys

Create a `.env` file in the project root with the following keys:

```
OPENAI_API_KEY=sk-...          # Required for all chapters
GEMINI_API_KEY=...            # Required for CH05 and scratch_agents.rag embeddings
ANTHROPIC_API_KEY=sk-ant-...   # Required for CH02 Anthropic examples
TAVILY_API_KEY=tvly-...        # Required for CH03 web search
HF_TOKEN=hf_...                # Required for CH02 GAIA benchmark
E2B_API_KEY=e2b_...            # Required for CH08 code execution
```

`OPENAI_API_KEY` covers the OpenAI examples, not every cell in every chapter.
For **Restart Kernel and Run All**, prepare each chapter's prerequisites first:

| Chapters | Additional prerequisites |
|---|---|
| CH02 | Anthropic key; Hugging Face account with accepted GAIA access and `HF_TOKEN` |
| CH03–CH04 | Tavily key; Node.js/npm (`npx`) for the Tavily MCP server; CH04 also needs GAIA access |
| CH05 | Gemini key for embeddings; Tavily key and GAIA access/downloads for attachment exercises |
| CH06 | OpenAI key for model calls and ChromaDB embeddings |
| CH07 | Tavily key for search examples |
| CH08 | E2B key; Tavily key only for sandbox tools that use it |
| CH09 | E2B/Tavily keys when running the specialist agents that use those services |
| CH10 | OpenAI key |

Set keys in the repository's `.env`. Notebook setup finds this file from the
chapter directory. Select the project environment's Python kernel in Jupyter.
If a different environment is selected, install the dependencies in that kernel
or restart Jupyter with `uv run jupyter lab`.

These examples make real, potentially billable requests. CH02 includes a
100-request concurrency example and multi-model GAIA evaluation; reduce the
example counts when doing a quick live check. Model IDs are examples and require
access from your provider account.

CH05 creates `notebooks/ch05/gaia_workspace` and resets its contents for the
attachment exercise; do not keep personal files there. CH06 creates its own
throwaway deletion target, and CH08 uses the GAIA spreadsheet prepared in CH05.

## Chapters

| Chapter | Topic | Key Modules |
|---------|-------|-------------|
| CH02 | LLM API Basics | eval/gaia.py |
| CH03 | Tools and Function Calling | tools/helpers.py, tools/calculator.py, tools/search.py |
| CH04 | ReAct Agent | types.py, context.py, llm.py, agent.py, tools/base.py |
| CH05 | RAG and File Tools | rag.py, callbacks.py, tools/file_tools.py |
| CH06 | Memory Systems | memory/session.py, memory/long_term.py, memory/context_optimizer.py |
| CH07 | Planning and Reflection | planning.py |
| CH08 | Code Execution | tools/code_execution.py, skills.py |
| CH09 | Multi-Agent Systems | workflows/, transfer.py, tools/agent_tool.py |
| CH10 | Evaluation | eval/prompts.py |

## Chapter Snapshot Files

Some notebook directories (ch04, ch05, ch06, ch08, ch09) contain `.py` snapshot
files showing the core modules at that chapter's stage. Use them to compare the
implementation with the book. The runnable integration examples import
`scratch_agents`.

## Running the notebooks

Run code cells in order from a fresh kernel. Blocks labeled **Implementation
excerpt** show part of a class or method; read them with the surrounding book
explanation. They are not standalone programs.

To run CH08's three optional agent examples, uncomment their calls after setting
up the required API keys. The Excel example also requires
`7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f.xlsx` in `notebooks/ch05/gaia_workspace`,
prepared using the CH05 attachment workflow.

If a parallel workflow has failed or is awaiting approval, it raises
`ParallelWorkflowIncomplete` (available from `scratch_agents.workflows`). Inspect
its `branch_results` for each agent's result and context, and `branch_errors` for
exceptions. Automatic retry/resume is not supported. Re-running the whole
workflow can repeat actions from branches that already completed.

## Tests

```bash
uv sync --locked --extra test
uv run --extra test pytest -q
```

Tests use simulated external services and do not require API keys. To verify
provider access and live responses, run the notebooks with your own credentials.
