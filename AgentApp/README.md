# AgentApp

This project is an advanced agentic application for Additive Manufacturing (AM). It is built to facilitate interaction with machine learning models and tool-use agents through a web-based user interface.

**The agent framework is derived from the open-source project [OpenManus](https://github.com/FoundationAgents/OpenManus), while the MCP tools related to Additive Manufacturing (AM) and Asset Administration Shell (AAS) are our original contributions.**

## Project Structure

- **`app/`**: Core application logic — agent implementations (`agent/`), MCP tools (`tool/`), the ML service (`ml_service.py`), AAS utilities (`aas_utils/`), LLM clients (`llm.py`), flow control (`flow/`), prompts (`prompt/`), schemas, and the local sandbox.
- **`amcore/`**: Shared library used by the agent's tools (label/parsing helpers, RAG, fusion logic, prompt builders, LLM client).
- **`config/`**: Configuration templates. Copy `config.example.toml` to `config.toml` and fill in your API keys / model endpoints (the live `config.toml` is git-ignored).
- **`webui.py`**: Main entry point — starts the Gradio UI and the background ML service.
- **`run_mcp_server.py`**: Optional entry that exposes the agent's tools as a standalone MCP server (SSE / stdio).
- **`ml_models/`** and **`results_AM/`**: created at runtime and git-ignored.
- **`pyproject.toml` / `uv.lock`**: Dependency management via `uv`.

## Installation and Deployment

### Prerequisites

- Python 3.11 or higher (up to 3.12).
- [uv](https://github.com/astral-sh/uv) (recommended for fast dependency management) or standard `pip`.
- CUDA-enabled GPU (recommended for local model inference).

### Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd AMAgent/AgentApp
    ```

2.  **Install Dependencies**:

    Using `uv` (Recommended):
    ```bash
    # Install dependencies from uv.lock
    uv sync
    ```

    Using `pip`:
    ```bash
    pip install .
    ```

3.  **Configuration**:
    - Ensure your environment variables (e.g., API keys for LLMs) are set up. Refer to `config/` for template configurations if available.
    - Check `app/config.py` (if applicable) for default settings.

## Running the Application

To start the AgentApp, run the `webui.py` script. This will launch both the backend ML service and the Gradio frontend.

```bash
# using uv
uv run webui.py

# or using python directly (if dependencies are installed in current env)
python webui.py
```

After running the command, the application should be accessible in your browser at `http://127.0.0.1:7860`.

## Features

- **Chat Interface**: Interact with the agent using natural language.
- **Tool Use**: The agent can utilize various tools defined in `app/tool/` to perform tasks (e.g., literature retrieval, data processing).
- **ML Integration**: Integrated machine learning models for AM-specific predictions and analysis.
