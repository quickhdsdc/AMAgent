import os, multiprocessing as mp

# 1) Tame BLAS/numexpr threads (must be before numpy/sklearn import)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# 2) Make joblib use threading by default (no nested process pools)
os.environ.setdefault("JOBLIB_START_METHOD", "threading")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# 3) Windows especially: use 'spawn' so children start cleanly
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import asyncio
import logging
import gradio as gr
import io
from loguru import logger as loguru_logger
from app.agent.mcp import MCPAgent
import contextlib

try:
    from app.config import config
    DEFAULT_SERVER_URL = getattr(config, "server_url", "http://127.0.0.1:8000/sse")
except Exception:
    DEFAULT_SERVER_URL = "http://127.0.0.1:8000/sse"

class LiveSummary:
    def __init__(self):
        self.in_thoughts = False
        self.thoughts_lines: list[str] = []
        self.prepared_line: str | None = None
        self.completed_lines: list[str] = []

    def _is_end_marker(self, line: str) -> bool:
        # be lenient about case/spacing
        low = line.lower()
        return (
            "🛠️" in line
            or "🔧" in line
            or "🎯" in line
            or "executing step" in low
            or ("selected" in low and "tools" in low)
        )

    def feed(self, line: str) -> bool:
        """
        Feed one log line. Returns True if the summary should be re-rendered.
        """
        updated = False
        s = line.strip()

        # Start of thoughts
        if "mcp_agent's thoughts:" in s:
            self.in_thoughts = True
            # chop everything before the colon to be safe
            after = s.split("mcp_agent's thoughts:", 1)[1].strip()
            if after:
                self.thoughts_lines.append(after)
            updated = True
            return updated

        # While inside thoughts, collect until an end marker
        if self.in_thoughts:
            if self._is_end_marker(s):
                self.in_thoughts = False
                # we will also process the marker below
            else:
                # accumulate bullets / paragraphs (can be blank)
                self.thoughts_lines.append(s)
                updated = True
                return updated

        # Tool prepared
        if "🧰" in s and "Tools being prepared" in s:
            # keep the whole tail after the colon
            tail = s.split("Tools being prepared:", 1)[1].strip() if "Tools being prepared:" in s else s
            self.prepared_line = tail or s
            updated = True

        # Tool completed
        if "🎯" in s and "Tool" in s and "completed" in s:
            self.completed_lines.append(s)
            updated = True

        return updated

    def render(self) -> str:
        parts: list[str] = []
        if self.thoughts_lines:
            # Normalize blank lines; join with \n
            body = "\n".join(self.thoughts_lines).strip()
            if body:
                parts.append(f"{body}")
        if self.prepared_line:
            parts.append(f"**Tools being prepared**: {self.prepared_line}")
        if self.completed_lines:
            parts.append("**Completed**\n" + "\n".join(self.completed_lines))
        return parts and "\n\n".join(parts) or ""


class MCPService:
    def __init__(self):
        self.agent: MCPAgent | None = None
        self.initialized = False
        self._lock = asyncio.Lock()

    async def start(self, connection: str, server_url: str | None):
        async with self._lock:
            if self.initialized:
                return
            self.agent = MCPAgent()
            if connection == "stdio":
                import sys
                await self.agent.initialize(
                    connection_type="stdio",
                    command=sys.executable,
                    args=["-m", "app.mcp.server"],
                )
            else:
                await self.agent.initialize(
                    connection_type="sse",
                    server_url=server_url or DEFAULT_SERVER_URL,
                )
            self.initialized = True

    async def ask(self, message: str) -> str:
        if not self.initialized or self.agent is None:
            raise RuntimeError("Agent not initialized")
        return await self.agent.run(message)

    async def stop(self):
        if self.agent is not None:
            try:
                await self.agent.cleanup()
            except Exception:
                logging.exception("[UI] Cleanup error")
        self.agent = None
        self.initialized = False


service = MCPService()


class RequestLogCapture:
    """Capture Loguru logs for a single request into an in-memory buffer."""
    def __init__(self, level="INFO"):
        self.level = level
        self.stream = io.StringIO()
        self._token = None

    def __enter__(self):
        self._token = loguru_logger.add(self.stream, level=self.level)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            try:
                loguru_logger.remove(self._token)
            except Exception:
                pass

    def get_text(self) -> str:
        return self.stream.getvalue()


def make_sink(q: asyncio.Queue[str]):
    """Return a Loguru sink function that pushes plain text lines into q."""
    def _sink(msg):
        # msg may be a str or a loguru.Message
        try:
            text = msg if isinstance(msg, str) else getattr(msg, "message", str(msg))
        except Exception:
            text = str(msg)
        # enqueue line-by-line so the parser sees bullets incrementally
        for line in str(text).splitlines():
            try:
                asyncio.get_running_loop().call_soon_threadsafe(q.put_nowait, line)
            except RuntimeError:
                pass
    return _sink


async def on_send_stream(user_msg, chat_history, connection, server_url, verbosity):
    await service.start(connection, server_url)
    if not user_msg.strip():
        yield gr.update(), chat_history, "```\nNo logs.\n```", chat_history
        return

    level = "INFO" if verbosity == "Verbose (INFO)" else "DEBUG" if verbosity == "Debug" else "WARNING"

    # 1) append user + placeholder assistant
    chat_history = chat_history + [{"role": "user", "content": user_msg}]
    chat_history = chat_history + [{"role": "assistant", "content": "_working…_"}]
    yield gr.update(value=""), chat_history, "```\nStarting…\n```", chat_history

    # 2) live logs via queue
    q: asyncio.Queue[str] = asyncio.Queue()
    stop = asyncio.Event()
    parser = LiveSummary()
    buffer_lines: list[str] = []

    token = loguru_logger.add(make_sink(q), level=level)

    async def run_agent():
        try:
            await service.ask(user_msg)
        except Exception as e:
            await q.put(f"ERROR: {e}")
        finally:
            stop.set()

    task = asyncio.create_task(run_agent())

    # simple throttle to avoid too many UI updates
    last_emit = 0.0

    try:
        while True:
            try:
                line = await asyncio.wait_for(q.get(), timeout=0.15)
            except asyncio.TimeoutError:
                line = None

            changed = False
            if line is not None:
                buffer_lines.append(line)
                changed = parser.feed(line)

            # emit when summary changed or periodically while busy
            now = asyncio.get_running_loop().time()
            should_emit = changed or (now - last_emit > 0.5)
            if should_emit:
                last_emit = now
                chat_history[-1] = {"role": "assistant", "content": parser.render()}
                logs_md = "```\n" + "\n".join(buffer_lines[-1000:]) + "\n```"
                yield gr.update(value=""), chat_history, logs_md, chat_history

            if stop.is_set() and q.empty():
                break

        # final emit
        full_log = "\n".join(buffer_lines)
        chat_history[-1] = {"role": "assistant", "content": parser.render()}
        logs_md = "```\n" + (full_log or "No logs.") + "\n```"
        yield gr.update(value=""), chat_history, logs_md, chat_history

    finally:
        loguru_logger.remove(token)
        with contextlib.suppress(Exception):
            await task


def on_undo(chat_state):
    if chat_state:
        chat_state = chat_state[:-1]
    return chat_state


def on_clear():
    return []


def make_app():
    with gr.Blocks(title="MCP Agent") as demo:
        gr.Markdown("# MCP Agent\nTalk to your MCP server with a simple chat UI.")

        with gr.Row():
            connection = gr.Dropdown(
                label="Connection",
                choices=["stdio", "sse"],
                value="stdio",
            )
            server_url = gr.Textbox(
                label="SSE Server URL",
                value=DEFAULT_SERVER_URL,
                placeholder="http://127.0.0.1:8000/sse",
            )
            verbosity = gr.Dropdown(
                label="Verbosity",
                choices=["Quiet (WARN)", "Verbose (INFO)", "Debug"],
                value="Verbose (INFO)",
            )

        chatbot = gr.Chatbot(height=420, label="Chat", type="messages")
        msg = gr.Textbox(placeholder="Type your instruction…", lines=2)

        with gr.Row():
            send = gr.Button("Send", variant="primary")
            undo = gr.Button("Undo")
            clear = gr.Button("Clear")

        with gr.Accordion("Details", open=True):
            logs = gr.Markdown(value="```\nNo logs yet.\n```")

        state = gr.State([])

        send.click(
            on_send_stream,
            inputs=[msg, state, connection, server_url, verbosity],
            outputs=[msg, chatbot, logs, state],
            queue=True,
        )

        msg.submit(
            on_send_stream,
            inputs=[msg, state, connection, server_url, verbosity],
            outputs=[msg, chatbot, logs, state],
            queue=True,
        )

        undo.click(on_undo, inputs=state, outputs=state).then(lambda s: s, inputs=state, outputs=chatbot)
        clear.click(on_clear, outputs=state).then(lambda s: s, inputs=state, outputs=chatbot)

        def _cleanup():
            asyncio.run(service.stop())
        demo.unload(_cleanup)

    return demo


if __name__ == "__main__":
    app = make_app()
    app.queue()
    app.launch(server_name="127.0.0.1", server_port=7860)
