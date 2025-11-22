from contextlib import AsyncExitStack
from typing import List, Optional
import asyncio, logging, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.tool_collection import ToolCollection


class MCPClientTool(BaseTool):
    """Represents a tool proxy that can be called on the MCP server from the client side."""

    session: Optional[ClientSession] = None

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool by making a remote call to the MCP server."""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        try:
            result = await self.session.call_tool(self.name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")


class MCPClients(ToolCollection):
    """
    A collection of tools that connects to an MCP server and manages available tools through the Model Context Protocol.
    """

    session: Optional[ClientSession] = None
    exit_stack: AsyncExitStack = None
    description: str = "MCP client tools for server interaction"

    def __init__(self):
        super().__init__()
        self.name = "mcp"
        self.exit_stack = AsyncExitStack()
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self.tool_map = {}
        self.tools = tuple()

    async def connect_sse(self, server_url: str) -> None:
        """Connect to an MCP server using SSE transport."""
        if not server_url:
            raise ValueError("Server URL is required.")
        if self.session:
            await self.disconnect()

        streams_context = sse_client(url=server_url)
        streams = await self.exit_stack.enter_async_context(streams_context)
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(*streams)
        )

        await self._initialize_and_list_tools()

    async def connect_stdio(self, command: str, args: List[str], env: dict | None = None) -> None:
        """
        Connect to an MCP server using stdio transport.

        IMPORTANT: We launch exactly ONE subprocess here and wire its
        stdout/stdin directly to the ClientSession. We also capture stderr
        and forward to the 'mcp.server' logger so the UI can display it.
        """
        if not command:
            raise ValueError("Server command is required.")
        if self.session:
            await self.disconnect()

        server_params = StdioServerParameters(command=command, args=args, env=env)  # <- pass env

        # stdio_client will spawn and manage the process; it yields (read, write) streams
        read, write = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self._initialize_and_list_tools()

    async def _initialize_and_list_tools(self) -> None:
        if not self.session:
            raise RuntimeError("Session not initialized.")

        await self.session.initialize()
        response = await self.session.list_tools()

        # Rebuild tool map
        self.tools = tuple()
        self.tool_map = {}
        for tool in response.tools:
            server_tool = MCPClientTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=self.session,
            )
            self.tool_map[tool.name] = server_tool
        self.tools = tuple(self.tool_map.values())

        logger.info(
            f"Connected to server with tools: {[tool.name for tool in response.tools]}"
        )

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        if self.session and self.exit_stack:
            try:
                await self.exit_stack.aclose()  # closes session and the stdio_client context (kills proc)
            except Exception:
                logging.getLogger("app").exception("Error closing exit stack")
        self.session = None
        self.tools = tuple()
        self.tool_map = {}
        logger.info("Disconnected from MCP server")
