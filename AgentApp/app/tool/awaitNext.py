from app.tool.base import BaseTool


class AwaitNext(BaseTool):
    name: str = "await_next"
    description: str = "Use this when the current task is complete and you are awaiting the user's next instruction."
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> str:
        return "Task completed. Awaiting your next instruction."

