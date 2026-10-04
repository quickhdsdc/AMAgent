# coding: utf-8
from app.mcp.server import MCPServer, parse_args


if __name__ == "__main__":
    args = parse_args()
    server = MCPServer()
    server.run(transport=args.transport)

