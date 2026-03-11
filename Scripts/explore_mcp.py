import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@arabold/docs-mcp-server@latest"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.list_tools()

            print("Инструменты сервера:\n")
            for tool in result.tools:
                print(f"Название: {tool.name}")
                print(f"Описание: {tool.description or '(нет описания)'}")
                print("Схема входных параметров:")
                print(json.dumps(tool.inputSchema, ensure_ascii=False, indent=2))
                print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
