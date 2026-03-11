# check_mcp.py
import argparse
import asyncio
import json
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin

import httpx


def load_mcp_servers(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("mcpServers", {})


def jsonrpc_request(
    _id: int, method: str, params: Optional[dict] = None
) -> Dict[str, Any]:
    req = {"jsonrpc": "2.0", "id": _id, "method": method}
    if params is not None:
        req["params"] = params
    return req


def jsonrpc_notification(method: str, params: Optional[dict] = None) -> Dict[str, Any]:
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


async def sse_reader(
    resp: httpx.Response, queue: asyncio.Queue[Tuple[str, str]]
) -> None:
    """
    Minimal SSE parser:
      - collects "event:" and one/many "data:" lines
      - dispatches event on empty line
    """
    event_name: Optional[str] = None
    data_lines = []

    try:
        async for line in resp.aiter_lines():
            if line is None:
                continue
            line = line.rstrip("\n")

            # comment / keep-alive
            if line.startswith(":"):
                continue

            if line == "":
                # dispatch
                if data_lines:
                    data = "\n".join(data_lines)
                    await queue.put((event_name or "message", data))
                event_name = None
                data_lines = []
                continue

            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
                continue

            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
                continue

        # stream ended
        await queue.put(("__eof__", ""))
    except Exception as e:
        await queue.put(("__error__", repr(e)))


async def wait_for_event(
    queue: asyncio.Queue[Tuple[str, str]],
    want_event: str,
    timeout: float = 10.0,
) -> str:
    async def _wait() -> str:
        while True:
            ev, data = await queue.get()
            if ev == "__error__":
                raise RuntimeError(f"SSE error: {data}")
            if ev == "__eof__":
                raise RuntimeError("SSE stream closed")
            if ev == want_event:
                return data

    return await asyncio.wait_for(_wait(), timeout=timeout)


async def wait_for_jsonrpc_response_id(
    queue: asyncio.Queue[Tuple[str, str]],
    want_id: int,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    async def _wait() -> Dict[str, Any]:
        while True:
            ev, data = await queue.get()
            if ev == "__error__":
                raise RuntimeError(f"SSE error: {data}")
            if ev == "__eof__":
                raise RuntimeError("SSE stream closed")
            if ev != "message":
                continue

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            if isinstance(msg, dict) and msg.get("id") == want_id:
                return msg

    return await asyncio.wait_for(_wait(), timeout=timeout)


async def try_sse_mcp_check(name: str, url: str, headers: Dict[str, str]) -> None:
    """
    Typical MCP-over-SSE flow:
      - GET (SSE) to url (or url/sse) -> receive 'endpoint' event with POST url
      - POST initialize
      - wait init response
      - POST notifications/initialized
      - POST tools/list
      - wait tools/list response
    """
    sse_candidates = [url, url.rstrip("/") + "/sse"]

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        last_err = None

        for sse_url in sse_candidates:
            try:
                sse_headers = dict(headers)
                sse_headers["Accept"] = "text/event-stream"

                async with client.stream("GET", sse_url, headers=sse_headers) as resp:
                    resp.raise_for_status()

                    queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
                    reader_task = asyncio.create_task(sse_reader(resp, queue))

                    try:
                        endpoint_data = await wait_for_event(
                            queue, "endpoint", timeout=10.0
                        )
                        post_url = endpoint_data.strip()

                        # if server returned relative endpoint
                        if post_url.startswith("/"):
                            post_url = urljoin(sse_url, post_url)

                        # --- initialize ---
                        init_id = 1
                        init = jsonrpc_request(
                            init_id,
                            "initialize",
                            params={
                                "protocolVersion": "2024-11-05",
                                "clientInfo": {
                                    "name": "mcp-healthcheck",
                                    "version": "0.1.0",
                                },
                                "capabilities": {
                                    "tools": {},
                                    "resources": {},
                                    "prompts": {},
                                },
                            },
                        )
                        r_init = await client.post(
                            post_url,
                            headers={
                                **headers,
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                            },
                            json=init,
                        )
                        r_init.raise_for_status()

                        init_resp = await wait_for_jsonrpc_response_id(
                            queue, init_id, timeout=20.0
                        )
                        if "error" in init_resp:
                            raise RuntimeError(
                                f"initialize error: {init_resp['error']}"
                            )

                        server_info = (init_resp.get("result") or {}).get(
                            "serverInfo"
                        ) or {}
                        print(f"[{name}] SSE OK. serverInfo={server_info}")

                        # --- initialized notification (best practice) ---
                        await client.post(
                            post_url,
                            headers={
                                **headers,
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                            },
                            json=jsonrpc_notification("notifications/initialized"),
                        )

                        # --- tools/list ---
                        tools_id = 2
                        await client.post(
                            post_url,
                            headers={
                                **headers,
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                            },
                            json=jsonrpc_request(tools_id, "tools/list", params={}),
                        )
                        tools_resp = await wait_for_jsonrpc_response_id(
                            queue, tools_id, timeout=20.0
                        )
                        if "error" in tools_resp:
                            raise RuntimeError(
                                f"tools/list error: {tools_resp['error']}"
                            )

                        tools = (tools_resp.get("result") or {}).get("tools") or []
                        tool_names = [
                            t.get("name") for t in tools if isinstance(t, dict)
                        ]
                        print(
                            f"[{name}] tools/list OK. tools_count={len(tools)} tools={tool_names[:20]}"
                        )
                        if len(tool_names) > 20:
                            print(f"[{name}] ... ({len(tool_names) - 20} more)")

                        return
                    finally:
                        reader_task.cancel()

            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"[{name}] SSE check failed for all candidates ({sse_candidates}): {last_err}"
        )


async def try_simple_post_check(name: str, url: str, headers: Dict[str, str]) -> None:
    """
    Fallback: try direct POST initialize and expect JSON response.
    Some servers may support plain request/response transport.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        init = jsonrpc_request(
            1,
            "initialize",
            params={
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "mcp-healthcheck", "version": "0.1.0"},
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            },
        )
        resp = await client.post(
            url,
            headers={
                **headers,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=init,
        )
        resp.raise_for_status()

        try:
            payload = resp.json()
        except Exception:
            raise RuntimeError(
                f"[{name}] POST check: non-JSON response, content-type={resp.headers.get('content-type')}"
            )

        if "error" in payload:
            raise RuntimeError(
                f"[{name}] POST check: initialize error: {payload['error']}"
            )

        server_info = (payload.get("result") or {}).get("serverInfo") or {}
        print(f"[{name}] Direct POST OK. serverInfo={server_info}")


async def check_server(name: str, cfg: Dict[str, Any]) -> None:
    url = cfg["url"]
    headers = cfg.get("headers") or {}

    # headers in mcp.json are expected to be HTTP headers for remote servers
    headers = {str(k): str(v) for k, v in headers.items()}

    try:
        await try_sse_mcp_check(name, url, headers)
    except Exception as e_sse:
        print(f"[{name}] SSE failed: {e_sse}")
        # fallback
        await try_simple_post_check(name, url, headers)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mcp.json", help="Path to mcp.json")
    args = ap.parse_args()

    servers = load_mcp_servers(args.config)
    if not servers:
        raise SystemExit("No mcpServers found in config")

    failed = 0
    for name, cfg in servers.items():
        try:
            await check_server(name, cfg)
        except Exception as e:
            failed += 1
            print(f"[{name}] FAILED: {e}")

    if failed:
        raise SystemExit(f"{failed} server(s) failed")
    print("All servers OK")


if __name__ == "__main__":
    asyncio.run(main())
