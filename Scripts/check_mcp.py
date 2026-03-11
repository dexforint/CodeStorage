#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx


JSON = Dict[str, Any]


def _mask_secret(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return "*" * (len(value) - keep) + value[-keep:]


def _sanitize_headers_for_print(headers: Dict[str, str]) -> Dict[str, str]:
    secret_markers = ("key", "token", "secret", "authorization", "api")
    out: Dict[str, str] = {}
    for k, v in (headers or {}).items():
        if any(m in k.lower() for m in secret_markers):
            out[k] = _mask_secret(v)
        else:
            out[k] = v
    return out


def _jsonrpc_request(
    req_id: Union[int, str], method: str, params: Optional[Dict[str, Any]] = None
) -> JSON:
    msg: JSON = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _jsonrpc_notification(method: str, params: Optional[Dict[str, Any]] = None) -> JSON:
    msg: JSON = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


class MCPTransportError(RuntimeError):
    pass


def _iter_sse_json_messages(resp: httpx.Response):
    """
    Парсер SSE, который достает JSON из строк `data: ...`
    и yield'ит decoded JSON (dict или list).
    """
    buf = ""
    event_lines: List[str] = []

    def flush_event(lines: List[str]):
        data_lines = []
        for ln in lines:
            if ln.startswith("data:"):
                data_lines.append(ln[5:].lstrip())
        if not data_lines:
            return None
        data = "\n".join(data_lines).strip()
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    for chunk in resp.iter_text():
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")

            # конец SSE-event'а
            if line == "":
                parsed = flush_event(event_lines)
                event_lines = []
                if parsed is None:
                    continue
                yield parsed
                continue

            # комментарии/поля event/id/retry нам не важны, но сохраняем строку
            event_lines.append(line)

    # хвост
    if event_lines:
        parsed = flush_event(event_lines)
        if parsed is not None:
            yield parsed


@dataclass
class MCPServerConfig:
    name: str
    url: str
    headers: Dict[str, str]


@dataclass
class MCPCheckResult:
    ok: bool
    details: List[str]


def _normalize_auth_headers(h: Dict[str, str]) -> Dict[str, str]:
    """
    Context7 MCP часто настраивают через заголовок CONTEXT7_API_KEY (например, Cursor). <!--citation:5-->
    Некоторые клиенты используют Authorization: Bearer ... <!--citation:6-->

    Скрипт:
      - всегда отправляет то, что есть в mcp.json
      - дополнительно подставляет Authorization: Bearer <key>, если видит CONTEXT7_API_KEY
        (лишний заголовок обычно не мешает).
    """
    headers = dict(h or {})
    if "Authorization" not in headers:
        api_key = headers.get("CONTEXT7_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _http_ping_if_supported(
    client: httpx.Client, base_mcp_url: str, headers: Dict[str, str], timeout_s: float
) -> Tuple[bool, str]:
    ping_url = base_mcp_url.rstrip("/") + "/ping"
    try:
        r = client.get(ping_url, headers=headers, timeout=timeout_s)
        if r.status_code == 200:
            # ожидаемый формат у Context7: {"status":"ok","message":"pong"} <!--citation:1-->
            return True, f"HTTP ping {ping_url}: 200 OK"
        return False, f"HTTP ping {ping_url}: HTTP {r.status_code}"
    except Exception as e:
        return False, f"HTTP ping {ping_url}: error {type(e).__name__}: {e}"


def _post_jsonrpc(
    client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    msg: JSON,
    timeout_s: float,
) -> Tuple[httpx.Response, Optional[JSON]]:
    """
    Отправляет один JSON-RPC message (request/notification) как HTTP POST.
    Для request возвращает (response, decoded_message_or_none).
    Для notification часто будет 202 без тела. <!--citation:7-->
    """
    req_headers = dict(headers)
    req_headers.setdefault("Content-Type", "application/json")
    req_headers.setdefault("Accept", "application/json, text/event-stream")

    r = client.post(url, headers=req_headers, json=msg, timeout=timeout_s)

    ct = (r.headers.get("Content-Type") or "").lower()
    if "text/event-stream" in ct:
        # SSE: достаем сообщения, но ответ на конкретный request нужно искать по id
        return r, None

    # JSON (или пусто)
    if r.status_code == 202:
        return r, None
    if r.content:
        try:
            return r, r.json()
        except Exception:
            return r, None
    return r, None


def _send_request_wait_response_sse(
    client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    req: JSON,
    timeout_s: float,
) -> Tuple[httpx.Response, JSON]:
    """
    Для случая, когда сервер отвечает SSE на JSON-RPC request:
    читаем stream, пока не придет JSON-RPC response с нужным id.
    """
    req_headers = dict(headers)
    req_headers.setdefault("Content-Type", "application/json")
    req_headers.setdefault("Accept", "application/json, text/event-stream")

    with client.stream(
        "POST", url, headers=req_headers, json=req, timeout=timeout_s
    ) as r:
        ct = (r.headers.get("Content-Type") or "").lower()
        if "text/event-stream" not in ct:
            # сервер внезапно вернул JSON
            if r.status_code >= 400:
                raise MCPTransportError(f"HTTP {r.status_code}: {r.text[:500]}")
            try:
                return r, r.json()
            except Exception as e:
                raise MCPTransportError(f"Non-JSON response, cannot decode: {e}")

        target_id = req.get("id")
        start = time.time()
        for decoded in _iter_sse_json_messages(r):
            # бывают batch-ответы
            msgs = decoded if isinstance(decoded, list) else [decoded]
            for m in msgs:
                if (
                    isinstance(m, dict)
                    and m.get("id") == target_id
                    and ("result" in m or "error" in m)
                ):
                    return r, m

            if time.time() - start > timeout_s:
                raise MCPTransportError("Timeout waiting JSON-RPC response over SSE")

        raise MCPTransportError("SSE stream ended before receiving JSON-RPC response")


def check_one_server(
    server: MCPServerConfig, timeout_s: float, do_http_ping: bool = True
) -> MCPCheckResult:
    details: List[str] = []
    ok = True

    headers = _normalize_auth_headers(server.headers)
    safe_headers = _sanitize_headers_for_print(headers)

    details.append(f"Server={server.name}")
    details.append(f"URL={server.url}")
    details.append(f"Headers={safe_headers}")

    with httpx.Client(follow_redirects=True) as client:
        # 1) HTTP /ping (если есть)
        if do_http_ping:
            ping_ok, ping_msg = _http_ping_if_supported(
                client, server.url, headers, timeout_s
            )
            details.append(ping_msg)
            # /ping не обязателен для MCP в целом, поэтому не валим проверку жестко

        # 2) JSON-RPC ping (может работать до init, зависит от сервера)
        try:
            ping_req = _jsonrpc_request("ping-1", "ping")
            r, decoded = _post_jsonrpc(client, server.url, headers, ping_req, timeout_s)
            if (
                decoded is None
                and (r.headers.get("Content-Type") or "")
                .lower()
                .find("text/event-stream")
                >= 0
            ):
                # SSE-пинг
                r2, decoded2 = _send_request_wait_response_sse(
                    client, server.url, headers, ping_req, timeout_s
                )
                decoded = decoded2
                r = r2

            if isinstance(decoded, dict) and decoded.get("result") == {}:
                details.append("JSON-RPC ping: OK")
            else:
                details.append(
                    f"JSON-RPC ping: got HTTP {r.status_code}, body={str(decoded)[:300]}"
                )
        except Exception as e:
            details.append(f"JSON-RPC ping: error {type(e).__name__}: {e}")

        # 3) initialize (Streamable HTTP)
        # Пытаемся с 2025-06-18, при проблемах откатываемся на 2025-03-26. <!--citation:7-->
        protocol_candidates = ["2025-06-18", "2025-03-26"]
        init_resp: Optional[JSON] = None
        session_id: Optional[str] = None
        negotiated_version: Optional[str] = None

        last_init_err: Optional[str] = None

        for pv in protocol_candidates:
            try:
                init_req = _jsonrpc_request(
                    1,
                    "initialize",
                    params={
                        "protocolVersion": pv,
                        "capabilities": {},
                        "clientInfo": {"name": "mcp-healthcheck", "version": "0.1.0"},
                    },
                )

                # пробуем обычный POST; если пришел SSE — читаем stream
                r, decoded = _post_jsonrpc(
                    client, server.url, headers, init_req, timeout_s
                )

                if (
                    decoded is None
                    and (r.headers.get("Content-Type") or "")
                    .lower()
                    .find("text/event-stream")
                    >= 0
                ):
                    r2, decoded2 = _send_request_wait_response_sse(
                        client, server.url, headers, init_req, timeout_s
                    )
                    decoded = decoded2
                    r = r2

                if r.status_code >= 400:
                    raise MCPTransportError(
                        f"initialize HTTP {r.status_code}: {r.text[:500]}"
                    )

                if not isinstance(decoded, dict):
                    raise MCPTransportError(f"initialize: unexpected body={decoded!r}")

                if "error" in decoded:
                    raise MCPTransportError(
                        f"initialize JSON-RPC error: {decoded['error']}"
                    )

                init_resp = decoded
                session_id = r.headers.get("Mcp-Session-Id")
                result = decoded.get("result") or {}
                negotiated_version = result.get("protocolVersion")
                details.append(
                    f"initialize: OK (requested={pv}, negotiated={negotiated_version}, session={session_id})"
                )
                break

            except Exception as e:
                last_init_err = f"{type(e).__name__}: {e}"
                details.append(f"initialize (pv={pv}): FAIL ({last_init_err})")

        if init_resp is None:
            ok = False
            details.append("FATAL: initialize failed for all protocol versions")
            return MCPCheckResult(ok=ok, details=details)

        # 4) notifications/initialized (завершение handshake) <!--citation:2-->
        # После init: добавляем Mcp-Session-Id (если есть) и MCP-Protocol-Version (обязателен “на последующих запросах”). <!--citation:7-->
        post_headers = dict(headers)
        if session_id:
            post_headers["Mcp-Session-Id"] = session_id
        if negotiated_version:
            post_headers["MCP-Protocol-Version"] = negotiated_version

        try:
            init_notif = _jsonrpc_notification("notifications/initialized")
            r = client.post(
                server.url,
                headers={
                    **post_headers,
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json=init_notif,
                timeout=timeout_s,
            )
            if r.status_code in (200, 202):
                details.append(f"notifications/initialized: OK (HTTP {r.status_code})")
            else:
                details.append(
                    f"notifications/initialized: FAIL (HTTP {r.status_code}, body={r.text[:300]})"
                )
                ok = False
        except Exception as e:
            details.append(f"notifications/initialized: error {type(e).__name__}: {e}")
            ok = False

        # 5) tools/list <!--citation:3-->
        try:
            tools_req = _jsonrpc_request(2, "tools/list")
            r, decoded = _post_jsonrpc(
                client, server.url, post_headers, tools_req, timeout_s
            )
            if (
                decoded is None
                and (r.headers.get("Content-Type") or "")
                .lower()
                .find("text/event-stream")
                >= 0
            ):
                r2, decoded2 = _send_request_wait_response_sse(
                    client, server.url, post_headers, tools_req, timeout_s
                )
                decoded = decoded2
                r = r2

            if r.status_code >= 400:
                raise MCPTransportError(
                    f"tools/list HTTP {r.status_code}: {r.text[:500]}"
                )

            if not isinstance(decoded, dict):
                raise MCPTransportError(f"tools/list: unexpected body={decoded!r}")
            if "error" in decoded:
                raise MCPTransportError(
                    f"tools/list JSON-RPC error: {decoded['error']}"
                )

            tools = (decoded.get("result") or {}).get("tools") or []
            tool_names = [t.get("name") for t in tools if isinstance(t, dict)]
            details.append(f"tools/list: OK (count={len(tool_names)})")
            if tool_names:
                details.append(
                    "tools: "
                    + ", ".join(tool_names[:30])
                    + (" ..." if len(tool_names) > 30 else "")
                )

        except Exception as e:
            details.append(f"tools/list: FAIL ({type(e).__name__}: {e})")
            ok = False

        # 6) (опционально) закрыть сессию DELETE
        # Спецификация разрешает DELETE с Mcp-Session-Id, но сервер может вернуть 405. <!--citation:7-->
        if session_id:
            try:
                r = client.delete(
                    server.url,
                    headers={
                        **post_headers,
                        "Accept": "application/json, text/event-stream",
                    },
                    timeout=timeout_s,
                )
                details.append(
                    f"DELETE session: HTTP {r.status_code} (allowed=not guaranteed)"
                )
            except Exception as e:
                details.append(f"DELETE session: error {type(e).__name__}: {e}")

    return MCPCheckResult(ok=ok, details=details)


def load_servers(path: str) -> List[MCPServerConfig]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("mcpServers") or {}
    servers: List[MCPServerConfig] = []
    for name, cfg in m.items():
        url = cfg.get("url")
        if not url:
            continue
        headers = cfg.get("headers") or {}
        servers.append(MCPServerConfig(name=name, url=url, headers=headers))
    return servers


def main() -> int:
    ap = argparse.ArgumentParser(description="MCP (Streamable HTTP) server checker")
    ap.add_argument("config", help="Path to mcp.json", default="./mcp.json")
    ap.add_argument("--server", help="Check only one server by name")
    ap.add_argument(
        "--timeout", type=float, default=20.0, help="Timeout seconds (default: 20)"
    )
    ap.add_argument(
        "--no-http-ping", action="store_true", help="Do not call GET <url>/ping"
    )
    args = ap.parse_args()

    servers = load_servers(args.config)
    if args.server:
        servers = [s for s in servers if s.name == args.server]

    if not servers:
        print("No servers found in config (or filtered out).", file=sys.stderr)
        return 2

    all_ok = True
    for s in servers:
        res = check_one_server(
            s, timeout_s=args.timeout, do_http_ping=not args.no_http_ping
        )
        status = "OK" if res.ok else "FAIL"
        print("=" * 80)
        print(f"[{status}] {s.name}")
        for line in res.details:
            print("  " + line)
        all_ok = all_ok and res.ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
