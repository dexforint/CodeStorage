import asyncio
from time import perf_counter
from typing import Optional, Dict, List, Tuple
import json

import requests

import aiohttp
from aiohttp import ClientTimeout
from aiohttp_socks import ProxyConnector
from tqdm import tqdm


IP_URLS = [
    "http://api.ipify.org?format=json",
    "https://api.ipify.org?format=json",
]

GEO_BATCH_URL = "http://ip-api.com/batch?fields=status,country,query,message"


async def _get_ip_via_http_proxy(
    session: aiohttp.ClientSession,
    proxy_url: str,
    timeout_s: float,
) -> Tuple[bool, Optional[str], Optional[float]]:
    timeout = ClientTimeout(total=timeout_s)
    for url in IP_URLS:
        t0 = perf_counter()
        try:
            async with session.get(url, proxy=proxy_url, timeout=timeout) as r:
                r.raise_for_status()
                data = await r.json(content_type=None)
                ip = data.get("ip")
                if ip:
                    return True, ip, (perf_counter() - t0) * 1000.0
        except Exception:
            continue
    return False, None, None


async def _get_ip_via_socks_proxy(
    proxy_url: str,
    timeout_s: float,
) -> Tuple[bool, Optional[str], Optional[float]]:
    timeout = ClientTimeout(total=timeout_s)
    connector = ProxyConnector.from_url(proxy_url)

    async with aiohttp.ClientSession(connector=connector) as session:
        for url in IP_URLS:
            t0 = perf_counter()
            try:
                async with session.get(url, timeout=timeout) as r:
                    r.raise_for_status()
                    data = await r.json(content_type=None)
                    ip = data.get("ip")
                    if ip:
                        return True, ip, (perf_counter() - t0) * 1000.0
            except Exception:
                continue

    return False, None, None


async def probe_proxy(
    idx: int,
    proxy_url: str,
    http_session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    timeout_s: float,
) -> Tuple[int, dict]:
    scheme = proxy_url.split("://", 1)[0].lower()

    async with sem:
        if scheme.startswith("socks"):
            ok, ip, latency_ms = await _get_ip_via_socks_proxy(proxy_url, timeout_s)
        else:
            ok, ip, latency_ms = await _get_ip_via_http_proxy(
                http_session, proxy_url, timeout_s
            )

    return idx, {
        "proxy": proxy_url,
        "ok": ok,
        "ip": ip,  # внутреннее поле, потом удалим
        "latency_ms": latency_ms,
        "country": None,  # заполним после geo
    }


async def geolocate_ips_batch(
    ips: List[str],
    direct_session: aiohttp.ClientSession,
    timeout_s: float,
    batch_size: int = 100,
    show_progress: bool = True,
) -> Dict[str, Optional[str]]:
    timeout = ClientTimeout(total=timeout_s)
    ip_to_country: Dict[str, Optional[str]] = {}

    # уникальные IP, сохраняя порядок
    seen = set()
    uniq_ips = []
    for ip in ips:
        if ip and ip not in seen:
            seen.add(ip)
            uniq_ips.append(ip)

    chunks = [uniq_ips[i : i + batch_size] for i in range(0, len(uniq_ips), batch_size)]

    iterator = chunks
    if show_progress:
        iterator = tqdm(chunks, desc="Geo lookup (ip-api batch)", unit="batch")

    for chunk in iterator:
        payload = [{"query": ip} for ip in chunk]
        try:
            async with direct_session.post(
                GEO_BATCH_URL, json=payload, timeout=timeout
            ) as r:
                r.raise_for_status()
                data = await r.json(content_type=None)  # list[dict]
        except Exception:
            for ip in chunk:
                ip_to_country[ip] = None
            continue

        for item in data:
            ip = item.get("query")
            if item.get("status") == "success":
                ip_to_country[ip] = item.get("country")
            else:
                ip_to_country[ip] = None

    return ip_to_country


async def check_proxies_async(
    proxies: List[str],
    *,
    concurrency: int = 200,
    timeout_s: float = 8.0,
    show_progress: bool = True,
) -> List[dict]:
    sem = asyncio.Semaphore(concurrency)

    # общий session для HTTP-прокси
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(
        connector=connector
    ) as http_session, aiohttp.ClientSession(
        timeout=ClientTimeout(total=timeout_s)
    ) as direct_session:

        results: List[Optional[dict]] = [None] * len(proxies)

        tasks = [
            asyncio.create_task(probe_proxy(i, p, http_session, sem, timeout_s))
            for i, p in enumerate(proxies)
        ]

        iterator = asyncio.as_completed(tasks)
        if show_progress:
            iterator = tqdm(
                iterator, total=len(tasks), desc="Probing proxies", unit="proxy"
            )

        for fut in iterator:
            idx, res = await fut
            results[idx] = res

        # geo только для тех, кто ok
        ips = [r["ip"] for r in results if r and r["ok"] and r.get("ip")]
        ip_to_country = await geolocate_ips_batch(
            ips, direct_session, timeout_s, show_progress=show_progress
        )

        # финальный формат
        final: List[dict] = []
        for r in results:
            if not r:
                continue
            if r["ok"] and r.get("ip"):
                r["country"] = ip_to_country.get(r["ip"])
            r.pop("ip", None)  # убираем IP, оставляем только ok/country/latency
            final.append(r)

        return final


def get_proxies_thespeedx(kind="http", timeout=10):
    # kind: "http", "socks4", "socks5"
    url = f"https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/{kind}.txt"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return [f"{kind}://{line.strip()}" for line in r.text.splitlines() if line.strip()]


if __name__ == "__main__":
    proxies_list = get_proxies_thespeedx("http")

    check_results = asyncio.run(
        check_proxies_async(
            proxies_list, concurrency=200, timeout_s=8.0, show_progress=True
        )
    )
    ok_proxies = [proxy for proxy in check_results if proxy["ok"] == True]
    ok_proxies.sort(key=lambda el: el["latency_ms"])

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(ok_proxies, f, ensure_ascii=False, indent=2)

    ok_counter = len(ok_proxies)
    print(f"{ok_counter} / {len(check_results)}")
