import time
import requests
from urllib.parse import urlparse, urlunparse
from typing import Dict, Any


def normalize_proxy(proxy: str) -> str:
    """
    Приводит прокси к удобному для requests виду.
    Поддерживает: ip:port, login:pass@ip:port, ip:port:login:pass, socks5://...
    """
    proxy = proxy.strip()
    if not proxy:
        return proxy

    # Уже с протоколом
    if any(
        proxy.lower().startswith(s)
        for s in [
            "http://",
            "https://",
            "socks4://",
            "socks5://",
            "socks4a://",
            "socks5h://",
        ]
    ):
        return proxy

    # login:pass@ip:port
    if "@" in proxy:
        return f"http://{proxy}"

    parts = proxy.split(":")
    if len(parts) == 4:  # ip:port:login:password
        ip, port, login, password = parts
        return f"http://{login}:{password}@{ip}:{port}"

    # ip:port
    return f"http://{proxy}"


def check_proxy(proxy: str, timeout: int = 12) -> Dict[str, Any]:
    """
    Проверяет прокси.

    Возвращает:
        working (bool),
        latency_ms (float | None),
        country (str | None),
        country_name (str | None),
        ip (str | None),
        error (str | None)
    """
    proxy_url = normalize_proxy(proxy)
    proxies = {"http": proxy_url, "https": proxy_url}

    # Для SOCKS лучше использовать remote DNS resolve (h/a)
    parsed = urlparse(proxy_url)
    if parsed.scheme in ("socks5", "socks4"):
        scheme = "socks5h" if parsed.scheme == "socks5" else "socks4a"
        proxy_url = urlunparse(parsed._replace(scheme=scheme))
        proxies = {"http": proxy_url, "https": proxy_url}

    start = time.perf_counter()

    try:
        response = requests.get(
            "https://ipinfo.io/json",
            proxies=proxies,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            verify=False,  # иногда прокси имеют проблемы с SSL
        )
        response.raise_for_status()
        data = response.json()

        latency = round((time.perf_counter() - start) * 1000, 2)

        return {
            "working": True,
            "latency_ms": latency,
            "ip": data.get("ip"),
            "country": data.get("country"),  # ISO код (RU, US, DE...)
            "country_name": data.get("country_name"),
            "city": data.get("city"),
            "error": None,
            "proxy": proxy_url,
        }

    except Exception as e:
        return {
            "working": False,
            "latency_ms": None,
            "ip": None,
            "country": None,
            "country_name": None,
            "city": None,
            "error": str(e),
            "proxy": proxy_url,
        }


# ====================== АСИНХРОННАЯ ВЕРСИЯ ======================

import aiohttp
from aiohttp_socks import ProxyConnector


async def check_proxy_async(proxy: str, timeout: int = 12) -> Dict[str, Any]:
    """
    Асинхронный аналог check_proxy.
    Требует: pip install aiohttp aiohttp_socks
    """
    proxy_url = normalize_proxy(proxy)
    start = time.perf_counter()

    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        if any(s in proxy_url.lower() for s in ["socks4", "socks5"]):
            connector = ProxyConnector.from_url(proxy_url)
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout_obj, headers=headers
            ) as session:
                async with session.get("https://ipinfo.io/json") as resp:
                    data = await resp.json()
        else:
            # HTTP/HTTPS
            async with aiohttp.ClientSession(
                timeout=timeout_obj, headers=headers
            ) as session:
                async with session.get(
                    "https://ipinfo.io/json", proxy=proxy_url
                ) as resp:
                    data = await resp.json()

        latency = round((time.perf_counter() - start) * 1000, 2)

        return {
            "working": True,
            "latency_ms": latency,
            "ip": data.get("ip"),
            "country": data.get("country"),
            "country_name": data.get("country_name"),
            "city": data.get("city"),
            "error": None,
            "proxy": proxy_url,
        }

    except Exception as e:
        return {
            "working": False,
            "latency_ms": None,
            "ip": None,
            "country": None,
            "country_name": None,
            "city": None,
            "error": str(e),
            "proxy": proxy_url,
        }


# ====================== АСИНХРОННАЯ ВЕРСИЯ ПРОВЕРКИ СПИСКА ПРОКСИ ======================

import asyncio
from typing import List, Dict, Any
from tqdm.asyncio import tqdm


async def check_proxies_async(
    proxies: List[str],
    concurrency: int = 100,
    timeout: int = 12,
    only_working: bool = False,
    desc: str = "Проверка прокси",
) -> List[Dict[str, Any]]:
    """
    Асинхронная проверка списка прокси с ограничением параллельности.

    Параметры:
        proxies — список прокси (в любом формате)
        concurrency — сколько проверять одновременно (рекомендуется 60–150)
        timeout — таймаут на один прокси
        only_working — если True, возвращает только рабочие прокси
        desc — описание прогресс-бара
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_check(proxy: str) -> Dict[str, Any]:
        async with semaphore:
            result = await check_proxy_async(proxy, timeout=timeout)
            result["original_proxy"] = proxy.strip()  # сохраняем оригинальную строку
            return result

    # Запускаем все задачи с красивым прогресс-баром
    tasks = [bounded_check(proxy) for proxy in proxies]

    results = await tqdm.gather(*tasks, desc=desc, total=len(proxies), colour="blue")

    if only_working:
        results = [r for r in results if r.get("working", False)]

    return results


# ====================== MAIN FUNCTION ======================

import asyncio


async def main():
    proxy_list = get_proxies2()
    results = await check_proxies_async(
        proxies=proxy_list,
        concurrency=120,  # подберите под своё соединение
        timeout=15,
        only_working=True,  # False — вернёт все результаты
        desc="Проверка 5000 прокси",
    )

    # Вывод результатов
    print(f"\nНайдено рабочих прокси: {len(results)}\n")

    for r in results:
        if r["working"]:
            print(
                f"✅ {r['original_proxy']:35} → "
                f"{r.get('country', '??')} | "
                f"{r['latency_ms']:6.1f}ms | "
                f"{r.get('city', '')}"
            )


def get_proxies1() -> list[str]:
    response = requests.get(
        "https://github.com/zloi-user/hideip.me/raw/refs/heads/main/http.txt"
    )
    result = response.text.strip()

    proxies = result.split("\n")
    proxies = ["http://" + ":".join(proxy.split(":")[:-1]) for proxy in proxies]
    return proxies


def get_proxies2() -> list[str]:
    response = requests.get(
        "https://github.com/themiralay/Proxy-List-World/raw/refs/heads/master/data.txt"
    )
    result = response.text.strip()

    proxies = result.split("\n")
    proxies = ["http://" + proxy for proxy in proxies]
    return proxies


if __name__ == "__main__":
    asyncio.run(main())

    # print(get_proxies2())
