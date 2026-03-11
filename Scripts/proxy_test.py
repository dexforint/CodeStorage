import time
import statistics
from dataclasses import dataclass, asdict

import requests


@dataclass
class ProxyTestResult:
    proxy: str
    ok: bool
    error: str | None

    my_ip: str | None
    proxy_ip: str | None

    country: str | None
    country_code: str | None
    city: str | None
    isp: str | None

    supports_https: bool
    anonymity: str | None  # transparent / anonymous / elite / unknown

    latency_avg_ms: float | None
    latency_median_ms: float | None
    success_rate: float | None

    download_mbps: float | None
    downloaded_bytes: int | None


def _get_json(url: str, *, proxies=None, timeout=10) -> dict:
    r = requests.get(url, proxies=proxies, timeout=timeout)
    r.raise_for_status()
    return r.json()


def test_proxy(proxy: str, *, timeout=10, tries=3) -> ProxyTestResult:
    proxies = {"http": proxy, "https": proxy}

    # 1) Мой IP без прокси (нужен для оценки анонимности)
    try:
        my_ip = _get_json("https://api.ipify.org?format=json", timeout=timeout)["ip"]
    except Exception:
        my_ip = None  # не критично

    # 2) IP, который виден через прокси + HTTPS поддержка
    proxy_ip = None
    supports_https = False
    error = None

    try:
        proxy_ip = _get_json(
            "https://api.ipify.org?format=json",
            proxies=proxies,
            timeout=timeout,
        )["ip"]
        supports_https = True
    except Exception as e_https:
        # fallback на HTTP (некоторые прокси не тянут HTTPS)
        try:
            proxy_ip = _get_json(
                "http://api.ipify.org?format=json",
                proxies=proxies,
                timeout=timeout,
            )["ip"]
            supports_https = False
        except Exception as e_http:
            return ProxyTestResult(
                proxy=proxy,
                ok=False,
                error=f"Не удалось получить IP через прокси. HTTPS error: {e_https}. HTTP error: {e_http}",
                my_ip=my_ip,
                proxy_ip=None,
                country=None,
                country_code=None,
                city=None,
                isp=None,
                supports_https=False,
                anonymity=None,
                latency_avg_ms=None,
                latency_median_ms=None,
                success_rate=None,
                download_mbps=None,
                downloaded_bytes=None,
            )

    # 3) Гео по IP (делаем без прокси, чтобы не зависеть от его качества)
    country = country_code = city = isp = None
    try:
        geo = _get_json(
            f"http://ip-api.com/json/{proxy_ip}?fields=status,country,countryCode,city,isp,message",
            timeout=timeout,
        )
        if geo.get("status") == "success":
            country = geo.get("country")
            country_code = geo.get("countryCode")
            city = geo.get("city")
            isp = geo.get("isp")
        else:
            error = f"Geo lookup failed: {geo.get('message')}"
    except Exception as e:
        error = f"Geo lookup error: {e}"

    # 4) Оценка анонимности через httpbin (проверяем X-Forwarded-For и origin)
    anonymity = "unknown"
    try:
        hb = _get_json("https://httpbin.org/get", proxies=proxies, timeout=timeout)
        headers = {k.lower(): v for k, v in hb.get("headers", {}).items()}
        xff = headers.get("x-forwarded-for")
        origin = hb.get("origin")  # иногда "ip, ip"
        origin_ips = (
            [x.strip() for x in origin.split(",")] if isinstance(origin, str) else []
        )

        if my_ip and xff and my_ip in xff:
            anonymity = "transparent"
        else:
            # если в origin видно больше одного IP — часто признак прокси/цепочки, но не всегда
            if my_ip and any(ip == my_ip for ip in origin_ips):
                # бывает, если прокси не сработал или прозрачный без XFF
                anonymity = "transparent"
            else:
                # если XFF есть, но там нет моего IP — чаще "anonymous"
                # если XFF нет и IP сменился — чаще "elite"
                if xff:
                    anonymity = "anonymous"
                else:
                    anonymity = "elite"
    except Exception:
        anonymity = "unknown"

    # 5) Latency/успешность: быстрый URL (204)
    latencies = []
    ok_count = 0
    test_url = (
        "https://www.google.com/generate_204"
        if supports_https
        else "http://www.google.com/generate_204"
    )

    for _ in range(tries):
        t0 = time.perf_counter()
        try:
            r = requests.get(test_url, proxies=proxies, timeout=timeout)
            # 204/200 и т.п. — считаем успехом
            if 200 <= r.status_code < 400 or r.status_code == 204:
                ok_count += 1
            latencies.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            latencies.append(None)

    good_lat = [x for x in latencies if isinstance(x, (int, float))]
    success_rate = ok_count / tries if tries > 0 else None
    latency_avg_ms = statistics.mean(good_lat) if good_lat else None
    latency_median_ms = statistics.median(good_lat) if good_lat else None

    # 6) Скорость скачивания (Mbps): небольшой файл
    download_url = (
        "https://speed.hetzner.de/1MB.bin"
        if supports_https
        else "http://speed.hetzner.de/1MB.bin"
    )
    downloaded_bytes = 0
    download_mbps = None
    try:
        t0 = time.perf_counter()
        with requests.get(
            download_url, proxies=proxies, timeout=timeout, stream=True
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                downloaded_bytes += len(chunk)
        dt = time.perf_counter() - t0
        if dt > 0:
            download_mbps = (downloaded_bytes * 8) / (dt * 1_000_000)  # Mbps
    except Exception:
        # скорость не критична — просто оставим None
        downloaded_bytes = downloaded_bytes or None
        download_mbps = None

    return ProxyTestResult(
        proxy=proxy,
        ok=True,
        error=error,
        my_ip=my_ip,
        proxy_ip=proxy_ip,
        country=country,
        country_code=country_code,
        city=city,
        isp=isp,
        supports_https=supports_https,
        anonymity=anonymity,
        latency_avg_ms=latency_avg_ms,
        latency_median_ms=latency_median_ms,
        success_rate=success_rate,
        download_mbps=download_mbps,
        downloaded_bytes=downloaded_bytes,
    )


if __name__ == "__main__":
    # Примеры:
    # proxy = "http://1.2.3.4:8080"
    # proxy = "http://user:pass@1.2.3.4:8080"
    # proxy = "socks5://1.2.3.4:1080"
    proxy = "http://104.21.61.174:80"

    result = test_proxy(proxy, timeout=10, tries=3)
    print(asdict(result))
