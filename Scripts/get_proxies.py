import requests


def get_proxies_thespeedx(kind="http", timeout=10):
    # kind: "http", "socks4", "socks5"
    url = f"https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/{kind}.txt"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return [line.strip() for line in r.text.splitlines() if line.strip()]


http_proxies = get_proxies_thespeedx("http")

print("count:", len(http_proxies))
print(http_proxies[:5])
