from PicImageSearch import Network

############# ASYNC

# Без прокси
async with Network() as client:
    ...

# С HTTP-прокси
async with Network(proxies="http://127.0.0.1:7890") as client:
    ...

# С SOCKS5-прокси (требуется pip install PicImageSearch[socks])
async with Network(proxies="socks5://127.0.0.1:1080") as client:
    ...

# С таймаутом (секунды)
async with Network(timeout=30) as client:
    ...


############# dSYNC

from PicImageSearch.sync import SauceNAO

saucenao = SauceNAO(proxies="http://127.0.0.1:7890", api_key="ваш_ключ")
resp = saucenao.search("image.jpg")
