import asyncio
from PicImageSearch import Network, Tineye


async def search_tineye():
    async with Network() as client:
        tineye = Tineye(client=client)
        resp = await tineye.search(
            "https://i.pinimg.com/originals/6d/3a/68/6d3a68efe558025a1c797ea91fe5c6ca.png"
        )

        if resp and resp.raw:
            # print(f"Всего найдено совпадений: {resp.total}")
            for item in resp.raw:
                print(f"URL: {item.url}")
                print(f"Домен: {item.domain}")
                print(f"Размер: {item.size}")
                print(f"Дата индексации: {item.crawl_date}")
                print("---")


asyncio.run(search_tineye())
