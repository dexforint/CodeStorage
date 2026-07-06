import asyncio
from PicImageSearch import Network, Yandex


async def search_yandex():
    async with Network() as client:
        yandex = Yandex(client=client)
        resp = await yandex.search(
            "https://i.pinimg.com/originals/6d/3a/68/6d3a68efe558025a1c797ea91fe5c6ca.png"
        )

        if resp and resp.raw:
            for item in resp.raw:
                print(f"Заголовок: {item.title}")
                print(f"URL: {item.url}")
                print(f"Превью: {item.thumbnail}")
                print(f"Источник: {item.source}")
                print(f"Размер: {item.size}")
                print("---")


asyncio.run(search_yandex())
