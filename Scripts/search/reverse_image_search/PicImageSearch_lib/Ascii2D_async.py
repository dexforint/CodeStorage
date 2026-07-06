import asyncio
from PicImageSearch import Network, Ascii2D


async def search_ascii2d():
    async with Network() as client:
        # bovw=True  — поиск по характерным признакам (feature search)
        # bovw=False — поиск по цвету (по умолчанию)
        ascii2d = Ascii2D(client=client, bovw=True)
        resp = await ascii2d.search(
            "https://i.pinimg.com/originals/6d/3a/68/6d3a68efe558025a1c797ea91fe5c6ca.png"
        )

        if resp and resp.raw:
            for item in resp.raw:
                print(f"Заголовок: {item.title}")
                print(f"Автор: {item.author}")
                print(f"Ссылка на автора: {item.author_url}")
                print(f"URL: {item.url}")
                print(f"Превью: {item.thumbnail}")
                print("---")


asyncio.run(search_ascii2d())
