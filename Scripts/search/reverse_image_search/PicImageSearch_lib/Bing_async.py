import asyncio
from PicImageSearch import Network, Bing


async def search_bing():
    async with Network() as client:
        bing = Bing(client=client)
        resp = await bing.search(
            "https://i.pinimg.com/originals/6d/3a/68/6d3a68efe558025a1c797ea91fe5c6ca.png"
        )
        print("resp")
        print(resp.best_guess)
        print(resp.entities)
        print(resp.origin)
        print(resp.pages_including)
        print(resp.raw)
        print(resp.related_searches)
        print(resp.travel)
        print(resp.url)
        print(resp.visual_search)
        print("####")
        if resp and resp.raw:
            for item in resp.raw:
                print(f"Заголовок: {item.title}")
                print(f"URL: {item.url}")
                print(f"Превью: {item.thumbnail}")
                print("---")


asyncio.run(search_bing())
