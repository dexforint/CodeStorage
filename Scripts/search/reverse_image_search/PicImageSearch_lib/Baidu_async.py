import asyncio
from PicImageSearch import Network, BaiDu


async def search_baidu():
    async with Network() as client:
        baidu = BaiDu(client=client)
        resp = await baidu.search(
            "https://i.pinimg.com/originals/6d/3a/68/6d3a68efe558025a1c797ea91fe5c6ca.png"
        )

        if resp and resp.raw:
            for item in resp.raw:
                print(f"Превью: {item.thumbnail}")
                print(f"URL: {item.url}")
                print("---")


asyncio.run(search_baidu())
