import asyncio
from PicImageSearch import Network, Copyseeker


async def search_copyseeker():
    async with Network() as client:
        copyseeker = Copyseeker(client=client)
        resp = await copyseeker.search(
            "https://i.pinimg.com/originals/6d/3a/68/6d3a68efe558025a1c797ea91fe5c6ca.png"
        )

        if resp and resp.raw:
            for item in resp.raw:
                print(f"URL: {item.url}")
                print(f"Превью: {item.thumbnail}")
                print("---")


asyncio.run(search_copyseeker())
