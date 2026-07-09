import asyncio
from PicImageSearch import Network, SauceNAO


async def main():
    async with Network() as client:
        saucenao = SauceNAO(client=client, api_key="ваш_api_ключ")
        resp = await saucenao.search("https://example.com/image.jpg")
        if resp and resp.raw:
            print(resp.raw[0].title)
            print(resp.raw[0].similarity)
            print(resp.raw[0].url)


asyncio.run(main())
