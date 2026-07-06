from PicImageSearch.sync import Iqdb

iqdb = Iqdb(is_3d=True)  # поиск реалистичных фото
resp = iqdb.search(
    "https://i.pinimg.com/originals/6d/3a/68/6d3a68efe558025a1c797ea91fe5c6ca.png"
)

if resp and resp.raw:
    print(resp.raw[0].title)
    print(resp.raw[0].similarity)
    print(resp.raw[0].url)
    # for el in resp.raw:
    #     print(el.title)
    #     print(el.similarity)
    #     print(el.url)
