from PicImageSearch.sync import SauceNAO


def main():
    saucenao = SauceNAO(api_key="e512095bc11a33dfa27a4b3d06fe7b355db5c310")
    resp = saucenao.search(
        "https://i.pinimg.com/originals/6d/3a/68/6d3a68efe558025a1c797ea91fe5c6ca.png"
    )
    if resp and resp.raw:
        print(resp.raw[0].title)
        print(resp.raw[0].similarity)
        print(resp.raw[0].url)


main()
