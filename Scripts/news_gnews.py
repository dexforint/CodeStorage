from gnews import GNews

google_news = GNews(language="ru", country="RU")
news = google_news.get_news("драко малфой, китай")

for article in news:
    print(article["title"], article["published date"])
