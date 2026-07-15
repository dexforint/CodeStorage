from openai import OpenAI
import numpy as np

client = OpenAI()


# ─── Получить эмбеддинг одного текста ────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",  # или "text-embedding-3-large"
        input=text,
        encoding_format="float",  # "float" или "base64"
    )
    return response.data[0].embedding


# ─── Батч-эмбеддинги (несколько текстов за раз) ──────────────────────────────
def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


# ─── Косинусное сходство ─────────────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_np, b_np = np.array(a), np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


# ─── Семантический поиск ─────────────────────────────────────────────────────
def semantic_search(query: str, documents: list[str], top_k: int = 3) -> list[tuple]:
    """Находит наиболее похожие документы по смыслу."""
    all_texts = [query] + documents
    all_embeddings = get_embeddings_batch(all_texts)

    query_emb = all_embeddings[0]
    doc_embs = all_embeddings[1:]

    scored = [
        (cosine_similarity(query_emb, doc_emb), doc)
        for doc_emb, doc in zip(doc_embs, documents)
    ]
    return sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]


# ─── Пример ──────────────────────────────────────────────────────────────────
docs = [
    "Python — язык программирования общего назначения.",
    "Django — веб-фреймворк для Python.",
    "Кошки — домашние животные.",
    "FastAPI позволяет быстро создавать REST API.",
    "Рецепт борща: свёкла, капуста, картофель.",
]

results = semantic_search("Как создать веб-приложение?", docs)
for score, doc in results:
    print(f"[{score:.3f}] {doc}")
