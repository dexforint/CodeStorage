# -*- coding: utf-8 -*-
# https://arena.ai/c/01a04de9-26b6-7682-81c6-e9c5342eb746
"""
ИМИТАЦИЯ ЧЕЛОВЕЧЕСКОЙ ПАМЯТИ НА ОСНОВЕ НЕЙРОННЫХ СЕТЕЙ И ЭМБЕДДИНГОВ
=====================================================================
Данный код моделирует основные компоненты человеческой памяти:
- сенсорную память (кратковременное хранилище сенсорных стимулов),
- рабочую (кратковременную) память с ограниченной ёмкостью,
- долговременную память, разделённую на эпизодическую, семантическую
  и процедурную.

Для представления информации используются векторные эмбеддинги,
полученные с помощью предобученной языковой модели (sentence-transformers).
Если библиотека недоступна, используется запасной вариант на основе
TF-IDF + SVD (менее семантичный, но не требующий дополнительных загрузок).

Автор: ассистент-исследователь
Дата: 2025
"""

import time
import math
import heapq
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Попытка импорта sentence-transformers для качественных эмбеддингов.
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    # Запасной вариант: TF-IDF + truncated SVD для создания векторных представлений.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Вычисляет косинусное сходство между двумя векторами."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Мягкий максимум для нормализации оценок сходства."""
    x = np.array(x) / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ============================================================
# 1. СЕНСОРНАЯ ПАМЯТЬ
# ============================================================
class SensoryMemory:
    """
    Сенсорная память — ультракороткое хранилище сенсорной информации.
    В биологии: иконическая (зрительная) и эхоическая (слуховая) память.
    Здесь мы моделируем буфер, который хранит последние несколько стимулов
    в течение очень короткого времени (например, 0.5 секунды).
    """

    def __init__(self, capacity: int = 5, duration: float = 0.5):
        """
        :param capacity: максимальное количество элементов в буфере.
        :param duration: время жизни элемента в секундах (после истечения удаляется).
        """
        self.capacity = capacity
        self.duration = duration
        self.buffer = deque()  # очередь (timestamp, item)

    def add(self, item: Any) -> None:
        """Добавляет новый сенсорный стимул с текущей временной меткой."""
        now = time.time()
        self.buffer.append((now, item))
        # Удаляем слишком старые элементы
        self._cleanup(now)
        # Если буфер переполнен, удаляем самые старые
        while len(self.buffer) > self.cacity:
            self.buffer.popleft()

    def _cleanup(self, now: float) -> None:
        """Удаляет элементы, время жизни которых истекло."""
        while self.buffer and (now - self.buffer[0][0] > self.duration):
            self.buffer.popleft()

    def get_recent(self) -> List[Any]:
        """Возвращает список последних стимулов (для передачи в рабочую память)."""
        self._cleanup(time.time())
        return [item for _, item in self.buffer]


# ============================================================
# 2. РАБОЧАЯ (КРАТКОВРЕМЕННАЯ) ПАМЯТЬ
# ============================================================
class WorkingMemory:
    """
    Рабочая память — система ограниченной ёмкости для временного хранения
    и манипулирования информацией. По классическим исследованиям,
    ёмкость составляет 7±2 элемента (Миллер, 1956).
    Здесь реализована LRU-подобная схема: при переполнении вытесняется
    элемент, который дольше всего не использовался (не повторялся).
    """

    def __init__(self, capacity: int = 7):
        """
        :param capacity: максимальное количество элементов (по умолчанию 7).
        """
        self.capacity = capacity
        # OrderedDict: ключ - текст элемента, значение - (timestamp последнего доступа, объект)
        self.items = OrderedDict()

    def add(self, item: Any, key: Optional[str] = None) -> None:
        """
        Добавляет элемент в рабочую память.
        Если элемент уже существует (по ключу), обновляет его timestamp (повторение).
        Если память переполнена, удаляет наименее недавно использованный элемент.
        """
        if key is None:
            # Если ключ не задан, используем строковое представление элемента
            key = str(item)
        now = time.time()
        if key in self.items:
            # Повторение: перемещаем в конец, обновляем время
            self.items.move_to_end(key)
            self.items[key] = (now, item)
        else:
            # Новый элемент: добавляем, возможно вытесняя старый
            self.items[key] = (now, item)
            if len(self.items) > self.capacity:
                # Удаляем самый старый по времени доступа (первый в OrderedDict)
                self.items.popitem(last=False)

    def refresh(self, key: str) -> bool:
        """
        Обновляет время последнего доступа для элемента (имитация повторения).
        Возвращает True, если элемент найден.
        """
        if key in self.items:
            now = time.time()
            # Перемещаем в конец и обновляем время
            self.items.move_to_end(key)
            self.items[key] = (now, self.items[key][1])
            return True
        return False

    def get_contents(self) -> List[Any]:
        """Возвращает список всех элементов в порядке от старых к новым."""
        return [val for _, val in self.items.values()]

    def clear(self) -> None:
        """Очищает рабочую память."""
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)


# ============================================================
# 3. ДОЛГОВРЕМЕННАЯ ПАМЯТЬ
# ============================================================
@dataclass
class Episode:
    """Структура эпизодического воспоминания."""

    id: int
    text: str  # описание события
    embedding: np.ndarray  # векторное представление
    timestamp: float  # время запоминания
    strength: float  # сила следа (от 0 до 1)
    emotional_weight: float  # эмоциональная значимость (0..1)
    context: Optional[str] = None  # контекст (например, место, ситуация)


@dataclass
class SemanticFact:
    """Структура семантического факта."""

    subject: str
    predicate: str
    object: str
    embedding: np.ndarray  # векторное представление всего факта
    strength: float = 1.0  # сила следа


@dataclass
class Skill:
    """Структура процедурного навыка."""

    name: str
    description: str  # описание навыка
    steps: List[str]  # последовательность действий
    embedding: np.ndarray  # векторное представление описания
    strength: float = 1.0


class LongTermMemory:
    """
    Долговременная память — практически неограниченное хранилище знаний.
    Разделяется на три подсистемы:
    - эпизодическая (личные события с временем и местом)
    - семантическая (факты о мире)
    - процедурная (навыки и умения)

    Для каждой подсистемы используются векторные представления, полученные
    с помощью языковой модели, и поиск по косинусному сходству.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_tfidf_fallback: bool = True,
    ):
        """
        :param embedding_model_name: название модели sentence-transformers.
        :param use_tfidf_fallback: использовать ли TF-IDF, если sentence-transformers недоступен.
        """
        self.episodic: List[Episode] = []
        self.semantic: List[SemanticFact] = []
        self.procedural: List[Skill] = []
        self.next_episode_id = 0

        # Инициализация энкодера
        self.encoder = None
        self.tfidf_vectorizer = None
        self.tfidf_svd = None

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.encoder = SentenceTransformer(embedding_model_name)
                print(f"Загружена модель эмбеддингов: {embedding_model_name}")
            except Exception as e:
                print(f"Не удалось загрузить sentence-transformers: {e}")
                self.encoder = None

        if self.encoder is None and use_tfidf_fallback:
            print("Использую запасной вариант: TF-IDF + SVD для эмбеддингов.")
            # Создаём заглушку: в реальном использовании нужно обучить TF-IDF на корпусе.
            # Для простоты создаём пустой векторизатор, который будет обучаться на лету.
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_svd = TruncatedSVD(n_components=50)

    def encode(self, text: str) -> np.ndarray:
        """
        Преобразует текст в векторное представление.
        Если доступна model sentence-transformers, используется она,
        иначе используется TF-IDF + SVD (обучается на добавляемых текстах).
        """
        if self.encoder is not None:
            # sentence-transformers возвращает numpy array
            return self.encoder.encode(text, convert_to_numpy=True)
        else:
            # Запасной вариант: TF-IDF (обучается на всех ранее добавленных текстах).
            # Для простоты мы будем переобучать на каждом вызове (неэффективно, но наглядно).
            all_texts = []
            all_texts.extend([ep.text for ep in self.episodic])
            all_texts.extend(
                [f"{f.subject} {f.predicate} {f.object}" for f in self.semantic]
            )
            all_texts.extend([skill.description for skill in self.procedural])
            all_texts.append(text)
            self.tfidf_vectorizer.fit(all_texts)
            tfidf_matrix = self.tfidf_vectorizer.transform(all_texts)
            self.tfidf_svd.fit(tfidf_matrix)
            reduced = self.tfidf_svd.transform(tfidf_matrix)
            return reduced[-1]  # вектор последнего текста

    # ---------- Эпизодическая память ----------
    def add_episode(
        self, text: str, emotional_weight: float = 0.5, context: Optional[str] = None
    ) -> int:
        """Добавляет новое эпизодическое воспоминание."""
        emb = self.encode(text)
        ep = Episode(
            id=self.next_episode_id,
            text=text,
            embedding=emb,
            timestamp=time.time(),
            strength=1.0,  # начальная сила следа максимальна
            emotional_weight=emotional_weight,
            context=context,
        )
        self.episodic.append(ep)
        self.next_episode_id += 1
        return ep.id

    def retrieve_episodic(
        self, query: str, top_k: int = 3, context: Optional[str] = None
    ) -> List[Episode]:
        """
        Поиск наиболее релевантных эпизодических воспоминаний.
        Сходство вычисляется как косинусная близость запроса и эпизода,
        умноженная на силу следа (учёт забывания) и эмоциональную значимость.
        Также учитывается контекст (если задан, он кодируется и добавляется к сходству).
        """
        query_emb = self.encode(query)
        scores = []
        for ep in self.episodic:
            base_sim = cosine_similarity(query_emb, ep.embedding)
            # Учет контекста: если задан контекст запроса, сравниваем с контекстом эпизода
            context_sim = 0.0
            if context is not None and ep.context is not None:
                context_emb = self.encode(context)
                ep_context_emb = self.encode(ep.context)
                context_sim = cosine_similarity(context_emb, ep_context_emb)
            # Итоговая оценка: сходство * сила следа * эмоциональный вес + бонус за контекст
            score = base_sim * ep.strength * (0.5 + 0.5 * ep.emotional_weight)
            if context is not None:
                score += 0.3 * context_sim * ep.strength
            scores.append((score, ep))
        # Сортировка по убыванию и выбор top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scores[:top_k]]

    # ---------- Семантическая память ----------
    def add_semantic_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Добавляет семантический факт (тройка субъект-предикат-объект)."""
        text = f"{subject} {predicate} {obj}"
        emb = self.encode(text)
        fact = SemanticFact(
            subject=subject, predicate=predicate, object=obj, embedding=emb
        )
        self.semantic.append(fact)

    def retrieve_semantic(self, query: str, top_k: int = 3) -> List[SemanticFact]:
        """Ищет наиболее похожие семантические факты по текстовому запросу."""
        query_emb = self.encode(query)
        scored = []
        for fact in self.semantic:
            fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
            fact_emb = self.encode(fact_text)  # пересчитываем для единообразия
            sim = cosine_similarity(query_emb, fact_emb)
            scored.append((sim * fact.strength, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]

    # ---------- Процедурная память ----------
    def add_skill(self, name: str, description: str, steps: List[str]) -> None:
        """Добавляет процедурный навык с описанием и шагами."""
        emb = self.encode(description)
        skill = Skill(name=name, description=description, steps=steps, embedding=emb)
        self.procedural.append(skill)

    def retrieve_skill(self, query: str, top_k: int = 1) -> List[Skill]:
        """Находит навык по описанию задачи."""
        query_emb = self.encode(query)
        scored = []
        for skill in self.procedural:
            sim = cosine_similarity(query_emb, skill.embedding)
            scored.append((sim * skill.strength, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:top_k]]

    # ---------- Консолидация и забывание ----------
    def consolidate_from_working_memory(
        self,
        working_memory: WorkingMemory,
        emotional_weight: float = 0.5,
        context: Optional[str] = None,
    ) -> None:
        """
        Переносит содержимое рабочей памяти в долговременную (в основном эпизодическую).
        В реальном мозге консолидация включает перенос из гиппокампа в кору,
        здесь мы просто добавляем все элементы рабочей памяти как эпизоды.
        """
        for item in working_memory.get_contents():
            # Преобразуем элемент в строку (если это не строка)
            text = str(item)
            self.add_episode(text, emotional_weight=emotional_weight, context=context)
        # После консолидации рабочая память очищается (как будто информация перешла)
        working_memory.clear()

    def forget(self, decay_rate: float = 0.95) -> None:
        """
        Имитация забывания: экспоненциальное затухание силы следа.
        decay_rate < 1: чем меньше, тем быстрее забывание.
        Элементы с очень малой силой удаляются.
        """
        # Эпизодическая память
        for ep in self.episodic:
            ep.strength *= decay_rate
        self.episodic = [
            ep for ep in self.episodic if ep.strength > 0.01
        ]  # порог удаления

        # Семантическая память (забывание медленнее, можно задать другой rate)
        for fact in self.semantic:
            fact.strength *= decay_rate**0.5  # семантические знания более устойчивы
        self.semantic = [fact for fact in self.semantic if fact.strength > 0.01]

        # Процедурная память (ещё медленнее)
        for skill in self.procedural:
            skill.strength *= decay_rate**0.2
        self.procedural = [skill for skill in self.procedural if skill.strength > 0.01]


# ============================================================
# 4. ОБЩАЯ СИСТЕМА ПАМЯТИ
# ============================================================
class MemorySystem:
    """
    Интеграция всех компонентов памяти.
    Моделирует поток информации: сенсорная память -> внимание -> рабочая память
    -> консолидация -> долговременная память.
    """

    def __init__(self, attention_threshold: float = 0.3):
        """
        :param attention_threshold: порог "внимания" для переноса из сенсорной в рабочую память.
                                     В реальности внимание зависит от новизны, значимости и т.п.
        """
        self.sensory = SensoryMemory(capacity=5, duration=0.5)
        self.working = WorkingMemory(capacity=7)
        self.longterm = LongTermMemory()
        self.attention_threshold = attention_threshold
        self.last_sensory_items = []  # для сравнения новизны

    def perceive(
        self, text: str, emotional_weight: float = 0.5, context: Optional[str] = None
    ) -> None:
        """
        Обработка нового сенсорного стимула (текстового описания).
        Этапы:
        1. Стимул попадает в сенсорную память.
        2. Если он достаточно "интересен" (новизна или эмоциональная значимость),
           он переносится в рабочую память.
        3. Рабочая память может быть ограничена; при её заполнении старые элементы вытесняются.
        4. Периодически (здесь каждый раз, но можно по таймеру) рабочая память
           консолидируется в долговременную.
        """
        # Добавляем в сенсорную память
        self.sensory.add(text)
        # Вычисляем "интересность" стимула (простая эвристика)
        # Новизна: насколько текст отличается от последних стимулов в сенсорной памяти
        novelty = 1.0
        recent_items = self.sensory.get_recent()
        if len(recent_items) > 1:
            # Сравниваем с предыдущим (можно улучшить, усреднив все)
            prev = recent_items[-2] if len(recent_items) >= 2 else None
            if prev is not None:
                # Используем грубое сравнение строк (можно улучшить)
                novelty = 0.0 if prev == text else 1.0
        # Эмоциональная значимость влияет на внимание
        attention_score = novelty * 0.7 + emotional_weight * 0.3
        if attention_score >= self.attention_threshold:
            # Перенос в рабочую память (ключ - сам текст)
            self.working.add(text, key=text)
            # Консолидация в долговременную память (в реальности это происходит во время
            # сна или повторения, здесь для простоты сразу)
            self.longterm.consolidate_from_working_memory(
                self.working, emotional_weight=emotional_weight, context=context
            )
            print(f"Воспринято: '{text}' -> рабочая память -> эпизодическая память")
        else:
            print(f"Стимул '{text}' проигнорирован (низкое внимание).")

    def recall(
        self, query: str, context: Optional[str] = None, top_k: int = 3
    ) -> List[str]:
        """
        Извлечение воспоминаний по запросу.
        Возвращает тексты наиболее релевантных эпизодов и фактов.
        """
        episodes = self.longterm.retrieve_episodic(query, top_k=top_k, context=context)
        facts = self.longterm.retrieve_semantic(query, top_k=top_k)
        results = [f"Эпизод: {ep.text}" for ep in episodes]
        results += [
            f"Факт: {fact.subject} {fact.predicate} {fact.object}" for fact in facts
        ]
        return results

    def simulate_time(self, steps: int = 1, decay_rate: float = 0.9) -> None:
        """
        Имитация течения времени: забывание в долговременной памяти.
        """
        for _ in range(steps):
            self.longterm.forget(decay_rate)


# ============================================================
# 5. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================
if __name__ == "__main__":
    print("=== Демонстрация модели памяти ===\n")
    # Создаём систему памяти
    memory = MemorySystem(attention_threshold=0.4)

    # Подаём несколько сенсорных стимулов (событий)
    print("--- Поступление информации ---")
    memory.perceive(
        "Я пошёл в магазин и купил яблоки", emotional_weight=0.6, context="улица"
    )
    time.sleep(0.2)  # имитация короткой паузы
    memory.perceive("На улице шёл дождь", emotional_weight=0.4, context="улица")
    time.sleep(0.2)
    memory.perceive(
        "Я встретил друга по имени Алексей", emotional_weight=0.8, context="парк"
    )
    time.sleep(0.2)
    memory.perceive("Мы обсуждали новый фильм", emotional_weight=0.5, context="кафе")
    time.sleep(0.2)
    memory.perceive(
        "Завтра будет солнечно", emotional_weight=0.3, context="дом"
    )  # низкая значимость

    # Добавим семантические факты
    print("\n--- Добавление семантических знаний ---")
    memory.longterm.add_semantic_fact("Париж", "является столицей", "Франции")
    memory.longterm.add_semantic_fact("Яблоко", "это", "фрукт")
    memory.longterm.add_semantic_fact("Солнце", "является", "звездой")

    # Добавим процедурный навык
    memory.longterm.add_skill(
        "Приготовление чая",
        "Как заварить чай",
        [
            "Вскипятить воду",
            "Положить заварку",
            "Залить кипятком",
            "Подождать 3 минуты",
        ],
    )

    # Симулируем течение времени (забывание)
    print("\n--- Симуляция времени (забывание) ---")
    memory.simulate_time(steps=2, decay_rate=0.8)
    print("Прошло 2 шага времени, сила следов уменьшена.")

    # Извлечение воспоминаний
    print("\n--- Вспоминание ---")
    query1 = "Что я делал в магазине?"
    results1 = memory.recall(query1, context="улица")
    print(f"Запрос: '{query1}'")
    for res in results1:
        print(" -", res)

    query2 = "Какой фрукт я купил?"
    results2 = memory.recall(query2)
    print(f"\nЗапрос: '{query2}'")
    for res in results2:
        print(" -", res)

    query3 = "Как приготовить чай?"
    skills = memory.longterm.retrieve_skill(query3)
    print(f"\nЗапрос: '{query3}'")
    for skill in skills:
        print(f" - Навык: {skill.name}, шаги: {', '.join(skill.steps)}")

    print("\n=== Демонстрация завершена ===")
