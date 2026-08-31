# https://arena.ai/c/01a04de4-8991-776a-bb8d-7301316342bb
"""
================================================================================
СИМУЛЯЦИЯ ЧЕЛОВЕЧЕСКОЙ ПАМЯТИ
================================================================================
Модель имитирует полный цикл работы памяти по аналогии с нейробиологией:

    Восприятие → Сенсорная память → Рабочая (кратковременная) память
    → Консолидация (аналог сна) → Долговременная память
    → Извлечение (реконсолидация) → Забывание (кривая Эббингауза)

Используются:
  - numpy               - векторные вычисления (эмбеддинги, косинусное сходство)
  - sentence-transformers (опционально) - нейросетевые эмбеддинги текста,
    имитирующие семантическое "кодирование" информации в коре мозга

Если sentence-transformers не установлен, используется упрощённый
хэш-эмбеддинг (bag-of-words), чтобы код работал полностью автономно.

Установка (опционально, для качественных эмбеддингов):
    pip install sentence-transformers numpy
================================================================================
"""

import time
import math
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --------------------------------------------------------------------------
# Пытаемся подключить библиотеку для получения качественных семантических
# эмбеддингов (аналог того, как мозг превращает воспринятую информацию
# в распределённый паттерн нейронной активности).
# --------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False


# ============================================================================
# БЛОК 0. ВИРТУАЛЬНЫЕ ЧАСЫ
# ============================================================================
class VirtualClock:
    """
    Виртуальные часы позволяют "ускорять" время в симуляции.

    Зачем это нужно: в реальности забывание происходит часами/днями/годами,
    а консолидация во время сна занимает часы. Ждать реальное время при
    демонстрации неудобно, поэтому мы вводим виртуальные часы, которые можно
    "перемотать вперёд" методом advance(), не трогая системное время.
    """

    def __init__(self):
        self._offset = 0.0  # накопленное смещение виртуального времени, сек

    def now(self) -> float:
        """Текущее виртуальное время (реальное время + накопленное смещение)."""
        return time.time() + self._offset

    def advance(self, seconds: float):
        """Промотать виртуальное время вперёд, имитируя течение часов/дней."""
        self._offset += seconds


# ============================================================================
# БЛОК 1. КОДИРОВАНИЕ ИНФОРМАЦИИ (ЭМБЕДДИНГИ)
# ============================================================================
class EmbeddingEncoder:
    """
    Превращает текст в числовой вектор (эмбеддинг) — аналог того, как
    сенсорная информация превращается в распределённый паттерн активности
    нейронов в коре головного мозга. Близкие по смыслу воспоминания получают
    близкие в векторном пространстве представления, что позволяет
    моделировать ассоциативный (а не только точный) поиск в памяти.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        if _HAS_SBERT:
            # Реальная нейросеть-трансформер, обученная строить смысловые эмбеддинги
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            # Заглушка на случай отсутствия библиотеки: простой хэш-эмбеддинг
            # по принципу "мешка слов". Он гораздо примитивнее нейросети,
            # но сохраняет общую структуру интерфейса.
            self.dim = 256
            print(
                "[EmbeddingEncoder] sentence-transformers не найден, "
                "используется упрощённый хэш-эмбеддинг."
            )

    def encode(self, text: str) -> np.ndarray:
        """Возвращает нормированный вектор-эмбеддинг текста."""
        if _HAS_SBERT:
            vec = self.model.encode(text)
            vec = np.asarray(vec, dtype=np.float32)
        else:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                idx = hash(word) % self.dim
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами — мера семантической близости."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


# ============================================================================
# БЛОК 2. СЕНСОРНАЯ ПАМЯТЬ
# ============================================================================
class SensoryMemory:
    """
    Сенсорная память — самый первый, ультракороткий буфер восприятия
    (в мозге: иконическая память для зрения ~0.5 сек, эхоическая для
    слуха ~2-4 сек). Информация тут держится доли секунд и либо
    "подхватывается" вниманием, либо безвозвратно исчезает.
    """

    def __init__(
        self, clock: VirtualClock, decay_seconds: float = 1.0, capacity: int = 5
    ):
        self.clock = clock
        self.decay_seconds = decay_seconds
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def perceive(self, stimulus: str):
        """Регистрация нового стимула в сенсорном буфере."""
        self.buffer.append({"content": stimulus, "time": self.clock.now()})
        # Ограничиваем размер буфера, отбрасывая самые старые записи
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def get_active(self) -> List[str]:
        """Возвращает стимулы, которые ещё не успели "угаснуть"."""
        now = self.clock.now()
        return [
            item["content"]
            for item in self.buffer
            if now - item["time"] <= self.decay_seconds
        ]


# ============================================================================
# БЛОК 3. РАБОЧАЯ (КРАТКОВРЕМЕННАЯ) ПАМЯТЬ
# ============================================================================
class WorkingMemory:
    """
    Рабочая память — ограниченный по объёму и времени буфер активного
    удержания информации (модель Баддели-Хитча в упрощённом виде).

    Ограничения:
      - capacity      — объём (аналог "магического числа" 7±2 / ~4 у Коуэна)
      - decay_seconds — время удержания без повторения (десятки секунд)

    Повторение (rehearsal) продлевает жизнь элемента в буфере и повышает
    шанс его перехода в долговременную память при консолидации.
    """

    def __init__(
        self, clock: VirtualClock, capacity: int = 4, decay_seconds: float = 20.0
    ):
        self.clock = clock
        self.capacity = capacity
        self.decay_seconds = decay_seconds
        self.items: List[Dict[str, Any]] = []

    def _cleanup(self):
        """Удаляет элементы, чьё время удержания истекло (пассивное забывание)."""
        now = self.clock.now()
        self.items = [it for it in self.items if now - it["time"] <= self.decay_seconds]

    def hold(self, content: str, embedding: np.ndarray, emotional_weight: float = 0.0):
        """Помещает новый элемент в рабочую память."""
        self._cleanup()
        if len(self.items) >= self.capacity:
            # Превышение ёмкости — вытеснение самого старого элемента
            # (аналог эффекта интерференции при перегрузке рабочей памяти)
            self.items.pop(0)
        self.items.append(
            {
                "content": content,
                "embedding": embedding,
                "time": self.clock.now(),
                "rehearsal": 1,  # счётчик повторений
                "emotional_weight": emotional_weight,
            }
        )

    def rehearse(self, content: str):
        """Повторение элемента — продлевает удержание и повышает счётчик."""
        self._cleanup()
        for it in self.items:
            if it["content"] == content:
                it["time"] = self.clock.now()
                it["rehearsal"] += 1

    def get_contents(self) -> List[Dict[str, Any]]:
        self._cleanup()
        return self.items


# ============================================================================
# БЛОК 4. СЛЕД ПАМЯТИ (ЭНГРАММА)
# ============================================================================
@dataclass
class MemoryTrace:
    """
    Единица долговременной памяти — аналог "энграммы", то есть физического
    следа воспоминания, закреплённого в синаптических связях нейронной сети.
    """

    id: str
    content: str
    embedding: np.ndarray
    timestamp: float  # момент создания следа
    memory_type: str  # 'episodic' | 'semantic' | 'procedural'
    emotional_weight: float = 0.0  # аналог влияния миндалевидного тела
    stability: float = 3600.0  # "S" в модели Эббингауза — устойчивость следа
    retrieval_count: int = 0  # сколько раз след извлекался
    last_access: float = field(default=0.0)
    context: Dict[str, Any] = field(default_factory=dict)

    def current_strength(self, now: float) -> float:
        """
        Кривая забывания Эббингауза: R(t) = exp(-t / S)

        R — вероятность/сила успешного вспоминания,
        t — время, прошедшее с последнего обращения к следу,
        S — устойчивость следа (растёт при повторных извлечениях —
            эффект интервального повторения / долговременной потенциации).
        """
        dt = max(0.0, now - self.last_access)
        return math.exp(-dt / max(self.stability, 1e-6))


# ============================================================================
# БЛОК 5. ДОЛГОВРЕМЕННАЯ ПАМЯТЬ
# ============================================================================
class LongTermMemory:
    """
    Долговременная память. Хранит следы (MemoryTrace) и обеспечивает:
      - запись новой информации (аналог консолидации в неокортексе);
      - ассоциативное извлечение по семантическому сходству;
      - усиление следа при каждом извлечении (реконсолидация, LTP);
      - естественное забывание слабых, невостребованных следов.
    """

    def __init__(self, encoder: EmbeddingEncoder, clock: VirtualClock):
        self.encoder = encoder
        self.clock = clock
        self.traces: Dict[str, MemoryTrace] = {}

    def store(
        self,
        content: str,
        memory_type: str = "episodic",
        emotional_weight: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> MemoryTrace:
        """Создание нового следа памяти."""
        embedding = self.encoder.encode(content)
        now = self.clock.now()
        # Эмоционально значимые события консолидируются прочнее
        # (роль миндалевидного тела в усилении памяти о значимых/опасных событиях)
        base_stability = 3600.0 * (1 + 4 * emotional_weight)
        trace = MemoryTrace(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            timestamp=now,
            memory_type=memory_type,
            emotional_weight=emotional_weight,
            stability=base_stability,
            last_access=now,
            context=context or {},
        )
        self.traces[trace.id] = trace
        return trace

    def _reinforce(self, trace: MemoryTrace):
        """
        Реконсолидация: каждое успешное извлечение делает след устойчивее.
        Биологический аналог — долговременная потенциация (LTP): чем чаще
        активируется нейронный ансамбль, тем прочнее становятся его связи.
        """
        trace.retrieval_count += 1
        trace.stability *= 1.3
        trace.last_access = self.clock.now()

    def recall(
        self, query: str, top_k: int = 5, min_strength: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Извлечение воспоминаний по запросу.

        Итоговый score = семантическое_сходство * текущая_сила_следа.
        Это моделирует то, что человек скорее вспомнит семантически близкое
        и при этом ещё не забытое (не угасшее) воспоминание.
        """
        query_emb = self.encoder.encode(query)
        now = self.clock.now()
        candidates = []
        for trace in self.traces.values():
            similarity = cosine_similarity(query_emb, trace.embedding)
            strength = trace.current_strength(now)
            if strength < min_strength:
                continue  # след слишком слаб — практически "забыт"
            candidates.append((similarity * strength, similarity, strength, trace))

        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, sim, strength, trace in candidates[:top_k]:
            self._reinforce(trace)  # сам акт вспоминания укрепляет след
            results.append(
                {
                    "content": trace.content,
                    "similarity": round(sim, 3),
                    "strength_before_recall": round(strength, 3),
                    "score": round(score, 3),
                    "retrieval_count": trace.retrieval_count,
                    "type": trace.memory_type,
                    "trace_id": trace.id,
                }
            )
        return results

    def forget_weak_traces(self, threshold: float = 0.01) -> int:
        """
        Активное забывание: следы с силой ниже порога удаляются насовсем.
        Это отражает функциональную роль забывания — предотвращение
        информационной перегрузки системы.
        """
        now = self.clock.now()
        to_delete = [
            tid
            for tid, tr in self.traces.items()
            if tr.current_strength(now) < threshold
        ]
        for tid in to_delete:
            del self.traces[tid]
        return len(to_delete)


# ============================================================================
# БЛОК 6. РЕКОНСТРУКТИВНОЕ ПРИПОМИНАНИЕ (место для интеграции LLM)
# ============================================================================
def reconstruct_memory_with_llm(query: str, retrieved: List[Dict[str, Any]]) -> str:
    """
    Важная особенность человеческой памяти: она НЕ является точной "видеозаписью"
    прошлого. Мозг реконструирует воспоминание заново при каждом вспоминании,
    опираясь на фрагменты следов и текущий контекст (реконструктивный характер
    памяти, эффект "ложных воспоминаний").

    Эта функция — точка расширения: сюда можно подключить настоящую LLM
    (например, вызов API GPT/Claude/локальной модели), которая получит
    найденные фрагменты воспоминаний и сгенерирует связный "пересказ",
    как это делает человеческий мозг. Здесь для автономности используется
    простой шаблонный вариант без внешних вызовов.
    """
    if not retrieved:
        return "Ничего не удалось вспомнить по этому запросу — след памяти утрачен."

    fragments = "; ".join(r["content"] for r in retrieved)
    return (
        f"По запросу «{query}» реконструирована следующая картина "
        f"(на основе {len(retrieved)} фрагментов памяти): {fragments}."
    )

    # ---- Пример интеграции с реальной LLM (закомментировано) ------------
    # from openai import OpenAI
    # client = OpenAI()
    # prompt = f"Восстанови связное воспоминание по запросу '{query}' " \
    #          f"на основе фрагментов: {fragments}"
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    # ------------------------------------------------------------------


# ============================================================================
# БЛОК 7. ГЛАВНЫЙ ОРКЕСТРАТОР — ПОЛНАЯ СИСТЕМА ПАМЯТИ
# ============================================================================
class HumanMemorySystem:
    """
    Объединяет все подсистемы в единый конвейер, имитирующий полный цикл
    работы человеческой памяти:

        perceive() -> attend() -> [рабочая память] -> sleep()/consolidate()
        -> [долговременная память] -> recall()
    """

    def __init__(self):
        self.clock = VirtualClock()
        self.encoder = EmbeddingEncoder()
        self.sensory = SensoryMemory(self.clock)
        self.working = WorkingMemory(self.clock)
        self.ltm = LongTermMemory(self.encoder, self.clock)

    def perceive(self, stimulus: str, emotional_weight: float = 0.0):
        """Шаг 1-2: восприятие стимула и перенос его вниманием в рабочую память."""
        self.sensory.perceive(stimulus)
        embedding = self.encoder.encode(stimulus)
        self.working.hold(stimulus, embedding, emotional_weight)

    def rehearse(self, stimulus: str):
        """Сознательное повторение информации (например, проговаривание про себя)."""
        self.working.rehearse(stimulus)

    def consolidate(self) -> List[str]:
        """
        Шаг 3: консолидация — перенос информации из рабочей памяти
        в долговременную. Вероятность переноса растёт с числом повторений
        и эмоциональной значимостью (аналог того, что происходит во сне,
        когда гиппокамп "проигрывает" дневные события неокортексу).
        """
        transferred = []
        for item in self.working.get_contents():
            transfer_prob = min(1.0, 0.2 * item["rehearsal"] + item["emotional_weight"])
            if np.random.rand() < transfer_prob:
                trace = self.ltm.store(
                    item["content"],
                    memory_type="episodic",
                    emotional_weight=item["emotional_weight"],
                )
                transferred.append(trace.content)
        return transferred

    def sleep(self, hours: float = 8.0) -> Dict[str, Any]:
        """
        Имитация сна: консолидация свежих воспоминаний + промотка времени
        вперёд + удаление слишком ослабевших следов.
        """
        transferred = self.consolidate()
        self.clock.advance(hours * 3600)  # "проматываем" ночь
        forgotten = self.ltm.forget_weak_traces()
        return {"transferred_to_ltm": transferred, "forgotten_count": forgotten}

    def recall(self, query: str, top_k: int = 5) -> str:
        """Шаг 4: извлечение и реконструктивная сборка воспоминания."""
        retrieved = self.ltm.recall(query, top_k=top_k)
        return reconstruct_memory_with_llm(query, retrieved), retrieved


# ============================================================================
# БЛОК 8. ДЕМОНСТРАЦИЯ РАБОТЫ
# ============================================================================
if __name__ == "__main__":
    memory = HumanMemorySystem()

    print("=== 1. Восприятие информации в течение дня ===")
    memory.perceive("Утром на кухне пахло свежесваренным кофе", emotional_weight=0.1)
    memory.perceive(
        "Коллега на работе рассказал важную новость о проекте", emotional_weight=0.4
    )
    memory.perceive("По дороге домой видел красную машину", emotional_weight=0.0)
    memory.perceive(
        "Пёс укусил меня за руку, было очень больно и страшно", emotional_weight=0.9
    )

    # Повторение информации усиливает шанс её консолидации
    memory.rehearse("Коллега на работе рассказал важную новость о проекте")
    memory.rehearse("Коллега на работе рассказал важную новость о проекте")

    print(
        "Активно в рабочей памяти:",
        [it["content"] for it in memory.working.get_contents()],
    )

    print("\n=== 2. Ночной сон — консолидация памяти ===")
    report = memory.sleep(hours=8)
    print("Перенесено в долговременную память:", report["transferred_to_ltm"])

    print("\n=== 3. Проходит несколько дней (симуляция) ===")
    memory.clock.advance(3600 * 24 * 5)  # +5 суток виртуального времени
    forgotten = memory.ltm.forget_weak_traces()
    print(f"Забыто (удалено) слабых следов: {forgotten}")

    print("\n=== 4. Попытка вспомнить ===")
    text, raw = memory.recall("что случилось с собакой?")
    print(text)
    print("Сырые результаты поиска:", raw)

    text2, raw2 = memory.recall("что рассказал коллега?")
    print("\n" + text2)
    print("Сырые результаты поиска:", raw2)
