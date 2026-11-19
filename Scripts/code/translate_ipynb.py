import argparse
import os
import re
from typing import Dict, List, Tuple
import nbformat
from openai import OpenAI, OpenAIError
import pyperclip

# =====================================================================
# СИСТЕМНЫЙ ПРОМПТ
# =====================================================================
SYSTEM_PROMPT = """Ты — опытный Senior Python-разработчик и талантливый преподаватель.
Тебе будет дана часть содержимого Jupyter Notebook (IPYNB), разбитая на ячейки с помощью тегов <CELL id="N" type="TYPE">...</CELL>.

Твоя главная цель — адаптировать этот ноутбук для русскоговорящего Junior Python-программиста, который не до конца понимает контекст происходящего.

Твои задачи:
1. ДЛЯ ЯЧЕЕК MARKDOWN (type="markdown"):
   - Переведи весь текст на русский язык, сохраняя оригинальное форматирование (заголовки, списки, жирный шрифт, ссылки).
   - КРИТИЧЕСКИ ВАЖНО: Ни в коем случае не изменяй математические формулы LaTeX (внутри $...$ и $$...$$) и блоки кода. Они должны остаться в точности как в оригинале.
   - Если в ячейке описывается сложная концепция, добавь в конец ячейки блок от себя:
     > 💡 **Пояснение для Junior:** [твое понятное объяснение простыми словами]

2. ДЛЯ ЯЧЕЕК КОДА (type="code"):
   - НЕ ИЗМЕНЯЙ сам исполняемый код (названия переменных, функции, логику). Код должен работать точно так же, как до твоего вмешательства.
   - Переведи все существующие комментарии (#) и docstrings (\"\"\"...\"\"\") на русский язык.
   - Добавь свои подробные объясняющие комментарии (# 💡 Пояснение: ...) перед сложными строками или блоками кода. Объясняй НЕ ЧТО делает код (например, "присваиваем x = 5"), а ЗАЧЕМ это делается и КАК работают неочевидные конструкции/библиотеки.

СТРОГИЕ ПРАВИЛА ФОРМАТА ОТВЕТА:
- Верни ТОЛЬКО переведенные ячейки в точно таком же формате тегов:
<CELL id="N" type="TYPE">
новое содержимое
</CELL>
- Обязательно сохраняй исходные номера `id` (даже если они начинаются не с нуля) и `type` каждой ячейки!
- Не пропускай ячейки и не объединяй их.
- Не пиши никакого вступительного или заключительного текста вне тегов <CELL>."""


def create_chunks(
    nb: nbformat.NotebookNode, max_chars: int = 6000
) -> List[Tuple[List[int], str]]:
    """Разбивает ноутбук на части (чанки) для обработки больших файлов.

    Возвращает список кортежей: (список_ожидаемых_id_ячеек, текст_чанка).
    """
    chunks = []
    current_chunk_blocks = []
    current_chunk_ids = []
    current_length = 0

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type not in ["markdown", "code"] or not cell.source.strip():
            continue

        cell_block = (
            f'<CELL id="{idx}" type="{cell.cell_type}">\n' f"{cell.source}\n" f"</CELL>"
        )
        block_len = len(cell_block)

        if current_chunk_blocks and (current_length + block_len > max_chars):
            chunks.append((current_chunk_ids, "\n\n".join(current_chunk_blocks)))
            current_chunk_blocks = [cell_block]
            current_chunk_ids = [idx]
            current_length = block_len
        else:
            current_chunk_blocks.append(cell_block)
            current_chunk_ids.append(idx)
            current_length += block_len

    if current_chunk_blocks:
        chunks.append((current_chunk_ids, "\n\n".join(current_chunk_blocks)))

    return chunks


def parse_llm_response(llm_response: str) -> Dict[int, Tuple[str, str]]:
    """Парсит теги <CELL> из ответа LLM.

    Возвращает словарь: {cell_id: (cell_type, new_source)}
    """
    pattern = re.compile(
        r'<CELL\s+id="(\d+)"\s+type="(markdown|code)">\n?(.*?)\n?</CELL>',
        re.DOTALL,
    )
    matches = pattern.findall(llm_response)

    parsed_cells = {}
    for cell_id_str, cell_type, new_source in matches:
        parsed_cells[int(cell_id_str)] = (cell_type, new_source.strip())

    return parsed_cells


def process_chunk_manual(
    chunk_idx: int,
    total_chunks: int,
    expected_ids: List[int],
    chunk_text: str,
) -> Dict[int, Tuple[str, str]]:
    """Режим по умолчанию (arena.ai): работа через буфер обмена."""
    full_prompt = f"{SYSTEM_PROMPT}\n\nВОТ ЯЧЕЙКИ ДЛЯ ОБРАБОТКИ:\n\n{chunk_text}"

    while True:
        pyperclip.copy(full_prompt)

        print("\n" + "=" * 65)
        print(
            f"📦 [РУЧНОЙ РЕЖИМ] ЧАСТЬ {chunk_idx}/{total_chunks} | Ячейки: {expected_ids[0]}..{expected_ids[-1]} ({len(expected_ids)} шт.)"
        )
        print(f"📋 Промпт ({len(full_prompt)} симв.) СКОПИРОВАН В БУФЕР ОБМЕНА!")
        print("=" * 65)
        print("Ваши действия:")
        print("  1. Вставьте промпт (Ctrl+V) в чат на arena.ai и отправьте.")
        print("  2. Дождитесь ответа и нажмите кнопку 'Copy' под ответом модели.")

        input("\n👉 Когда скопируете ответ модели, нажмите [ENTER] здесь...")

        response = pyperclip.paste()

        if response.strip() == full_prompt.strip() or not response.strip():
            print("\n❌ [ОШИБКА] В буфере обмена всё ещё старый промпт (или пусто)!")
            input("Скопируйте ответ модели на сайте и нажмите Enter...")
            continue

        parsed_cells = parse_llm_response(response)

        if not parsed_cells:
            print("\n❌ [ОШИБКА] В скопированном тексте не найдено тегов <CELL>!")
            print(f"Содержимое буфера (первые 200 симв.):\n{response[:200]}...")
            input(
                "\nНажмите Enter, чтобы заново скопировать промпт в буфер и повторить..."
            )
            continue

        missing_ids = [cid for cid in expected_ids if cid not in parsed_cells]
        if missing_ids:
            print(
                f"\n⚠️ [ВНИМАНИЕ] Модель вернула не все ячейки! Пропущены ID: {missing_ids}"
            )
            choice = (
                input(
                    "Действие: [R] - Повторить эту часть / [I] - Сохранить как есть: "
                )
                .strip()
                .lower()
            )
            if choice != "i":
                continue

        print(
            f"✅ Успешно обработано ячеек в этой части: {len(parsed_cells)}/{len(expected_ids)}"
        )
        return parsed_cells


def process_chunk_api(
    client: OpenAI,
    model: str,
    chunk_idx: int,
    total_chunks: int,
    expected_ids: List[int],
    chunk_text: str,
) -> Dict[int, Tuple[str, str]]:
    """Режим API (OpenAI / OpenRouter / DeepSeek) с поддержкой отказоустойчивости и фоллбэка на ручной режим."""
    while True:
        print(
            f"\n[*] 🤖 [API РЕЖИМ] Отправка части {chunk_idx}/{total_chunks} "
            f"(ячейки {expected_ids[0]}..{expected_ids[-1]}, ~{len(chunk_text)} симв.) в модель {model}..."
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chunk_text},
                ],
                temperature=0.2,
            )
            raw_content = response.choices[0].message.content
            parsed_cells = parse_llm_response(raw_content)

            if not parsed_cells:
                print("❌ [ОШИБКА API] Модель не вернула теги <CELL>.")
            else:
                missing_ids = [cid for cid in expected_ids if cid not in parsed_cells]
                if not missing_ids:
                    print(
                        f"✅ Успешно получено от API ячеек: {len(parsed_cells)}/{len(expected_ids)}"
                    )
                    return parsed_cells
                else:
                    print(
                        f"⚠️ [ВНИМАНИЕ] API вернул не все ячейки! Пропущены ID: {missing_ids}"
                    )

        except OpenAIError as e:
            print(f"\n❌ [ОШИБКА OPENAI API]: {e}")
            parsed_cells = {}

        # Если произошла ошибка API или вернулись не все ячейки, спрашиваем пользователя:
        print("\nВыберите действие для этой части:")
        print("  [R] - Повторить запрос по API")
        print("  [M] - Переключиться на РУЧНОЙ режим (arena.ai) для этой части")
        if parsed_cells:
            print("  [I] - Игнорировать пропуски и сохранить то, что пришло")

        choice = input("Ваш выбор [R/M/I]: ").strip().lower()
        if choice == "m":
            return process_chunk_manual(
                chunk_idx, total_chunks, expected_ids, chunk_text
            )
        elif choice == "i" and parsed_cells:
            return parsed_cells


def main():
    parser = argparse.ArgumentParser(
        description="Перевод и объяснение IPYNB ноутбуков (через буфер обмена для arena.ai ИЛИ через OpenAI API)."
    )
    parser.add_argument("input", help="Путь к исходному .ipynb файлу")
    parser.add_argument(
        "-o", "--output", help="Путь для сохранения нового .ipynb файла"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=60000,
        help="Максимальный размер одной части (чанка) в символах (по умолчанию: 6000)",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=1,
        help="С какой по счету части начать (полезно, если прервали большой файл)",
    )

    # Настройки для режима API (дополнение к основному режиму)
    api_group = parser.add_argument_group("Настройки режима API (OpenAI)")
    api_group.add_argument(
        "--api",
        action="store_true",
        help="Использовать OpenAI API вместо ручного режима (по умолчанию выключено)",
    )
    api_group.add_argument(
        "--api-key",
        default=None,
        help="API ключ (если не указан, берется из переменной окружения OPENAI_API_KEY)",
    )
    api_group.add_argument(
        "--model",
        default="gpt-4o",
        help="Имя модели для API (по умолчанию: gpt-4o)",
    )
    api_group.add_argument(
        "--base-url",
        default=None,
        help="Кастомный URL для API (OpenRouter, DeepSeek, локальная LLM и т.д.)",
    )

    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_ru_explained{ext}"

    # Инициализация клиента OpenAI, если выбран режим --api
    openai_client = None
    if args.api:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Для режима --api необходим ключ! Передайте его через --api-key "
                "или установите переменную окружения OPENAI_API_KEY."
            )
        openai_client = OpenAI(api_key=api_key, base_url=args.base_url)
        print(f"[*] Активирован режим API (Модель: {args.model})")
    else:
        print("[*] Активирован ручной режим через буфер обмена (для arena.ai)")

    # 1. Читаем ноутбук
    print(f"[*] Чтение файла: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # 2. Разбиваем на части (чанки)
    chunks = create_chunks(nb, max_chars=args.chunk_size)
    total_chunks = len(chunks)
    print(
        f"[*] Ноутбук разбит на частей: {total_chunks} (макс. ~{args.chunk_size} симв. в части)"
    )

    # 3. Обрабатываем каждую часть
    total_updated = 0
    for idx, (expected_ids, chunk_text) in enumerate(chunks, start=1):
        if idx < args.start_chunk:
            print(f"[*] Пропуск части {idx}/{total_chunks} (--start-chunk)")
            continue

        if args.api:
            parsed_cells = process_chunk_api(
                client=openai_client,
                model=args.model,
                chunk_idx=idx,
                total_chunks=total_chunks,
                expected_ids=expected_ids,
                chunk_text=chunk_text,
            )
        else:
            parsed_cells = process_chunk_manual(
                chunk_idx=idx,
                total_chunks=total_chunks,
                expected_ids=expected_ids,
                chunk_text=chunk_text,
            )

        # Обновляем ячейки в ноутбуке
        for cell_id, (cell_type, new_source) in parsed_cells.items():
            if 0 <= cell_id < len(nb.cells):
                nb.cells[cell_id].source = new_source
                total_updated += 1

        # Сохраняем результат после каждого чанка
        with open(args.output, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"💾 Промежуточный результат сохранен в: {args.output}")

    print("\n" + "🎉" * 20)
    print(f"ГОТОВО! Обновлено ячеек: {total_updated}.")
    print(f"Итоговый файл сохранен: {args.output}")


if __name__ == "__main__":
    main()
