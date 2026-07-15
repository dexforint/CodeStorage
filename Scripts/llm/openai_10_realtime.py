import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def realtime_voice_session():
    """Голосовой агент в реальном времени."""
    async with client.realtime.connect(model="gpt-4o-realtime-preview") as conn:

        # Настраиваем сессию
        await conn.session.update(
            session={
                "modalities": ["text", "audio"],
                "voice": "nova",
                "instructions": "Ты полезный ассистент. Отвечай кратко.",
                "turn_detection": {"type": "server_vad"},  # Авто-детекция паузы
            }
        )

        # Отправляем текстовое сообщение
        await conn.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Расскажи анекдот."}],
            }
        )

        await conn.response.create()

        # Читаем события
        async for event in conn:
            if event.type == "response.text.delta":
                print(event.delta, end="", flush=True)

            elif event.type == "response.audio.delta":
                # Здесь можно воспроизводить аудио в реальном времени
                pass

            elif event.type == "response.done":
                print("\n✅ Ответ получен")
                break

            elif event.type == "error":
                print(f"❌ {event.error.message}")
                break


asyncio.run(realtime_voice_session())
