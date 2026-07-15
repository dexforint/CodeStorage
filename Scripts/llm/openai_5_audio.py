from openai import OpenAI
from pathlib import Path

client = OpenAI()


# ─── Speech-to-Text (Whisper) ─────────────────────────────────────────────────
def transcribe(audio_path: str, language: str = "ru") -> str:
    """Транскрибирует аудио в текст."""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,  # Явное указание языка — точнее и быстрее
            response_format="text",  # "text" / "json" / "srt" / "vtt" / "verbose_json"
            prompt="Транскрипция беседы о Python и машинном обучении.",  # Подсказка
            temperature=0,
        )
    return transcript


# ─── Перевод аудио на английский ─────────────────────────────────────────────
def translate_audio(audio_path: str) -> str:
    """Переводит аудио на любом языке в английский текст."""
    with open(audio_path, "rb") as audio_file:
        translation = client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
        )
    return translation.text


# ─── Text-to-Speech ───────────────────────────────────────────────────────────
def text_to_speech(
    text: str,
    output_path: str = "speech.mp3",
    voice: str = "nova",
) -> None:
    """
    Доступные голоса: alloy, ash, ballad, coral, echo, fable,
                      nova, onyx, sage, shimmer
    """
    response = client.audio.speech.create(
        model="tts-1",  # "tts-1" (быстрый) или "tts-1-hd" (качественный)
        voice=voice,
        input=text,
        response_format="mp3",  # "mp3" / "opus" / "aac" / "flac" / "wav" / "pcm"
        speed=1.0,  # 0.25 – 4.0
    )
    response.stream_to_file(output_path)
    print(f"✅ Аудио сохранено: {output_path}")


text_to_speech("Привет! Это синтез речи от OpenAI.", voice="nova")
