#!/usr/bin/env python3
"""
Модуль для транскрибации видео/аудио с разделением речи по говорящим.

Использует передовые нейронные сети:
- Whisper (OpenAI) для распознавания речи
- PyAnnote для диаризации спикеров

Требования:
- Python 3.8+
- Установленные зависимости (см. requirements.txt)
- CUDA-совместимый GPU (опционально, для ускорения)
"""

import argparse
import logging
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Audio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation, Segment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

# Подавление предупреждений для чистоты вывода
warnings.filterwarnings("ignore")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class TranscriptionSegment:
    """
    Класс для хранения сегмента транскрибации.

    Attributes:
        speaker (str): Идентификатор говорящего
        start_time (float): Время начала речи в секундах
        end_time (float): Время окончания речи в секундах
        text (str): Транскрибированный текст
    """

    speaker: str
    start_time: float
    end_time: float
    text: str

    def __str__(self) -> str:
        """Форматированное представление сегмента."""
        return f"[{self.start_time:.2f}s - {self.end_time:.2f}s] {self.speaker}: {self.text}"


class VideoTranscriber:
    """
    Основной класс для транскрибации видео с диаризацией спикеров.

    Объединяет Whisper для распознавания речи и PyAnnote для разделения спикеров.
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-large-v3",
        diarization_model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "auto",
        language: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Инициализация транскрайбера.

        Args:
            whisper_model_name: Название модели Whisper для распознавания речи
            diarization_model_name: Название модели PyAnnote для диаризации
            device: Устройство для вычислений ('auto', 'cpu', 'cuda', 'mps')
            language: Язык аудио (None для автоопределения)
            hf_token: Токен Hugging Face для доступа к моделям

        Raises:
            RuntimeError: Если не удалось загрузить модели
        """
        self.device = self._setup_device(device)
        self.language = language
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if not self.hf_token:
            logger.warning(
                "Токен Hugging Face не предоставлен. "
                "Установите переменную окружения HF_TOKEN или передайте через параметр."
            )

        logger.info(f"Инициализация моделей на устройстве: {self.device}")

        # Проверка наличия ffmpeg для обработки видео
        self._check_ffmpeg()

        # Загрузка моделей
        self._load_whisper_model(whisper_model_name)
        self._load_diarization_model(diarization_model_name)

    def _check_ffmpeg(self) -> None:
        """
        Проверка наличия ffmpeg в системе.
        """
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            logger.info("FFmpeg найден в системе")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(
                "FFmpeg не найден в системе. "
                "Рекомендуется установить ffmpeg для лучшей совместимости:\n"
                "Windows: https://ffmpeg.org/download.html\n"
                "Или через Chocolatey: choco install ffmpeg"
            )

    def _setup_device(self, device: str) -> str:
        """Настройка устройства для вычислений."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA недоступен, переключение на CPU")
            return "cpu"

        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS недоступен, переключение на CPU")
            return "cpu"

        return device

    def _load_whisper_model(self, model_name: str) -> None:
        """Загрузка модели Whisper для распознавания речи."""
        try:
            logger.info(f"Загрузка модели Whisper: {model_name}")

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=self.hf_token,
            )
            self.whisper_model.to(self.device)

            self.whisper_processor = AutoProcessor.from_pretrained(
                model_name, token=self.hf_token
            )

            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model,
                tokenizer=self.whisper_processor.tokenizer,
                feature_extractor=self.whisper_processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=self.device,
            )

            logger.info("Модель Whisper успешно загружена")

        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели Whisper: {str(e)}")

    def _load_diarization_model(self, model_name: str) -> None:
        """Загрузка модели для диаризации спикеров."""
        try:
            logger.info(f"Загрузка модели диаризации: {model_name}")

            # Пробуем разные названия параметра для совместимости с версиями
            try:
                self.diarization_pipeline = SpeakerDiarization.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
            except TypeError:
                self.diarization_pipeline = SpeakerDiarization.from_pretrained(
                    model_name, token=self.hf_token
                )

            # Перемещение модели на нужное устройство
            if self.device != "cpu":
                self.diarization_pipeline.to(torch.device(self.device))

            logger.info("Модель диаризации успешно загружена")

        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели диаризации: {str(e)}")

    def extract_audio(self, video_path: Path) -> Path:
        """Извлечение аудио из видеофайла в формат WAV."""
        logger.info(f"Извлечение аудио из видео: {video_path}")

        temp_audio_path = video_path.parent / f"{video_path.stem}_temp_audio.wav"

        try:
            logger.info("Извлечение аудио через ffmpeg...")
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(temp_audio_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Аудио успешно извлечено через ffmpeg")

            if not temp_audio_path.exists() or temp_audio_path.stat().st_size == 0:
                raise RuntimeError("Создан пустой аудиофайл")

            return temp_audio_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка ffmpeg: {e.stderr}")
            raise RuntimeError(f"Не удалось извлечь аудио через ffmpeg: {e}")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg не найден. Установите ffmpeg:\n"
                "Windows: choco install ffmpeg или https://ffmpeg.org/download.html"
            )

    def load_audio_safe(self, audio_path: Path):
        """
        Безопасная загрузка аудио через soundfile.
        """
        import soundfile as sf

        logger.info("Загрузка аудио через soundfile...")
        data, sample_rate = sf.read(str(audio_path), dtype="float32")

        # Конвертируем в torch тензор
        if len(data.shape) == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)

        return waveform, sample_rate

    def perform_diarization(self, audio_path: Path) -> Annotation:
        """
        Выполнение диаризации аудио для определения спикеров.

        Args:
            audio_path: Путь к аудиофайлу

        Returns:
            Annotation: Результаты диаризации с временными метками спикеров
        """
        logger.info("Выполнение диаризации спикеров...")

        # Загружаем аудио безопасным способом
        waveform, sample_rate = self.load_audio_safe(audio_path)

        # Создаем словарь с данными для пайплайна
        audio_data = {"waveform": waveform, "sample_rate": sample_rate}

        # Запуск диаризации с индикатором прогресса
        logger.info("Запуск пайплайна диаризации...")
        with ProgressHook() as hook:
            result = self.diarization_pipeline(audio_data, hook=hook)

        # Извлекаем аннотацию из результата в зависимости от версии pyannote
        if hasattr(result, "speaker_diarization"):
            # Новая версия (DiarizeOutput) - используем speaker_diarization
            diarization = result.speaker_diarization
            logger.debug("Использован result.speaker_diarization")
        elif hasattr(result, "annotation"):
            # Старая версия - annotation
            diarization = result.annotation
            logger.debug("Использован result.annotation")
        elif isinstance(result, Annotation):
            # Ещё более старая версия - возвращает Annotation напрямую
            diarization = result
            logger.debug("Результат напрямую является Annotation")
        else:
            raise RuntimeError(
                f"Неизвестный тип результата диаризации: {type(result)}\n"
                f"Доступные атрибуты: {dir(result)}"
            )

        # Получение количества обнаруженных спикеров
        speakers = diarization.labels()
        logger.info(f"Обнаружено спикеров: {len(speakers)}")

        return diarization

    def transcribe_segment(self, audio_path: Path, segment: Segment) -> str:
        """
        Транскрибация отдельного сегмента аудио.

        Args:
            audio_path: Путь к аудиофайлу
            segment: Временной сегмент для транскрибации

        Returns:
            str: Транскрибированный текст
        """
        import soundfile as sf

        # Загружаем аудио
        waveform, sample_rate = self.load_audio_safe(audio_path)

        # Вырезаем сегмент
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)

        # Убеждаемся, что индексы в пределах допустимого
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)

        if end_sample <= start_sample:
            return ""

        segment_waveform = waveform[:, start_sample:end_sample]

        # Конвертируем в numpy для сохранения через soundfile
        segment_numpy = segment_waveform.squeeze().numpy()

        # Временное сохранение сегмента через soundfile
        temp_segment_path = audio_path.parent / f"temp_segment_{hash(segment)}.wav"
        sf.write(str(temp_segment_path), segment_numpy, sample_rate, subtype="PCM_16")

        # Транскрибация сегмента
        try:
            result = self.whisper_pipeline(
                str(temp_segment_path),
                generate_kwargs={"language": self.language} if self.language else {},
                return_timestamps=False,
            )
            text = result["text"].strip()
        finally:
            # Очистка временного файла
            if temp_segment_path.exists():
                temp_segment_path.unlink()

        return text

    def merge_transcription_and_diarization(
        self, audio_path: Path, diarization: Annotation
    ) -> List[TranscriptionSegment]:
        """
        Объединение результатов транскрибации и диаризации.
        """
        logger.info("Объединение результатов транскрибации и диаризации...")

        # Загружаем аудио один раз
        waveform, sample_rate = self.load_audio_safe(audio_path)

        segments = []

        # Итерация по всем речевым сегментам
        for segment, _, speaker in tqdm(
            diarization.itertracks(yield_label=True), desc="Транскрибация сегментов"
        ):
            # Транскрибация текущего сегмента
            text = self.transcribe_segment_from_waveform(
                waveform, sample_rate, segment, audio_path.parent
            )

            if text:
                transcription_segment = TranscriptionSegment(
                    speaker=speaker,
                    start_time=segment.start,
                    end_time=segment.end,
                    text=text,
                )
                segments.append(transcription_segment)

        segments.sort(key=lambda x: x.start_time)
        return segments

    def transcribe_segment_from_waveform(
        self, waveform: torch.Tensor, sample_rate: int, segment: Segment, temp_dir: Path
    ) -> str:
        """
        Транскрибация сегмента из предзагруженного waveform.
        Передает аудиоданные напрямую в pipeline, минуя torchcodec.
        """
        # Вырезаем сегмент
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)

        if end_sample <= start_sample:
            return ""

        segment_waveform = waveform[:, start_sample:end_sample]

        # Конвертируем в numpy массив для передачи в pipeline
        # Whisper ожидает float32 массив
        segment_numpy = segment_waveform.squeeze().numpy().astype(np.float32)

        try:
            # Передаем напрямую numpy массив вместо пути к файлу
            result = self.whisper_pipeline(
                segment_numpy,
                generate_kwargs={"language": self.language} if self.language else {},
                return_timestamps=False,
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Ошибка транскрибации сегмента: {e}")
            return ""

    def transcribe_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        delete_audio: bool = True,
    ) -> List[TranscriptionSegment]:
        """
        Полный пайплайн транскрибации видео с диаризацией.

        Args:
            video_path: Путь к видео или аудио файлу
            output_path: Путь для сохранения результатов (опционально)
            delete_audio: Удалять ли временный аудиофайл после обработки

        Returns:
            List[TranscriptionSegment]: Список сегментов транскрибации
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Файл не найден: {video_path}")

        # Проверка расширения файла
        audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
        is_audio = video_path.suffix.lower() in audio_extensions
        is_video = video_path.suffix.lower() in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
        }

        if not (is_audio or is_video):
            raise ValueError(
                f"Неподдерживаемый формат файла: {video_path.suffix}. "
                f"Поддерживаются видеоформаты и аудиоформаты: {audio_extensions}"
            )

        try:
            # Извлечение аудио если это видео
            if is_video:
                audio_path = self.extract_audio(video_path)
            else:
                audio_path = video_path
                delete_audio = False

            # Диаризация спикеров
            diarization = self.perform_diarization(audio_path)

            # Объединение с транскрибацией
            segments = self.merge_transcription_and_diarization(audio_path, diarization)

            # Сохранение результатов
            if output_path:
                self.save_results(segments, Path(output_path))

            return segments

        finally:
            # Очистка временных файлов
            if delete_audio and "audio_path" in locals() and audio_path != video_path:
                audio_path.unlink(missing_ok=True)
                logger.debug(f"Временный файл удален: {audio_path}")

    def save_results(
        self, segments: List[TranscriptionSegment], output_path: Path
    ) -> None:
        """Сохранение результатов транскрибации в файл."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        is_txt = output_path.suffix.lower() == ".txt"
        is_srt = output_path.suffix.lower() == ".srt"

        if is_srt:
            self._save_srt(segments, output_path)
        else:
            self._save_txt(segments, output_path)

        logger.info(f"Результаты сохранены в: {output_path}")

    def _save_txt(
        self, segments: List[TranscriptionSegment], output_path: Path
    ) -> None:
        """Сохранение в текстовом формате."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Транскрибация видео с разделением по спикерам\n")
            f.write("=" * 50 + "\n\n")

            for segment in segments:
                f.write(str(segment) + "\n")

    def _save_srt(
        self, segments: List[TranscriptionSegment], output_path: Path
    ) -> None:
        """Сохранение в формате SRT субтитров."""
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, segment in enumerate(segments, 1):
                start_srt = self._format_time_srt(segment.start_time)
                end_srt = self._format_time_srt(segment.end_time)

                f.write(f"{idx}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{segment.speaker}: {segment.text}\n\n")

    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        """Форматирование времени для SRT формата."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millisecs = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millisecs:03d}"


def main():
    """Главная функция для обработки аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Транскрибация видео с разделением речи по спикерам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Базовая транскрибация
  python transcribe.py video.mp4
  
  # С сохранением в файл
  python transcribe.py video.mp4 -o transcript.txt
  
  # С указанием языка
  python transcribe.py video.mp4 -l ru -o transcript.srt
  
  # Использование GPU
  python transcribe.py video.mp4 --device cuda
        """,
    )

    parser.add_argument("input", type=str, help="Путь к входному видео или аудио файлу")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Путь к выходному файлу"
    )
    parser.add_argument("-l", "--language", type=str, default=None, help="Язык аудио")
    parser.add_argument(
        "--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-large-v3")
    parser.add_argument(
        "--diarization-model", type=str, default="pyannote/speaker-diarization-3.1"
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        transcriber = VideoTranscriber(
            whisper_model_name=args.whisper_model,
            diarization_model_name=args.diarization_model,
            device=args.device,
            language=args.language,
            hf_token=args.hf_token,
        )

        segments = transcriber.transcribe_video(
            video_path=args.input,
            output_path=args.output,
            delete_audio=not args.keep_audio,
        )

        if not args.output:
            print("\n" + "=" * 50)
            print("РЕЗУЛЬТАТЫ ТРАНСКРИБАЦИИ:")
            print("=" * 50)
            for segment in segments:
                print(segment)

    except Exception as e:
        logger.error(f"Ошибка при выполнении транскрибации: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
