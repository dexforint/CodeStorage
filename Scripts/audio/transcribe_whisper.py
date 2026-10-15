#!/usr/bin/env python3
"""
python audio/transcribe1.py "C:\Users\user\Downloads\AI архитектор\[SuperSliv.biz] 02 Проектирование и оценка - от требований к плану, рискам и смете\02 Проектирование и оценка [Supersliv.biz].mp4" --device cuda -l ru -o transcript.txt

Модуль для транскрибации видео/аудио с разделением речи по говорящим.

Использует:
- Whisper (OpenAI / Hugging Face) для распознавания речи
- PyAnnote для диаризации спикеров

Whisper вызывается напрямую (processor + model.generate), без
transformers.pipeline и без torchcodec — это обходит типичную поломку
на Windows (libtorchcodec_core*.dll).
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
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation, Segment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

WHISPER_SR = 16000
WHISPER_MAX_SECONDS = 30
MIN_SEGMENT_SECONDS = 0.1


@dataclass
class TranscriptionSegment:
    """Сегмент транскрибации: спикер, таймкоды, текст."""

    speaker: str
    start_time: float
    end_time: float
    text: str

    def __str__(self) -> str:
        return (
            f"[{self.start_time:.2f}s - {self.end_time:.2f}s] "
            f"{self.speaker}: {self.text}"
        )


def resample_audio(
    audio: np.ndarray, sample_rate: int, target_sr: int = WHISPER_SR
) -> np.ndarray:
    """Линейный ресемпл без torchcodec/torchaudio."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or sample_rate == target_sr:
        return audio

    duration = len(audio) / float(sample_rate)
    new_length = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class VideoTranscriber:
    """Транскрибация видео/аудио с диаризацией спикеров."""

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-large-v3",
        diarization_model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "auto",
        language: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.device = self._setup_device(device)
        self.language = language
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if not self.hf_token:
            logger.warning(
                "Токен Hugging Face не предоставлен. "
                "Установите HF_TOKEN или передайте --hf-token."
            )

        logger.info(f"Инициализация моделей на устройстве: {self.device}")

        self._check_ffmpeg()
        self._load_whisper_model(whisper_model_name)
        self._load_diarization_model(diarization_model_name)

    def _check_ffmpeg(self) -> None:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            logger.info("FFmpeg найден в системе")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(
                "FFmpeg не найден. Для извлечения аудио из видео установите ffmpeg:\n"
                "Windows: https://ffmpeg.org/download.html\n"
                "Или: choco install ffmpeg"
            )

    def _setup_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA недоступен, переключение на CPU")
            return "cpu"

        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS недоступен, переключение на CPU")
            return "cpu"

        return device

    def _load_whisper_model(self, model_name: str) -> None:
        """Загрузка Whisper без ASR-pipeline (без torchcodec)."""
        try:
            logger.info(f"Загрузка модели Whisper: {model_name}")

            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=self.hf_token,
            )
            self.whisper_model.to(self.device)
            self.whisper_model.eval()

            self.whisper_processor = AutoProcessor.from_pretrained(
                model_name, token=self.hf_token
            )

            logger.info("Модель Whisper успешно загружена")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели Whisper: {e}") from e

    def _load_diarization_model(self, model_name: str) -> None:
        try:
            logger.info(f"Загрузка модели диаризации: {model_name}")

            try:
                self.diarization_pipeline = SpeakerDiarization.from_pretrained(
                    model_name, use_auth_token=self.hf_token
                )
            except TypeError:
                self.diarization_pipeline = SpeakerDiarization.from_pretrained(
                    model_name, token=self.hf_token
                )

            if self.device != "cpu":
                self.diarization_pipeline.to(torch.device(self.device))

            logger.info("Модель диаризации успешно загружена")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели диаризации: {e}") from e

    def extract_audio(self, video_path: Path) -> Path:
        """Извлечение моно WAV 16 kHz через ffmpeg."""
        logger.info(f"Извлечение аудио из видео: {video_path}")
        temp_audio_path = video_path.parent / f"{video_path.stem}_temp_audio.wav"

        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(WHISPER_SR),
                "-ac",
                "1",
                "-y",
                str(temp_audio_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Аудио успешно извлечено через ffmpeg")

            if not temp_audio_path.exists() or temp_audio_path.stat().st_size == 0:
                raise RuntimeError("Создан пустой аудиофайл")

            return temp_audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка ffmpeg: {e.stderr}")
            raise RuntimeError(f"Не удалось извлечь аудио через ffmpeg: {e}") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "FFmpeg не найден. Установите ffmpeg:\n"
                "Windows: choco install ffmpeg или https://ffmpeg.org/download.html"
            ) from e

    def load_audio_safe(self, audio_path: Path):
        """Загрузка аудио через soundfile (без torchcodec)."""
        import soundfile as sf

        logger.info("Загрузка аудио через soundfile...")
        data, sample_rate = sf.read(str(audio_path), dtype="float32")

        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            # soundfile: (samples, channels) -> (channels, samples)
            waveform = torch.from_numpy(data.T)

        return waveform, sample_rate

    def perform_diarization(self, audio_path: Path) -> Annotation:
        logger.info("Выполнение диаризации спикеров...")

        waveform, sample_rate = self.load_audio_safe(audio_path)
        audio_data = {"waveform": waveform, "sample_rate": sample_rate}

        logger.info("Запуск пайплайна диаризации...")
        with ProgressHook() as hook:
            result = self.diarization_pipeline(audio_data, hook=hook)

        if hasattr(result, "speaker_diarization"):
            diarization = result.speaker_diarization
        elif hasattr(result, "annotation"):
            diarization = result.annotation
        elif isinstance(result, Annotation):
            diarization = result
        else:
            raise RuntimeError(
                f"Неизвестный тип результата диаризации: {type(result)}\n"
                f"Доступные атрибуты: {dir(result)}"
            )

        speakers = diarization.labels()
        logger.info(f"Обнаружено спикеров: {len(speakers)}")
        return diarization

    def _prepare_audio_chunk(
        self, waveform: torch.Tensor, sample_rate: int, segment: Segment
    ) -> np.ndarray:
        start_sample = max(0, int(segment.start * sample_rate))
        end_sample = min(waveform.shape[-1], int(segment.end * sample_rate))

        if end_sample <= start_sample:
            return np.array([], dtype=np.float32)

        audio = waveform[..., start_sample:end_sample].detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        return resample_audio(audio, sample_rate, WHISPER_SR)

    def _transcribe_numpy(self, audio: np.ndarray) -> str:
        """Прямой вызов Whisper по numpy float32 @ 16 kHz."""
        if audio.size == 0:
            return ""

        max_samples = WHISPER_SR * WHISPER_MAX_SECONDS
        chunks = [
            audio[i : i + max_samples]
            for i in range(0, len(audio), max_samples)
            if audio[i : i + max_samples].size > 0
        ]
        if not chunks:
            return ""

        generate_kwargs = {"task": "transcribe", "do_sample": False}
        if self.language:
            generate_kwargs["language"] = self.language

        texts: List[str] = []
        for chunk in chunks:
            inputs = self.whisper_processor(
                chunk,
                sampling_rate=WHISPER_SR,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(
                self.device, dtype=self.torch_dtype
            )

            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(
                    input_features,
                    **generate_kwargs,
                )

            text = self.whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            if text:
                texts.append(text)

        return " ".join(texts)

    def transcribe_segment_from_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segment: Segment,
        temp_dir: Path,
    ) -> str:
        if (segment.end - segment.start) < MIN_SEGMENT_SECONDS:
            return ""

        audio = self._prepare_audio_chunk(waveform, sample_rate, segment)
        try:
            return self._transcribe_numpy(audio)
        except Exception as e:
            logger.error(f"Ошибка транскрибации сегмента {segment}: {e}")
            return ""

    def merge_transcription_and_diarization(
        self, audio_path: Path, diarization: Annotation
    ) -> List[TranscriptionSegment]:
        logger.info("Объединение результатов транскрибации и диаризации...")

        waveform, sample_rate = self.load_audio_safe(audio_path)
        tracks = list(diarization.itertracks(yield_label=True))
        segments: List[TranscriptionSegment] = []

        for segment, _, speaker in tqdm(tracks, desc="Транскрибация сегментов"):
            text = self.transcribe_segment_from_waveform(
                waveform, sample_rate, segment, audio_path.parent
            )
            if text:
                segments.append(
                    TranscriptionSegment(
                        speaker=speaker,
                        start_time=segment.start,
                        end_time=segment.end,
                        text=text,
                    )
                )

        segments.sort(key=lambda x: x.start_time)
        logger.info(f"Готовых сегментов с текстом: {len(segments)} из {len(tracks)}")
        return segments

    def transcribe_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        delete_audio: bool = True,
    ) -> List[TranscriptionSegment]:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Файл не найден: {video_path}")

        audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        suffix = video_path.suffix.lower()
        is_audio = suffix in audio_extensions
        is_video = suffix in video_extensions

        if not (is_audio or is_video):
            raise ValueError(
                f"Неподдерживаемый формат файла: {video_path.suffix}. "
                f"Аудио: {audio_extensions}, видео: {video_extensions}"
            )

        audio_path = None
        try:
            if is_video:
                audio_path = self.extract_audio(video_path)
            else:
                audio_path = video_path
                delete_audio = False

            diarization = self.perform_diarization(audio_path)
            segments = self.merge_transcription_and_diarization(audio_path, diarization)

            if output_path:
                self.save_results(segments, Path(output_path))

            return segments
        finally:
            if delete_audio and audio_path is not None and audio_path != video_path:
                audio_path.unlink(missing_ok=True)
                logger.debug(f"Временный файл удален: {audio_path}")

    def save_results(
        self, segments: List[TranscriptionSegment], output_path: Path
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".srt":
            self._save_srt(segments, output_path)
        else:
            self._save_txt(segments, output_path)

        logger.info(f"Результаты сохранены в: {output_path}")

    def _save_txt(
        self, segments: List[TranscriptionSegment], output_path: Path
    ) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Транскрибация видео с разделением по спикерам\n")
            f.write("=" * 50 + "\n\n")
            for segment in segments:
                f.write(str(segment) + "\n")

    def _save_srt(
        self, segments: List[TranscriptionSegment], output_path: Path
    ) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, segment in enumerate(segments, 1):
                start_srt = self._format_time_srt(segment.start_time)
                end_srt = self._format_time_srt(segment.end_time)
                f.write(f"{idx}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{segment.speaker}: {segment.text}\n\n")

    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millisecs = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millisecs:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Транскрибация видео с разделением речи по спикерам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python transcribe1.py video.mp4
  python transcribe1.py video.mp4 -o transcript.txt
  python transcribe1.py video.mp4 -l ru -o transcript.srt --device cuda
        """,
    )
    parser.add_argument("input", type=str, help="Путь к видео или аудио")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument(
        "-l", "--language", type=str, default=None, help="Язык, например ru"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-large-v3")
    parser.add_argument(
        "--diarization-model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
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

        if not segments:
            logger.warning(
                "Сегментов с текстом нет. Проверьте аудио и логи ошибок Whisper."
            )
    except Exception as e:
        logger.error(f"Ошибка при выполнении транскрибации: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
