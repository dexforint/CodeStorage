#!/usr/bin/env python3
"""
Транскрибация видео/аудио с диаризацией через WhisperX.

Пайплайн:
1. ffmpeg → моно WAV 16 kHz
2. WhisperX (faster-whisper) — батчевый ASR
3. wav2vec2 alignment — точные таймкоды
4. PyAnnote (через WhisperX) — спикеры
5. assign_word_speakers — спикер на сегмент

Выход совместим с visualize_transcript.py:
  [12.34s - 15.67s] SPEAKER_00: текст

Примеры:
  python audio/transcribe_whisperx2.py video.mp4 --device cuda -l ru -o transcript_whisperx.txt
  python audio/transcribe_whisperx2.py video.mp4 --device cuda -l ru -o transcript_whisperx.srt
  python audio/visualize_transcript.py video.mp4 transcript_whisperx.txt -o viz.mp4 --nvenc
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=".*torchcodec is not installed correctly.*",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

TARGET_SR = 16000
MIN_SEGMENT_SECONDS = 0.1
DEFAULT_MERGE_GAP = 0.8


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


class WhisperXTranscriber:
    """Транскрибация + диаризация через WhisperX."""

    def __init__(
        self,
        whisper_model_name: str = "large-v3",
        diarization_model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "auto",
        language: Optional[str] = None,
        hf_token: Optional[str] = None,
        batch_size: int = 16,
        compute_type: str = "auto",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        merge_gap: float = DEFAULT_MERGE_GAP,
        align: bool = True,
    ):
        self.language = language
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.batch_size = batch_size
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.merge_gap = merge_gap
        self.align = align
        self.whisper_model_name = whisper_model_name
        self.diarization_model_name = diarization_model_name

        self.device = self._setup_device(device)
        self.compute_type = self._setup_compute_type(compute_type)

        if not self.hf_token:
            logger.warning(
                "Токен Hugging Face не предоставлен. "
                "Установите HF_TOKEN или передайте --hf-token "
                "(нужен для pyannote)."
            )

        logger.info(
            "Устройство: %s, compute_type: %s, модель: %s",
            self.device,
            self.compute_type,
            whisper_model_name,
        )

        self._check_ffmpeg()
        self._import_whisperx()
        self._load_whisper_model()
        self._load_diarization_model()

        self.align_model = None
        self.align_metadata = None
        self.align_language: Optional[str] = None

    def _import_whisperx(self) -> None:
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX не установлен. Поставьте:\n"
                "  pip install whisperx\n"
                "На CUDA также нужен ctranslate2 / faster-whisper."
            ) from exc
        self.whisperx = whisperx

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
            return "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA недоступен, переключение на CPU")
            return "cpu"

        if device == "mps":
            logger.warning(
                "WhisperX/faster-whisper почти не работает на MPS, используем CPU"
            )
            return "cpu"

        return device

    def _setup_compute_type(self, compute_type: str) -> str:
        if compute_type != "auto":
            return compute_type
        if self.device == "cuda":
            return "float16"
        return "int8"

    def _load_whisper_model(self) -> None:
        logger.info("Загрузка WhisperX ASR: %s", self.whisper_model_name)
        kwargs: Dict[str, Any] = {
            "device": self.device,
            "compute_type": self.compute_type,
        }
        if self.language:
            kwargs["language"] = self.language
        try:
            self.asr_model = self.whisperx.load_model(self.whisper_model_name, **kwargs)
        except TypeError:
            kwargs.pop("language", None)
            self.asr_model = self.whisperx.load_model(
                self.whisper_model_name,
                self.device,
                compute_type=self.compute_type,
            )
        logger.info("Модель WhisperX ASR загружена")

    def _load_diarization_model(self) -> None:
        logger.info("Загрузка диаризации: %s", self.diarization_model_name)

        pipeline_cls = getattr(self.whisperx, "DiarizationPipeline", None)
        if pipeline_cls is None:
            try:
                from whisperx.diarize import DiarizationPipeline as pipeline_cls
            except ImportError as exc:
                raise RuntimeError(
                    "В этой версии WhisperX нет DiarizationPipeline"
                ) from exc

        last_error: Optional[Exception] = None
        for kwargs in (
            {
                "token": self.hf_token,
                "device": self.device,
                "model_name": self.diarization_model_name,
            },
            {
                "use_auth_token": self.hf_token,
                "device": self.device,
                "model_name": self.diarization_model_name,
            },
            {"token": self.hf_token, "device": self.device},
            {"use_auth_token": self.hf_token, "device": self.device},
        ):
            try:
                self.diarize_model = pipeline_cls(**kwargs)
                logger.info("Модель диаризации загружена")
                return
            except TypeError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
                logger.debug("Вариант загрузки диаризации не подошёл: %s", exc)

        raise RuntimeError(f"Не удалось загрузить диаризацию WhisperX: {last_error}")

    def _ensure_align_model(self, language: str) -> None:
        if not self.align:
            return
        if self.align_model is not None and self.align_language == language:
            return

        logger.info("Загрузка align-модели для языка: %s", language)
        try:
            model, metadata = self.whisperx.load_align_model(
                language_code=language, device=self.device
            )
        except Exception as exc:
            logger.warning(
                "Align-модель для '%s' не загрузилась (%s). "
                "Продолжаю без выравнивания таймкодов.",
                language,
                exc,
            )
            self.align = False
            return

        self.align_model = model
        self.align_metadata = metadata
        self.align_language = language
        logger.info("Align-модель загружена")

    def extract_audio(self, video_path: Path) -> Path:
        logger.info("Извлечение аудио из видео: %s", video_path)
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
                str(TARGET_SR),
                "-ac",
                "1",
                "-y",
                str(temp_audio_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            if not temp_audio_path.exists() or temp_audio_path.stat().st_size == 0:
                raise RuntimeError("Создан пустой аудиофайл")
            logger.info("Аудио успешно извлечено через ffmpeg")
            return temp_audio_path
        except subprocess.CalledProcessError as e:
            logger.error("Ошибка ffmpeg: %s", e.stderr)
            raise RuntimeError(f"Не удалось извлечь аудио через ffmpeg: {e}") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "FFmpeg не найден. Установите ffmpeg:\n"
                "Windows: choco install ffmpeg или https://ffmpeg.org/download.html"
            ) from e

    def _load_audio_array(self, audio_path: Path) -> np.ndarray:
        try:
            audio = self.whisperx.load_audio(str(audio_path))
            return np.asarray(audio, dtype=np.float32)
        except Exception as exc:
            logger.warning(
                "whisperx.load_audio не сработал (%s), читаю через soundfile",
                exc,
            )
            import soundfile as sf

            data, sample_rate = sf.read(str(audio_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sample_rate != TARGET_SR:
                duration = len(data) / float(sample_rate)
                new_length = max(1, int(round(duration * TARGET_SR)))
                x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
                data = np.interp(x_new, x_old, data).astype(np.float32)
            return np.asarray(data, dtype=np.float32)

    def _transcribe(self, audio: np.ndarray) -> Dict[str, Any]:
        logger.info("ASR WhisperX (batch_size=%d)...", self.batch_size)
        kwargs: Dict[str, Any] = {"batch_size": self.batch_size}
        if self.language:
            kwargs["language"] = self.language
        result = self.asr_model.transcribe(audio, **kwargs)
        language = result.get("language") or self.language or "unknown"
        n_seg = len(result.get("segments") or [])
        logger.info("ASR готов: язык=%s, сегментов=%d", language, n_seg)
        return result

    def _align_segments(
        self, result: Dict[str, Any], audio: np.ndarray
    ) -> Dict[str, Any]:
        if not self.align:
            return result
        language = result.get("language") or self.language
        if not language:
            logger.warning("Язык неизвестен, выравнивание пропущено")
            return result

        self._ensure_align_model(language)
        if self.align_model is None:
            return result

        logger.info("Выравнивание таймкодов...")
        aligned = self.whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        if isinstance(aligned, dict):
            result = dict(result)
            result["segments"] = aligned.get("segments", result.get("segments"))
            if "word_segments" in aligned:
                result["word_segments"] = aligned["word_segments"]
            return result
        result = dict(result)
        result["segments"] = aligned
        return result

    def _diarize(self, audio: np.ndarray, audio_path: Path) -> Any:
        logger.info("Диаризация спикеров...")
        kwargs: Dict[str, Any] = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers

        waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
        candidates = [
            {"waveform": waveform, "sample_rate": TARGET_SR},
            audio,
            str(audio_path),
        ]

        last_error: Optional[Exception] = None
        for payload in candidates:
            try:
                diarize_segments = (
                    self.diarize_model(payload, **kwargs)
                    if kwargs
                    else self.diarize_model(payload)
                )
                n_spk = self._count_speakers(diarize_segments)
                logger.info("Диаризация готова, спикеров: %s", n_spk)
                return diarize_segments
            except TypeError:
                try:
                    diarize_segments = self.diarize_model(payload)
                    n_spk = self._count_speakers(diarize_segments)
                    logger.info("Диаризация готова, спикеров: %s", n_spk)
                    return diarize_segments
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc
                logger.debug("Вариант входа диаризации не подошёл: %s", exc)

        raise RuntimeError(f"Диаризация WhisperX не удалась: {last_error}")

    @staticmethod
    def _count_speakers(diarize_segments: Any) -> Union[int, str]:
        if hasattr(diarize_segments, "unique"):
            try:
                if "speaker" in getattr(diarize_segments, "columns", []):
                    return int(diarize_segments["speaker"].nunique())
            except Exception:
                pass
        if hasattr(diarize_segments, "labels"):
            try:
                return len(list(diarize_segments.labels()))
            except Exception:
                pass
        return "?"

    def _assign_speakers(
        self, diarize_segments: Any, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info("Назначение спикеров словам/сегментам...")
        assigned = self.whisperx.assign_word_speakers(diarize_segments, result)
        if isinstance(assigned, dict):
            return assigned
        result = dict(result)
        result["segments"] = assigned
        return result

    def _to_segments(self, result: Dict[str, Any]) -> List[TranscriptionSegment]:
        raw = result.get("segments") or []
        segments: List[TranscriptionSegment] = []
        for item in raw:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            if end - start < MIN_SEGMENT_SECONDS:
                continue
            speaker = item.get("speaker") or "SPEAKER_UNK"
            segments.append(
                TranscriptionSegment(
                    speaker=str(speaker),
                    start_time=start,
                    end_time=end,
                    text=text,
                )
            )
        segments.sort(key=lambda x: x.start_time)
        return segments

    def _merge_speaker_turns(
        self, segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        if self.merge_gap < 0 or len(segments) < 2:
            return segments

        merged: List[TranscriptionSegment] = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg.start_time - prev.end_time
            if seg.speaker == prev.speaker and gap <= self.merge_gap:
                prev.end_time = max(prev.end_time, seg.end_time)
                prev.text = f"{prev.text} {seg.text}".strip()
            else:
                merged.append(
                    TranscriptionSegment(
                        speaker=seg.speaker,
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        text=seg.text,
                    )
                )
        logger.info(
            "Склейка реплик одного спикера (gap<=%.2fs): %d → %d",
            self.merge_gap,
            len(segments),
            len(merged),
        )
        return merged

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

            audio = self._load_audio_array(audio_path)
            result = self._transcribe(audio)
            result = self._align_segments(result, audio)
            diarize_segments = self._diarize(audio, audio_path)
            result = self._assign_speakers(diarize_segments, result)

            segments = self._to_segments(result)
            segments = self._merge_speaker_turns(segments)
            logger.info("Готовых сегментов с текстом: %d", len(segments))

            if output_path:
                self.save_results(segments, Path(output_path))
            return segments
        finally:
            if delete_audio and audio_path is not None and audio_path != video_path:
                audio_path.unlink(missing_ok=True)
                logger.debug("Временный файл удален: %s", audio_path)

    def save_results(
        self, segments: List[TranscriptionSegment], output_path: Path
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".srt":
            self._save_srt(segments, output_path)
        else:
            self._save_txt(segments, output_path)
        logger.info("Результаты сохранены в: %s", output_path)

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
        description="Транскрибация и диаризация через WhisperX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python audio/transcribe_whisperx.py video.mp4 --device cuda -l ru -o transcript_whisperx.txt
  python audio/transcribe_whisperx.py video.mp4 --batch-size 8 --min-speakers 2 --max-speakers 6
  python audio/visualize_transcript.py video.mp4 transcript_whisperx.txt -o viz.mp4 --nvenc
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
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="Имя faster-whisper модели: large-v3, large-v2, medium, ...",
    )
    parser.add_argument(
        "--diarization-model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--compute-type",
        type=str,
        default="auto",
        help="float16 / float32 / int8 / auto",
    )
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=DEFAULT_MERGE_GAP,
        help="Склеивать соседние реплики одного спикера, если пауза <= N сек. "
        "-1 отключает склейку",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Не выравнивать таймкоды wav2vec2",
    )
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        transcriber = WhisperXTranscriber(
            whisper_model_name=args.whisper_model,
            diarization_model_name=args.diarization_model,
            device=args.device,
            language=args.language,
            hf_token=args.hf_token,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            merge_gap=args.merge_gap,
            align=not args.no_align,
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
            logger.warning("Сегментов с текстом нет.")
    except Exception as e:
        logger.error("Ошибка при выполнении транскрибации: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
