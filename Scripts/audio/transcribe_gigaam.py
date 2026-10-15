#!/usr/bin/env python3
"""
Модуль для транскрибации видео/аудио с разделением речи по говорящим.

Использует:
- GigaAM-Multilingual (ai-sage) для распознавания речи
- PyAnnote для диаризации спикеров

GigaAM.transcribe() принимает только короткие wav (~25 с).
transcribe_longform() на Windows падает (AudioDecoder / torchcodec),
поэтому длинные сегменты нарезаются вручную.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
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

from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation, Segment
from transformers import AutoModel
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

TARGET_SR = 16000
MIN_SEGMENT_SECONDS = 0.1
GIGAAM_SAFE_SECONDS = 24.0
GIGAAM_CHUNK_SECONDS = 20.0
GIGAAM_CHUNK_OVERLAP = 0.4
GIGAAM_ASR_REVISIONS = {"ctc", "large_ctc"}
GIGAAM_SSL_REVISIONS = {"ssl", "large_ssl"}

SKIP_TOO_SHORT = "too_short"
SKIP_EMPTY_AUDIO = "empty_audio"
SKIP_EMPTY_ASR = "empty_asr"
SKIP_ASR_ERROR = "asr_error"


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


@dataclass
class TrackOutcome:
    """Результат обработки одного трека диаризации."""

    speaker: str
    start_time: float
    end_time: float
    duration: float
    samples: int
    text: str = ""
    skip_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.text) and self.skip_reason is None


def resample_audio(
    audio: np.ndarray, sample_rate: int, target_sr: int = TARGET_SR
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
        asr_model_name: str = "ai-sage/GigaAM-Multilingual",
        asr_revision: str = "ctc",
        diarization_model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "auto",
        language: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.device = self._setup_device(device)
        self.language = language
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.asr_revision = asr_revision
        self.last_outcomes: List[TrackOutcome] = []
        self.last_audio_duration: float = 0.0

        if not self.hf_token:
            logger.warning(
                "Токен Hugging Face не предоставлен. "
                "Установите HF_TOKEN или передайте --hf-token."
            )

        logger.info(f"Инициализация моделей на устройстве: {self.device}")

        self._check_ffmpeg()
        self._load_asr_model(asr_model_name, asr_revision)
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

    def _load_asr_model(self, model_name: str, revision: str) -> None:
        """Загрузка GigaAM-Multilingual без ASR-pipeline (без torchcodec)."""
        if revision in GIGAAM_SSL_REVISIONS:
            raise ValueError(
                f"Ревизия '{revision}' — SSL-модель без транскрибации. "
                "Используйте 'ctc' или 'large_ctc'."
            )
        if revision not in GIGAAM_ASR_REVISIONS:
            logger.warning(
                f"Нестандартная ревизия GigaAM: '{revision}'. "
                f"Ожидаются: {sorted(GIGAAM_ASR_REVISIONS)}"
            )

        try:
            logger.info(f"Загрузка модели GigaAM: {model_name} (revision={revision})")

            kwargs = {
                "revision": revision,
                "trust_remote_code": True,
            }
            if self.hf_token:
                kwargs["token"] = self.hf_token

            self.asr_model = AutoModel.from_pretrained(model_name, **kwargs)

            if self.device != "cpu":
                self.asr_model.to(self.device)
            self.asr_model.eval()

            logger.info("Модель GigaAM успешно загружена")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели GigaAM: {e}") from e

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
                str(TARGET_SR),
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

        return resample_audio(audio, sample_rate, TARGET_SR)

    @staticmethod
    def _normalize_transcription(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            for key in ("text", "transcription", "result"):
                value = result.get(key)
                if value:
                    return str(value).strip()
            return str(result).strip()
        if isinstance(result, (list, tuple)):
            parts: List[str] = []
            for item in result:
                if isinstance(item, dict):
                    text = item.get("transcription") or item.get("text") or ""
                    if text:
                        parts.append(str(text).strip())
                elif item:
                    parts.append(str(item).strip())
            return " ".join(p for p in parts if p)
        return str(result).strip()

    @staticmethod
    def _is_too_long_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "too long" in msg or "transcribe_longform" in msg

    def _invoke_asr(self, method: Any, source: Any) -> Any:
        kwargs = {}
        if self.language:
            kwargs["language"] = self.language
        try:
            return method(source, **kwargs) if kwargs else method(source)
        except TypeError:
            return method(source)

    def _write_temp_wav(self, audio: np.ndarray) -> str:
        import soundfile as sf

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_path, audio, TARGET_SR)
        return temp_path

    def _transcribe_wav_file(self, wav_path: str) -> str:
        """Только короткий файл: transcribe_longform на Windows сломан."""
        result = self._invoke_asr(self.asr_model.transcribe, wav_path)
        return self._normalize_transcription(result)

    def _transcribe_short_array(self, audio: np.ndarray) -> str:
        """Сегмент короче лимита GigaAM CTC (~25 с)."""
        if hasattr(self.asr_model, "transcribe_wav"):
            wav = torch.from_numpy(np.ascontiguousarray(audio)).float()
            with torch.inference_mode():
                try:
                    result = self.asr_model.transcribe_wav(wav)
                    return self._normalize_transcription(result)
                except Exception as e:
                    if self._is_too_long_error(e):
                        raise
                    logger.debug("transcribe_wav недоступен, путь через wav: %s", e)

        temp_path = self._write_temp_wav(audio)
        try:
            with torch.inference_mode():
                return self._transcribe_wav_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _iter_audio_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        chunk_samples = int(GIGAAM_CHUNK_SECONDS * TARGET_SR)
        overlap_samples = int(GIGAAM_CHUNK_OVERLAP * TARGET_SR)
        step = max(1, chunk_samples - overlap_samples)
        min_samples = int(MIN_SEGMENT_SECONDS * TARGET_SR)

        chunks: List[np.ndarray] = []
        for start in range(0, len(audio), step):
            chunk = audio[start : start + chunk_samples]
            if chunk.size >= min_samples:
                chunks.append(chunk)
            if start + chunk_samples >= len(audio):
                break
        return chunks

    def _transcribe_chunked(self, audio: np.ndarray) -> str:
        """Длинный сегмент: нарезка < 25 с + обычный transcribe()."""
        texts: List[str] = []
        for chunk in self._iter_audio_chunks(audio):
            try:
                text = self._transcribe_short_array(chunk)
            except Exception as e:
                if self._is_too_long_error(e):
                    logger.warning(
                        "Чанк всё ещё слишком длинный (%.2f с), пропускаю",
                        len(chunk) / float(TARGET_SR),
                    )
                    continue
                logger.error("Ошибка транскрибации чанка: %s", e)
                continue
            if text:
                texts.append(text)
        return " ".join(texts)

    def _transcribe_numpy(self, audio: np.ndarray) -> str:
        """Транскрибация numpy float32 @ 16 kHz через GigaAM."""
        if audio.size == 0:
            return ""

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        duration = len(audio) / float(TARGET_SR)
        if duration > GIGAAM_SAFE_SECONDS:
            return self._transcribe_chunked(audio)

        try:
            return self._transcribe_short_array(audio)
        except Exception as e:
            if self._is_too_long_error(e):
                return self._transcribe_chunked(audio)
            raise

    def _process_track(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segment: Segment,
        speaker: str,
    ) -> TrackOutcome:
        duration = float(segment.end - segment.start)
        outcome = TrackOutcome(
            speaker=speaker,
            start_time=float(segment.start),
            end_time=float(segment.end),
            duration=duration,
            samples=0,
        )

        if duration < MIN_SEGMENT_SECONDS:
            outcome.skip_reason = SKIP_TOO_SHORT
            return outcome

        audio = self._prepare_audio_chunk(waveform, sample_rate, segment)
        outcome.samples = int(audio.size)
        if audio.size == 0:
            outcome.skip_reason = SKIP_EMPTY_AUDIO
            return outcome

        try:
            text = self._transcribe_numpy(audio)
        except Exception as e:
            logger.error("Ошибка транскрибации сегмента %s: %s", segment, e)
            outcome.skip_reason = SKIP_ASR_ERROR
            outcome.error = str(e)
            return outcome

        if not text:
            outcome.skip_reason = SKIP_EMPTY_ASR
            return outcome

        outcome.text = text
        return outcome

    def merge_transcription_and_diarization(
        self, audio_path: Path, diarization: Annotation
    ) -> List[TranscriptionSegment]:
        logger.info("Объединение результатов транскрибации и диаризации...")

        waveform, sample_rate = self.load_audio_safe(audio_path)
        self.last_audio_duration = float(waveform.shape[-1]) / float(sample_rate)
        tracks = list(diarization.itertracks(yield_label=True))
        outcomes: List[TrackOutcome] = []

        for segment, _, speaker in tqdm(tracks, desc="Транскрибация сегментов"):
            outcomes.append(
                self._process_track(waveform, sample_rate, segment, speaker)
            )

        self.last_outcomes = outcomes
        segments = [
            TranscriptionSegment(
                speaker=item.speaker,
                start_time=item.start_time,
                end_time=item.end_time,
                text=item.text,
            )
            for item in outcomes
            if item.ok
        ]
        segments.sort(key=lambda x: x.start_time)

        logger.info(f"Готовых сегментов с текстом: {len(segments)} из {len(tracks)}")
        self._log_skip_diagnostics(outcomes)
        return segments

    def _uncovered_gaps(self, outcomes: List[TrackOutcome]) -> List[Dict[str, float]]:
        """Интервалы аудио, которые диаризация вообще не разметила."""
        if not outcomes:
            if self.last_audio_duration > 0:
                return [
                    {
                        "start": 0.0,
                        "end": self.last_audio_duration,
                        "duration": self.last_audio_duration,
                    }
                ]
            return []

        covered = sorted((item.start_time, item.end_time) for item in outcomes)
        merged: List[List[float]] = []
        for start, end in covered:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        gaps: List[Dict[str, float]] = []
        cursor = 0.0
        for start, end in merged:
            if start - cursor >= MIN_SEGMENT_SECONDS:
                gaps.append({"start": cursor, "end": start, "duration": start - cursor})
            cursor = max(cursor, end)
        if self.last_audio_duration - cursor >= MIN_SEGMENT_SECONDS:
            gaps.append(
                {
                    "start": cursor,
                    "end": self.last_audio_duration,
                    "duration": self.last_audio_duration - cursor,
                }
            )
        return gaps

    def _log_skip_diagnostics(self, outcomes: List[TrackOutcome]) -> None:
        skipped = [item for item in outcomes if not item.ok]
        if not skipped:
            logger.info("Пропусков ASR нет: все треки диаризации дали текст")
            return

        counts: Dict[str, int] = {}
        durations: Dict[str, float] = {}
        for item in skipped:
            reason = item.skip_reason or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
            durations[reason] = durations.get(reason, 0.0) + item.duration

        logger.info("Диагностика пропусков ASR:")
        for reason in sorted(counts):
            logger.info(
                "  %-12s %3d шт, суммарно %.1f с",
                reason,
                counts[reason],
                durations[reason],
            )

        longest = sorted(skipped, key=lambda x: x.duration, reverse=True)[:15]
        logger.info("Самые длинные пропуски ASR:")
        for item in longest:
            logger.info(
                "  [%.2fs - %.2fs] %s dur=%.2fs samples=%d reason=%s%s",
                item.start_time,
                item.end_time,
                item.speaker,
                item.duration,
                item.samples,
                item.skip_reason,
                f" err={item.error}" if item.error else "",
            )

        all_gaps = self._uncovered_gaps(outcomes)
        gaps = [gap for gap in all_gaps if gap["duration"] >= 1.0]
        gap_total = sum(gap["duration"] for gap in all_gaps)
        logger.info(
            "Участки без диаризации (тишина/музыка): %.1f с из %.1f с аудио",
            gap_total,
            self.last_audio_duration,
        )
        if gaps:
            logger.info("Пробелы диаризации >= 1 с (первые 10):")
            for gap in gaps[:10]:
                logger.info(
                    "  [%.2fs - %.2fs] dur=%.2fs",
                    gap["start"],
                    gap["end"],
                    gap["duration"],
                )

    def build_debug_report(self) -> Dict[str, Any]:
        outcomes = self.last_outcomes
        skipped = [item for item in outcomes if not item.ok]
        ok = [item for item in outcomes if item.ok]
        counts: Dict[str, int] = {}
        for item in skipped:
            reason = item.skip_reason or "unknown"
            counts[reason] = counts.get(reason, 0) + 1

        return {
            "audio_duration": self.last_audio_duration,
            "tracks": len(outcomes),
            "with_text": len(ok),
            "without_text": len(skipped),
            "skip_counts": counts,
            "skip_duration_sec": {
                reason: sum(
                    item.duration
                    for item in skipped
                    if (item.skip_reason or "unknown") == reason
                )
                for reason in counts
            },
            "skipped": [asdict(item) for item in skipped],
            "diarization_gaps_ge_1s": [
                gap for gap in self._uncovered_gaps(outcomes) if gap["duration"] >= 1.0
            ],
        }

    def transcribe_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        delete_audio: bool = True,
        include_empty: bool = False,
        debug_report_path: Optional[Union[str, Path]] = None,
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

            if include_empty:
                segments = [
                    TranscriptionSegment(
                        speaker=item.speaker,
                        start_time=item.start_time,
                        end_time=item.end_time,
                        text=item.text or f"[{item.skip_reason or 'empty'}]",
                    )
                    for item in sorted(self.last_outcomes, key=lambda x: x.start_time)
                ]

            if output_path:
                self.save_results(segments, Path(output_path))

            report_path = None
            if debug_report_path:
                report_path = Path(debug_report_path)
            elif output_path:
                out = Path(output_path)
                report_path = out.with_name(out.stem + ".debug.json")
            if report_path is not None:
                self.save_debug_report(report_path)

            return segments
        finally:
            if delete_audio and audio_path is not None and audio_path != video_path:
                audio_path.unlink(missing_ok=True)
                logger.debug(f"Временный файл удален: {audio_path}")

    def save_debug_report(self, report_path: Path) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.build_debug_report()
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Диагностический отчёт сохранён в: %s", report_path)

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
        skipped = [item for item in self.last_outcomes if not item.ok]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Транскрибация видео с разделением по спикерам\n")
            f.write("=" * 50 + "\n\n")
            for segment in segments:
                f.write(str(segment) + "\n")

            if skipped:
                f.write("\n")
                f.write("=" * 50 + "\n")
                f.write("Пропущенные сегменты (без текста ASR)\n")
                f.write("=" * 50 + "\n")
                for item in sorted(skipped, key=lambda x: x.start_time):
                    f.write(
                        f"[{item.start_time:.2f}s - {item.end_time:.2f}s] "
                        f"{item.speaker}: [{item.skip_reason}] "
                        f"dur={item.duration:.2f}s samples={item.samples}"
                    )
                    if item.error:
                        f.write(f" err={item.error}")
                    f.write("\n")

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
  python transcribe1.py video.mp4 --asr-revision large_ctc --include-empty
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
        "--asr-model",
        type=str,
        default="ai-sage/GigaAM-Multilingual",
    )
    parser.add_argument(
        "--asr-revision",
        type=str,
        default="ctc",
        choices=["ctc", "large_ctc", "ssl", "large_ssl"],
        help="Ревизия GigaAM. Для транскрибации: ctc или large_ctc",
    )
    parser.add_argument(
        "--diarization-model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Писать в транскрипт и сегменты без текста (с пометкой причины)",
    )
    parser.add_argument(
        "--debug-report",
        type=str,
        default=None,
        help="Путь к JSON-отчёту. По умолчанию: <output>.debug.json",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        transcriber = VideoTranscriber(
            asr_model_name=args.asr_model,
            asr_revision=args.asr_revision,
            diarization_model_name=args.diarization_model,
            device=args.device,
            language=args.language,
            hf_token=args.hf_token,
        )
        segments = transcriber.transcribe_video(
            video_path=args.input,
            output_path=args.output,
            delete_audio=not args.keep_audio,
            include_empty=args.include_empty,
            debug_report_path=args.debug_report,
        )

        if not args.output:
            print("\n" + "=" * 50)
            print("РЕЗУЛЬТАТЫ ТРАНСКРИБАЦИИ:")
            print("=" * 50)
            for segment in segments:
                print(segment)

        if not segments:
            logger.warning(
                "Сегментов с текстом нет. Проверьте аудио и логи ошибок GigaAM."
            )
    except Exception as e:
        logger.error(f"Ошибка при выполнении транскрибации: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
