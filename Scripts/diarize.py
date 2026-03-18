"""
Speaker Diarization — разделение аудио на отдельных говорящих.
"""

import os
import sys
import argparse
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*torchcodec.*")

import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None

import soundfile as sf
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.core import Annotation

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def convert_to_wav_mono(
    input_path: str, output_path: str, target_sr: int = 16000
) -> None:
    print(f"  Конвертация в WAV mono {target_sr} Hz...")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sr)
    audio = audio.set_sample_width(2)
    audio.export(output_path, format="wav")


def load_audio_as_tensor(wav_path: str) -> dict:
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T
    waveform = torch.from_numpy(data)
    return {"waveform": waveform, "sample_rate": sr}


def extract_annotation(diarization_result) -> Annotation:
    """
    Универсальное извлечение Annotation из результата pipeline.
    Поддерживает все версии pyannote.
    """
    result_type = type(diarization_result).__name__
    public_attrs = [a for a in dir(diarization_result) if not a.startswith("_")]
    print(f"  Тип результата: {result_type}")
    print(f"  Атрибуты: {public_attrs}")

    # 1) Сам объект — Annotation
    if isinstance(diarization_result, Annotation):
        print(f"  → Результат сам является Annotation")
        return diarization_result

    if hasattr(diarization_result, "itertracks"):
        print(f"  → Результат имеет itertracks (duck-typing)")
        return diarization_result

    # 2) Известные атрибуты
    for attr_name in [
        "annotation",
        "speaker",
        "diarization",
        "output",
        "result",
        "labels",
        "segments",
        "speaker_annotation",
    ]:
        obj = getattr(diarization_result, attr_name, None)
        if obj is not None and (
            isinstance(obj, Annotation) or hasattr(obj, "itertracks")
        ):
            print(f"  → Annotation найден в .{attr_name}")
            return obj

    # 3) Перебор ВСЕХ атрибутов
    for attr_name in public_attrs:
        try:
            obj = getattr(diarization_result, attr_name)
            if isinstance(obj, Annotation) or hasattr(obj, "itertracks"):
                print(f"  → Annotation найден в .{attr_name}")
                return obj
        except Exception:
            continue

    # 4) Итерация
    try:
        for item in diarization_result:
            if isinstance(item, Annotation) or hasattr(item, "itertracks"):
                print(f"  → Annotation найден через итерацию")
                return item
    except TypeError:
        pass

    # 5) Ничего не нашли — диагностика
    print(f"\n  ОШИБКА: Не удалось извлечь Annotation из {result_type}")
    for attr_name in public_attrs:
        try:
            val = getattr(diarization_result, attr_name)
            val_repr = repr(val)[:300]
            print(f"    .{attr_name} ({type(val).__name__}): {val_repr}")
        except Exception as e:
            print(f"    .{attr_name}: ОШИБКА {e}")
    sys.exit(1)


def apply_fade(segment: np.ndarray, fade_samples: int) -> np.ndarray:
    result = segment.copy()
    length = len(result)
    fade_in_len = min(fade_samples, length // 2)
    fade_out_len = min(fade_samples, length // 2)
    if fade_in_len > 0:
        result[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len, dtype=np.float64)
    if fade_out_len > 0:
        result[-fade_out_len:] *= np.linspace(1.0, 0.0, fade_out_len, dtype=np.float64)
    return result


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Диаризация аудио — разделение на отдельных говорящих",
    )
    parser.add_argument("input", type=str, help="Путь к входному аудиофайлу")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--fade-ms", type=float, default=10.0)
    parser.add_argument("--output-sr", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ОШИБКА: Файл не найден: {input_path}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = input_path.parent / f"{input_path.stem}_speakers"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ДИАРИЗАЦИЯ АУДИО")
    print("=" * 60)
    print(f"  Вход:   {input_path}")
    print(f"  Выход:  {output_dir}")
    print(
        f"  GPU:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'НЕ НАЙДЕН'}"
    )
    print(f"  PyTorch: {torch.__version__}")

    # Версия pyannote
    try:
        import pyannote.audio

        print(f"  pyannote.audio: {pyannote.audio.__version__}")
    except AttributeError:
        print(f"  pyannote.audio: (версия неизвестна)")
    print("=" * 60)

    # ─── Шаг 1: Подготовка аудио ──────────────────────────────────────
    print("\n[1/4] Подготовка аудио...")

    temp_wav = output_dir / "_temp_input_16k.wav"
    convert_to_wav_mono(str(input_path), str(temp_wav), target_sr=16000)

    if input_path.suffix.lower() == ".wav":
        original_audio, original_sr = sf.read(str(input_path), dtype="float64")
    else:
        temp_original = output_dir / "_temp_original.wav"
        audio_seg = AudioSegment.from_file(str(input_path))
        audio_seg = audio_seg.set_channels(1).set_sample_width(2)
        audio_seg.export(str(temp_original), format="wav")
        original_audio, original_sr = sf.read(str(temp_original), dtype="float64")
        temp_original.unlink(missing_ok=True)

    if original_audio.ndim > 1:
        original_audio = original_audio.mean(axis=1)

    output_sr = args.output_sr or original_sr
    total_samples = len(original_audio)
    duration = total_samples / original_sr

    print(f"  Длительность:  {format_time(duration)}")
    print(f"  Sample rate:   {original_sr} Hz")
    print(f"  Сэмплов:      {total_samples:,}")

    # ─── Шаг 2: Загрузка модели ───────────────────────────────────────
    print("\n[2/4] Загрузка модели pyannote/speaker-diarization-3.1...")
    t_start = time.time()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    t_load = time.time() - t_start
    print(f"  Модель загружена за {t_load:.1f}с на {device}")

    # ─── Шаг 3: Диаризация ────────────────────────────────────────────
    print("\n[3/4] Выполняю диаризацию...")
    t_start = time.time()

    diarization_params = {}
    if args.num_speakers is not None:
        diarization_params["num_speakers"] = args.num_speakers
    if args.min_speakers is not None:
        diarization_params["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        diarization_params["max_speakers"] = args.max_speakers

    audio_in_memory = load_audio_as_tensor(str(temp_wav))
    print(
        f"  Аудио: shape={audio_in_memory['waveform'].shape}, sr={audio_in_memory['sample_rate']}"
    )

    diarization_result = pipeline(audio_in_memory, **diarization_params)

    t_diar = time.time() - t_start
    print(f"  Диаризация завершена за {t_diar:.1f}с")

    # ── Извлечение Annotation (совместимо с любой версией) ──
    annotation = extract_annotation(diarization_result)

    speaker_segments: dict[str, list[tuple[float, float]]] = {}
    all_turns = []

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))
        all_turns.append((turn.start, turn.end, speaker))

    num_speakers = len(speaker_segments)
    print(f"  Обнаружено говорящих: {num_speakers}")

    if num_speakers == 0:
        print("\nНе обнаружено ни одного говорящего.")
        temp_wav.unlink(missing_ok=True)
        sys.exit(1)

    for spk, segs in speaker_segments.items():
        total_spk_time = sum(end - start for start, end in segs)
        print(f"    {spk}: {len(segs)} сегментов, {format_time(total_spk_time)} речи")

    # ─── Шаг 4: Создание выходных файлов ──────────────────────────────
    print("\n[4/4] Создание аудиофайлов по говорящим...")

    fade_samples = int((args.fade_ms / 1000.0) * original_sr)

    for speaker_label, segments in speaker_segments.items():
        speaker_audio = np.zeros(total_samples, dtype=np.float64)
        total_speaking_time = 0.0

        for seg_start, seg_end in segments:
            start_sample = max(0, int(seg_start * original_sr))
            end_sample = min(total_samples, int(seg_end * original_sr))
            if end_sample <= start_sample:
                continue
            segment_data = original_audio[start_sample:end_sample].copy()
            if fade_samples > 0 and len(segment_data) > 0:
                segment_data = apply_fade(segment_data, fade_samples)
            speaker_audio[start_sample:end_sample] = segment_data
            total_speaking_time += seg_end - seg_start

        spk_index = (
            int(speaker_label.split("_")[-1]) + 1
            if "_" in speaker_label
            else speaker_label
        )
        filename = (
            f"speaker_{spk_index:02d}.wav"
            if isinstance(spk_index, int)
            else f"{speaker_label}.wav"
        )

        output_path = output_dir / filename
        sf.write(str(output_path), speaker_audio, output_sr, subtype="PCM_16")

        pct = (total_speaking_time / duration) * 100 if duration > 0 else 0
        print(
            f"  ✓ {filename:20s}  речь: {format_time(total_speaking_time)} ({pct:.1f}%)"
        )

    # ─── Отчёт ─────────────────────────────────────────────────────────
    report_path = output_dir / "diarization_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Файл: {input_path.name}\n")
        f.write(f"Длительность: {format_time(duration)}\n")
        f.write(f"Говорящих: {num_speakers}\n")
        f.write(f"Модель: pyannote/speaker-diarization-3.1\n")
        f.write(f"Устройство: {device}\n")
        f.write(f"Время обработки: {t_diar:.1f}с\n")
        f.write("\n" + "=" * 50 + "\nТАЙМЛАЙН\n" + "=" * 50 + "\n\n")
        for seg_start, seg_end, spk in sorted(all_turns, key=lambda x: x[0]):
            seg_dur = seg_end - seg_start
            f.write(
                f"[{format_time(seg_start)} → {format_time(seg_end)}]  {seg_dur:6.2f}с  {spk}\n"
            )
        f.write("\n" + "=" * 50 + "\nСТАТИСТИКА\n" + "=" * 50 + "\n\n")
        for spk, segs in sorted(speaker_segments.items()):
            total_spk = sum(e - s for s, e in segs)
            pct = (total_spk / duration) * 100 if duration > 0 else 0
            avg_seg = total_spk / len(segs) if segs else 0
            f.write(
                f"{spk}: {len(segs)} сегм., {format_time(total_spk)} ({pct:.1f}%), ср.={avg_seg:.2f}с\n"
            )

    rttm_path = output_dir / "diarization.rttm"
    with open(rttm_path, "w", encoding="utf-8") as f:
        for seg_start, seg_end, spk in sorted(all_turns, key=lambda x: x[0]):
            seg_dur = seg_end - seg_start
            f.write(
                f"SPEAKER {input_path.stem} 1 {seg_start:.3f} {seg_dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            )

    print(f"\n  📄 Отчёт: {report_path}")
    print(f"  📄 RTTM:  {rttm_path}")

    temp_wav.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("  ГОТОВО!")
    print(f"  Результаты в: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
