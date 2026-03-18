# https://arena.ai/c/019cfff9-e5e8-7c7f-8ef9-39e93334c4ef
# лучше создать виртуальное окружение
# python -m venv venv
# venv/Scripts/activate
# нужно установить актуальный Pytorch с оф. сайта
# pip install -U requests hyperpyyaml
# pip install huggingface_hub==0.25.2
# pip install -U speechbrain silero-vad soundfile demucs torchcodec

# Максимальное качество
# python speech_only.py ./data/test.mp3 ./data/test_output.wav --music-heavy --device cuda --mode mute
# Быстрее
# python speech_only.py ./data/test.mp3 ./data/test_output.wav --music-heavy --demucs-fast --device cuda --mode mute

import os
import inspect
import argparse
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_SR = 16000


def patch_torchaudio_for_speechbrain():
    """
    Совместимость новых версий torchaudio со SpeechBrain.
    Некоторые версии speechbrain ожидают старый backend API torchaudio.
    """
    if not hasattr(torchaudio, "list_audio_backends"):

        def _list_audio_backends():
            backends = []
            try:
                import soundfile  # noqa: F401

                backends.append("soundfile")
            except Exception:
                pass
            backends.append("ffmpeg")
            return backends

        torchaudio.list_audio_backends = _list_audio_backends

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: None

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend=None: None


def patch_huggingface_hub_for_speechbrain():
    """
    Совместимость старого SpeechBrain с новым huggingface_hub,
    где use_auth_token заменён на token.
    """
    try:
        import huggingface_hub
    except Exception:
        return

    try:
        sig = inspect.signature(huggingface_hub.hf_hub_download)
    except Exception:
        return

    if "use_auth_token" not in sig.parameters:
        original_hf_hub_download = huggingface_hub.hf_hub_download

        def wrapped_hf_hub_download(*args, use_auth_token=None, **kwargs):
            if use_auth_token is not None and "token" not in kwargs:
                kwargs["token"] = use_auth_token
            return original_hf_hub_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = wrapped_hf_hub_download


def get_speechbrain_enhancer_class():
    patch_torchaudio_for_speechbrain()
    patch_huggingface_hub_for_speechbrain()

    try:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        return SpectralMaskEnhancement
    except Exception:
        from speechbrain.pretrained import SpectralMaskEnhancement

        return SpectralMaskEnhancement


def resolve_device(requested: str = "auto") -> str:
    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен CUDA, но torch.cuda.is_available() == False")
        try:
            x = torch.zeros(1, device="cuda")
            y = x + 1
            _ = y.item()
            torch.cuda.synchronize()
            return "cuda"
        except Exception as e:
            raise RuntimeError(f"CUDA обнаружена, но неработоспособна: {e}") from e

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"[INFO] Найдена GPU: {name}, sm_{cap[0]}{cap[1]}")

            x = torch.zeros(1, device="cuda")
            y = x + 1
            _ = y.item()
            torch.cuda.synchronize()

            return "cuda"
        except Exception as e:
            print(f"[WARN] CUDA недоступна для реальных вычислений, откат на CPU: {e}")

    return "cpu"


def get_speechbrain_local_strategy():
    """
    Выбираем стратегию без symlink для Windows.
    """
    patch_torchaudio_for_speechbrain()

    try:
        from speechbrain.utils.fetching import LocalStrategy
    except Exception:
        return None

    for name in ("COPY", "COPY_SKIP_CACHE", "NO_LINK"):
        if hasattr(LocalStrategy, name):
            return getattr(LocalStrategy, name)

    return None


def load_metricgan_enhancer(device: str):
    """
    Загружает SpeechBrain MetricGAN+ без использования symlink,
    если версия speechbrain это поддерживает.
    """
    SpectralMaskEnhancement = get_speechbrain_enhancer_class()

    kwargs = dict(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device": device},
    )

    local_strategy = get_speechbrain_local_strategy()
    if local_strategy is not None:
        strategy_name = getattr(local_strategy, "name", str(local_strategy))
        print(f"[INFO] SpeechBrain local_strategy: {strategy_name}")

        try:
            return SpectralMaskEnhancement.from_hparams(
                **kwargs,
                local_strategy=local_strategy,
            )
        except TypeError:
            pass

    return SpectralMaskEnhancement.from_hparams(**kwargs)


def run_ffmpeg_extract_wav_preserve(input_path: str, output_path: str):
    """
    Извлекает аудио в WAV PCM без принудительного изменения sample rate / channels.
    Это high-quality master для финального результата, если Demucs не используется.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "FFmpeg не найден в PATH. Установи FFmpeg и проверь команду 'ffmpeg -version'."
        ) from e


def run_ffmpeg_to_wav(
    input_path: str, output_path: str, sample_rate: int, channels: int
):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "FFmpeg не найден в PATH. Установи FFmpeg и проверь команду 'ffmpeg -version'."
        ) from e


def run_ffmpeg_to_model_proxy(input_path: str, output_path: str):
    """
    Подготовка proxy для моделей: mono 16k.
    Используется только для SpeechBrain/VAD.
    """
    run_ffmpeg_to_wav(
        input_path=input_path,
        output_path=output_path,
        sample_rate=MODEL_SR,
        channels=1,
    )


def load_audio_sf(path: str):
    """
    Чтение аудио через soundfile.
    Возвращает:
      wav: torch.Tensor [channels, samples]
      sr: int
    """
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    wav = torch.from_numpy(data.T.copy())
    return wav, sr


def save_audio_sf(path: str, wav: torch.Tensor, sr: int):
    """
    Сохранение аудио через soundfile.
    Ожидает wav формы [samples] или [channels, samples].
    """
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    wav = wav.detach().cpu()

    if wav.ndim == 1:
        data = wav.numpy()
    elif wav.ndim == 2:
        data = wav.transpose(0, 1).contiguous().numpy()
    else:
        raise ValueError(f"Unexpected audio tensor shape: {tuple(wav.shape)}")

    data = data.clip(-1.0, 1.0)
    sf.write(path, data, sr, subtype="PCM_16")


def make_intermediate_path(final_output_path: Path, stage_name: str) -> Path:
    return final_output_path.parent / f"{final_output_path.stem}.{stage_name}.wav"


def get_demucs_attr(model, attr_name: str):
    if hasattr(model, attr_name):
        return getattr(model, attr_name)

    if (
        hasattr(model, "models")
        and len(model.models) > 0
        and hasattr(model.models[0], attr_name)
    ):
        return getattr(model.models[0], attr_name)

    raise AttributeError(f"Demucs model has no attribute '{attr_name}'")


def run_demucs_vocals(
    input_path: str,
    temp_dir: str,
    device: str,
    save_vocals_path: str,
    fast: bool = False,
) -> str:
    """
    Извлекает vocals через Demucs API, без demucs CLI.
    Если fast=True, используется только первый submodel вместо bag-of-models.
    """
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except Exception as e:
        raise RuntimeError(
            "Не удалось импортировать demucs как библиотеку. "
            "Убедись, что пакет установлен: pip install demucs"
        ) from e

    print("[INFO] Загружаю Demucs model htdemucs_ft...")
    model = get_model(name="htdemucs_ft")

    if fast and hasattr(model, "models") and len(model.models) > 1:
        print(
            f"[INFO] Demucs bag-of-{len(model.models)} -> использую только первый submodel"
        )
        model = model.models[0]

    model.to(device)
    model.eval()

    samplerate = int(get_demucs_attr(model, "samplerate"))
    audio_channels = int(get_demucs_attr(model, "audio_channels"))
    sources_list = list(get_demucs_attr(model, "sources"))

    demucs_input_wav = str(Path(temp_dir) / "demucs_input.wav")

    # Demucs ожидает свой samplerate / channels
    run_ffmpeg_to_wav(
        input_path=input_path,
        output_path=demucs_input_wav,
        sample_rate=samplerate,
        channels=audio_channels,
    )

    wav, sr = load_audio_sf(demucs_input_wav)
    if sr != samplerate:
        raise ValueError(f"Demucs expected {samplerate} Hz, got {sr}")

    if wav.shape[0] != audio_channels:
        raise ValueError(
            f"Demucs expected {audio_channels} channels, got {wav.shape[0]}"
        )

    print(f"[INFO] Запускаю Demucs на устройстве: {device}")
    with torch.inference_mode():
        sources = apply_model(
            model,
            wav[None].to(device),
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True,
            num_workers=0,
        )

    sources = sources[0].detach().cpu()

    if "vocals" not in sources_list:
        raise RuntimeError(f"'vocals' not found in Demucs sources: {sources_list}")

    vocals = sources[sources_list.index("vocals")]
    save_audio_sf(save_vocals_path, vocals, samplerate)

    del model
    del sources
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[INFO] Промежуточный файл сохранён: {Path(save_vocals_path).resolve()}")
    return save_vocals_path


def to_1d_audio(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()

    if x.ndim == 1:
        return x

    if x.ndim == 2:
        if x.shape[0] == 1:
            return x[0]
        if x.shape[1] == 1:
            return x[:, 0]
        if x.shape[0] < x.shape[1]:
            return x.mean(dim=0)
        return x.mean(dim=1)

    raise ValueError(f"Unexpected audio tensor shape: {tuple(x.shape)}")


def fix_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.numel() > target_len:
        return x[:target_len]
    if x.numel() < target_len:
        return torch.nn.functional.pad(x, (0, target_len - x.numel()))
    return x


def normalize_if_clipping(x: torch.Tensor, peak: float = 0.99) -> torch.Tensor:
    if x.numel() == 0:
        return x
    mx = float(x.abs().max())
    if mx > peak and mx > 0:
        x = x / mx * peak
    return x


def apply_fade_any(seg: torch.Tensor, fade_samples: int) -> torch.Tensor:
    seg = seg.clone()
    if seg.shape[-1] < 2:
        return seg

    fade_samples = min(fade_samples, seg.shape[-1] // 2)
    if fade_samples <= 0:
        return seg

    ramp = torch.linspace(0.0, 1.0, fade_samples, dtype=seg.dtype, device=seg.device)
    shape = [1] * seg.ndim
    shape[-1] = fade_samples
    ramp = ramp.view(*shape)

    seg[..., :fade_samples] *= ramp
    seg[..., -fade_samples:] *= torch.flip(ramp, dims=[-1])
    return seg


def enhance_in_chunks(
    enhancer,
    wav_path: str,
    device: str,
    chunk_sec: float = 30.0,
) -> torch.Tensor:
    """
    Очистка речи через SpeechBrain MetricGAN+ без записи чанков на диск.
    Возвращает 1D proxy-аудио с частотой MODEL_SR.
    """
    wav, sr = load_audio_sf(wav_path)
    if sr != MODEL_SR:
        raise ValueError(f"Expected {MODEL_SR} Hz, got {sr}")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.float()

    total_samples = wav.shape[1]
    chunk_samples = max(1, int(chunk_sec * sr))
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    enhanced_chunks = []

    print(f"[INFO] Усиление/очистка речи по чанкам: {num_chunks} шт.")

    for i in range(num_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, total_samples)
        chunk = wav[:, start:end]

        noisy = chunk.to(device)
        lengths = torch.tensor([1.0], dtype=torch.float32, device=device)

        with torch.inference_mode():
            try:
                enhanced = enhancer.enhance_batch(noisy, lengths=lengths)
            except TypeError:
                enhanced = enhancer.enhance_batch(noisy, lengths)

        if isinstance(enhanced, (tuple, list)):
            enhanced = enhanced[0]

        enhanced = to_1d_audio(enhanced)
        enhanced = fix_length(enhanced, chunk.shape[1])
        enhanced_chunks.append(enhanced)

        print(f"[INFO] Чанк {i + 1}/{num_chunks} готов")

        if device == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    out = torch.cat(enhanced_chunks, dim=0)
    return normalize_if_clipping(out)


def scale_timestamps(timestamps, src_sr: int, dst_sr: int):
    scale = dst_sr / src_sr
    out = []
    for ts in timestamps:
        out.append(
            {
                "start": int(round(ts["start"] * scale)),
                "end": int(round(ts["end"] * scale)),
            }
        )
    return out


def keep_only_speech_any(
    wav: torch.Tensor,
    speech_timestamps,
    sr: int,
    mode: str = "mute",
    fade_ms: int = 8,
) -> torch.Tensor:
    fade_samples = int(sr * fade_ms / 1000)

    if wav.ndim == 1:
        total_len = wav.shape[0]
    elif wav.ndim == 2:
        total_len = wav.shape[1]
    else:
        raise ValueError(f"Unexpected audio tensor shape: {tuple(wav.shape)}")

    if not speech_timestamps:
        if mode == "mute":
            return torch.zeros_like(wav)
        if wav.ndim == 1:
            return torch.zeros(1, dtype=wav.dtype)
        return torch.zeros((wav.shape[0], 1), dtype=wav.dtype)

    if mode == "cut":
        parts = []
        for ts in speech_timestamps:
            start = max(0, min(ts["start"], total_len))
            end = max(0, min(ts["end"], total_len))
            if end <= start:
                continue

            if wav.ndim == 1:
                seg = wav[start:end]
            else:
                seg = wav[:, start:end]

            if seg.numel() > 0:
                parts.append(apply_fade_any(seg, fade_samples))

        if not parts:
            if wav.ndim == 1:
                return torch.zeros(1, dtype=wav.dtype)
            return torch.zeros((wav.shape[0], 1), dtype=wav.dtype)

        return torch.cat(parts, dim=-1)

    out = torch.zeros_like(wav)
    for ts in speech_timestamps:
        start = max(0, min(ts["start"], total_len))
        end = max(0, min(ts["end"], total_len))
        if end <= start:
            continue

        if wav.ndim == 1:
            seg = wav[start:end]
            out[start:end] = apply_fade_any(seg, fade_samples)
        else:
            seg = wav[:, start:end]
            out[:, start:end] = apply_fade_any(seg, fade_samples)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Удаление всех звуков из аудио, кроме человеческой речи"
    )
    parser.add_argument("input", help="Путь к входному аудио/видео файлу")
    parser.add_argument("output", help="Путь к выходному WAV файлу")
    parser.add_argument(
        "--mode",
        choices=["mute", "cut"],
        default="mute",
        help="mute = сохранить тайминг и заглушить всё вне речи; cut = вырезать всё вне речи",
    )
    parser.add_argument(
        "--music-heavy",
        action="store_true",
        help="Сначала отделить vocals через Demucs. Полезно, если есть музыка/эффекты.",
    )
    parser.add_argument(
        "--demucs-fast",
        action="store_true",
        help="Использовать только 1 submodel Demucs вместо bag-of-models. Быстрее, но качество чуть хуже.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.45,
        help="Порог VAD. Ниже = больше шансов сохранить тихую речь, выше = агрессивнее режет мусор.",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=30.0,
        help="Размер чанка для обработки SpeechBrain",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Устройство для обработки",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    master_wav_path = make_intermediate_path(output_path, "00_master")
    demucs_vocals_path = make_intermediate_path(output_path, "01_demucs_vocals")
    prepared_wav_path = make_intermediate_path(output_path, "02_prepared_16k_mono")
    enhanced_wav_path = make_intermediate_path(output_path, "03_enhanced")

    device = resolve_device(args.device)
    print(f"[INFO] Device: {device}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) high-quality master
        if args.music_heavy:
            master_source = run_demucs_vocals(
                input_path=args.input,
                temp_dir=tmpdir,
                device=device,
                save_vocals_path=str(demucs_vocals_path),
                fast=args.demucs_fast,
            )
        else:
            run_ffmpeg_extract_wav_preserve(args.input, str(master_wav_path))
            master_source = str(master_wav_path)
            print(f"[INFO] Промежуточный файл сохранён: {master_wav_path.resolve()}")

        # 2) proxy 16k mono только для моделей
        run_ffmpeg_to_model_proxy(master_source, str(prepared_wav_path))
        print(f"[INFO] Промежуточный файл сохранён: {prepared_wav_path.resolve()}")

        # 3) SpeechBrain enhancement на proxy
        print("[INFO] Загружаю SpeechBrain MetricGAN+...")
        enhancer = load_metricgan_enhancer(device)

        enhanced = enhance_in_chunks(
            enhancer=enhancer,
            wav_path=str(prepared_wav_path),
            device=device,
            chunk_sec=args.chunk_sec,
        )

        save_audio_sf(str(enhanced_wav_path), enhanced.unsqueeze(0), MODEL_SR)
        print(f"[INFO] Промежуточный файл сохранён: {enhanced_wav_path.resolve()}")

        del enhancer
        if device == "cuda":
            torch.cuda.empty_cache()

        # 4) VAD на enhanced proxy
        print("[INFO] Загружаю Silero VAD...")
        vad_model = load_silero_vad()

        wav_for_vad = enhanced.float().cpu()

        speech_timestamps_proxy = get_speech_timestamps(
            wav_for_vad,
            vad_model,
            sampling_rate=MODEL_SR,
            threshold=args.vad_threshold,
            min_speech_duration_ms=200,
            min_silence_duration_ms=150,
            speech_pad_ms=80,
        )

        print(f"[INFO] Найдено речевых сегментов: {len(speech_timestamps_proxy)}")

        # 5) Переносим таймкоды на high-quality master и строим финальный output из master
        master_wav, master_sr = load_audio_sf(master_source)

        speech_timestamps_master = scale_timestamps(
            speech_timestamps_proxy,
            src_sr=MODEL_SR,
            dst_sr=master_sr,
        )

        speech_only = keep_only_speech_any(
            wav=master_wav,
            speech_timestamps=speech_timestamps_master,
            sr=master_sr,
            mode=args.mode,
            fade_ms=8,
        )

        speech_only = normalize_if_clipping(speech_only)
        save_audio_sf(str(output_path), speech_only, master_sr)

        print(f"[OK] Готово: {output_path.resolve()}")


if __name__ == "__main__":
    main()
