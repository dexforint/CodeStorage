#!/usr/bin/env python3
"""
Визуализация транскрибации и диаризации.

На выходе видео:
- субтитры цветом спикера;
- нижняя полоска на всю ширину (цвета = спикеры, тёмное = тишина);
- маркер текущего времени (слева начало, справа конец).

Примеры:
  python audio/visualize_transcript.py video.mp4 transcript_gigaam.txt
  python audio/visualize_transcript.py video.mp4 transcript_gigaam.txt -o out.mp4 --nvenc
  python audio/visualize_transcript.py video.mp4 transcript_gigaam.txt --preview 45
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TXT_SEGMENT_RE = re.compile(
    r"^\[(?P<start>[\d.]+)s\s*-\s*(?P<end>[\d.]+)s\]\s+"
    r"(?P<speaker>\S+):\s*(?P<text>.*)$"
)
SKIPPED_HEADER_RE = re.compile(r"Пропущенные сегменты", re.IGNORECASE)

SPEAKER_PALETTE: List[Tuple[int, int, int]] = [
    (244, 67, 54),
    (66, 165, 245),
    (102, 187, 106),
    (255, 167, 38),
    (171, 71, 188),
    (255, 202, 40),
    (38, 198, 218),
    (236, 64, 122),
    (141, 110, 99),
    (158, 158, 158),
]

LEGEND_H = 36
BAR_H = 42
PLAYHEAD_W = 4
MARGIN_BOTTOM = 14


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("Нужны ffmpeg и ffprobe в PATH")


def run_ffprobe(video_path: Path) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    rate = str(stream.get("r_frame_rate") or "25/1")
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den) if float(den) else 25.0
    else:
        fps = float(rate or 25.0)

    duration = stream.get("duration") or fmt.get("duration") or 0
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration=float(duration),
        fps=fps,
    )


def load_transcript(path: Path) -> List[Segment]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    return _load_txt(path)


def _load_json(path: Path) -> List[Segment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("segments") or raw.get("skipped") or []
    else:
        items = raw

    segments: List[Segment] = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if text.startswith("[") and text.endswith("]"):
            continue
        speaker = str(item.get("speaker") or "SPEAKER")
        start = float(item.get("start_time", item.get("start", 0.0)))
        end = float(item.get("end_time", item.get("end", start)))
        if end <= start:
            continue
        segments.append(Segment(speaker=speaker, start=start, end=end, text=text))
    return segments


def _load_txt(path: Path) -> List[Segment]:
    segments: List[Segment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if SKIPPED_HEADER_RE.search(line):
            break
        match = TXT_SEGMENT_RE.match(line.strip())
        if not match:
            continue
        text = match.group("text").strip()
        if not text or text.startswith("[too_short]") or text.startswith("[empty_"):
            continue
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end <= start:
            continue
        segments.append(
            Segment(
                speaker=match.group("speaker"),
                start=start,
                end=end,
                text=text,
            )
        )
    if not segments:
        raise ValueError(f"В файле нет сегментов: {path}")
    return segments


def clip_segments(
    segments: Sequence[Segment], start: float, end: float
) -> List[Segment]:
    clipped: List[Segment] = []
    for seg in segments:
        left = max(seg.start, start)
        right = min(seg.end, end)
        if right - left < 0.02:
            continue
        clipped.append(
            Segment(
                speaker=seg.speaker,
                start=left - start,
                end=right - start,
                text=seg.text,
            )
        )
    return clipped


def speaker_colors(speakers: Iterable[str]) -> Dict[str, Tuple[int, int, int]]:
    unique = list(dict.fromkeys(speakers))
    return {
        name: SPEAKER_PALETTE[i % len(SPEAKER_PALETTE)] for i, name in enumerate(unique)
    }


def rgb_to_ass(rgb: Tuple[int, int, int], alpha: int = 0) -> str:
    r, g, b = rgb
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}&"


def to_ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape_ass(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\N")
    )


def wrap_text(text: str, width: int = 90) -> str:
    words = text.split()
    if not words:
        return text
    lines: List[str] = []
    current: List[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and length + extra > width:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        lines.append(" ".join(current))
    return r"\N".join(lines)


def write_ass(
    path: Path,
    segments: Sequence[Segment],
    colors: Dict[str, Tuple[int, int, int]],
    play_res: Tuple[int, int],
    font_size: int,
    margin_v: int,
) -> None:
    width, height = play_res
    styles = [
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding"
    ]
    for speaker, rgb in colors.items():
        styles.append(
            "Style: {name},Arial,{size},{primary},&H000000FF,&H00101010,"
            "&H96000000,1,0,0,0,100,100,0,0,1,3,0,2,48,48,{margin},1".format(
                name=speaker,
                size=font_size,
                primary=rgb_to_ass(rgb),
                margin=margin_v,
            )
        )

    events = [
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]
    for seg in segments:
        if not seg.text:
            continue
        body = wrap_text(escape_ass(seg.text))
        label = escape_ass(seg.speaker.replace("SPEAKER_", "S"))
        events.append(
            "Dialogue: 0,{start},{end},{style},,0,0,0,,"
            "{{\\b1}}{label}{{\\b0}}  {body}".format(
                start=to_ass_time(seg.start),
                end=to_ass_time(seg.end),
                style=seg.speaker,
                label=label,
                body=body,
            )
        )

    content = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "YCbCr Matrix: TV.709",
            "",
            "[V4+ Styles]",
            *styles,
            "",
            "[Events]",
            *events,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _font(size: int):
    from PIL import ImageFont

    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def render_timeline(
    path: Path,
    segments: Sequence[Segment],
    colors: Dict[str, Tuple[int, int, int]],
    width: int,
    duration: float,
    legend_h: int = LEGEND_H,
    bar_h: int = BAR_H,
) -> None:
    from PIL import Image, ImageDraw

    duration = max(duration, 0.001)
    height = legend_h + bar_h
    image = Image.new("RGB", (width, height), (16, 16, 20))
    draw = ImageDraw.Draw(image)

    for seg in segments:
        x0 = int(seg.start / duration * width)
        x1 = max(x0 + 1, int(seg.end / duration * width))
        draw.rectangle([x0, legend_h, x1, height - 1], fill=colors[seg.speaker])

    draw.line([(0, legend_h), (width, legend_h)], fill=(70, 70, 78), width=1)

    font = _font(18)
    x = 10
    for speaker, rgb in colors.items():
        label = speaker.replace("SPEAKER_", "S")
        box = [x, 8, x + 14, 22]
        draw.rectangle(box, fill=rgb)
        draw.rectangle(box, outline=(240, 240, 240), width=1)
        draw.text((x + 20, 6), label, fill=(230, 230, 230), font=font)
        try:
            text_w = int(draw.textlength(label, font=font))
        except Exception:
            text_w = 8 * len(label)
        x += 20 + text_w + 18
        if x > width - 160:
            break

    image.save(path, format="PNG")


def escape_filter_path(path: Path) -> str:
    text = path.resolve().as_posix()
    return text.replace("\\", "/").replace(":", "\\:").replace("'", r"\'")


def build_filter(
    ass_path: Path,
    video_width: int,
    video_height: int,
    fps: float,
    timeline_h: int,
    bar_h: int,
    duration: float,
    font_path: Optional[Path],
) -> str:
    ass = escape_filter_path(ass_path)
    dur = max(duration, 0.001)
    fps = max(fps, 1.0)
    time_y = video_height + 8
    play_y = video_height + timeline_h - bar_h
    play_x = f"(main_w-{PLAYHEAD_W})*t/{dur:.6f}"

    parts = [
        f"[0:v]pad=iw:ih+{timeline_h}:0:0:color=0x101014[padded]",
        "[padded][1:v]overlay=0:main_h-overlay_h[withbar]",
        (
            f"color=c=0xFFFFFF:s={PLAYHEAD_W}x{bar_h}:d={dur:.6f}:r={fps:.6f},"
            "format=yuv420p[playhead]"
        ),
        f"[withbar][playhead]overlay=x='{play_x}':y={play_y}[marked]",
    ]

    drawtext = (
        f"[marked]drawtext=text='%{{pts\\:hms}}':"
        f"x={video_width}-tw-16:y={time_y}:"
        "fontsize=18:fontcolor=white:borderw=1:bordercolor=black@0.7"
    )
    if font_path and font_path.exists():
        drawtext += f":fontfile='{escape_filter_path(font_path)}'"
    parts.append(drawtext + "[timed]")
    parts.append(f"[timed]ass='{ass}'[vout]")
    return ";".join(parts)


def pick_fontfile() -> Optional[Path]:
    for candidate in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.exists():
            return candidate
    return None


def _parse_progress_seconds(key: str, value: str) -> Optional[float]:
    if value in {"N/A", ""}:
        return None
    if key == "out_time_ms" and value.lstrip("-").isdigit():
        raw = int(value)
        if raw < 0:
            return None
        return raw / 1_000_000.0 if raw > 10_000_000 else raw / 1000.0
    if key in {"out_time_us", "out_time_ns"} and value.lstrip("-").isdigit():
        raw = int(value)
        if raw < 0:
            return None
        return raw / (1_000_000.0 if key == "out_time_us" else 1_000_000_000.0)
    if key == "out_time":
        parts = value.split(":")
        if len(parts) != 3:
            return None
        try:
            hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None
        return hours * 3600 + minutes * 60 + seconds
    return None


def run_ffmpeg(
    video_path: Path,
    output_path: Path,
    timeline_path: Path,
    filter_complex: str,
    start: float,
    duration: float,
    fps: float,
    nvenc: bool,
    crf: int,
    preset: str,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
    ]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += [
        "-i",
        str(video_path),
        "-loop",
        "1",
        "-framerate",
        f"{fps:.6f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(timeline_path),
        "-t",
        f"{duration:.3f}",
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "0:a?",
        "-r",
        f"{fps:.6f}",
        "-fps_mode",
        "cfr",
    ]

    if nvenc:
        cmd += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            str(crf),
            "-b:v",
            "0",
        ]
    else:
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]

    cmd += [
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_path),
    ]

    logger.info("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    stderr_chunks: List[str] = []

    def drain_stderr() -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            stderr_chunks.append(line)

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=max(duration, 0.001), desc="Кодирование", unit="s")

    last = 0.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            current = _parse_progress_seconds(key, value)
            if current is None:
                continue
            current = min(max(current, 0.0), duration)
            if pbar is not None:
                pbar.update(max(0.0, current - last))
            last = current
    finally:
        code = proc.wait()
        reader.join(timeout=5)
        if pbar is not None:
            if code == 0 and last < duration:
                pbar.update(duration - last)
            pbar.close()

    stderr = "".join(stderr_chunks)
    if code != 0:
        raise RuntimeError(f"ffmpeg завершился с кодом {code}\n{stderr[-4000:]}")


def visualize(
    video_path: Path,
    transcript_path: Path,
    output_path: Path,
    start: float = 0.0,
    end: Optional[float] = None,
    nvenc: bool = False,
    crf: int = 20,
    preset: str = "veryfast",
    keep_temp: bool = False,
) -> None:
    check_ffmpeg()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)

    info = run_ffprobe(video_path)
    render_end = info.duration if end is None else min(end, info.duration)
    if render_end <= start:
        raise ValueError("Пустой интервал: --start/--end")
    render_duration = render_end - start

    segments = clip_segments(load_transcript(transcript_path), start, render_end)
    if not segments:
        raise ValueError("В выбранном интервале нет сегментов")

    colors = speaker_colors(seg.speaker for seg in segments)
    out_w, out_h = info.width, info.height + LEGEND_H + BAR_H
    font_size = max(28, int(info.height * 0.032))
    margin_v = LEGEND_H + BAR_H + MARGIN_BOTTOM

    logger.info(
        "Видео %dx%d, %.1f с; сегментов %d; спикеров %d; рендер %.1f с",
        info.width,
        info.height,
        info.duration,
        len(segments),
        len(colors),
        render_duration,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="viz_transcript_"))
    try:
        timeline_path = tmp_dir / "timeline.png"
        ass_path = tmp_dir / "subs.ass"
        render_timeline(
            timeline_path,
            segments,
            colors,
            width=info.width,
            duration=render_duration,
        )
        write_ass(
            ass_path,
            segments,
            colors,
            play_res=(out_w, out_h),
            font_size=font_size,
            margin_v=margin_v,
        )
        filter_complex = build_filter(
            ass_path,
            video_width=info.width,
            video_height=info.height,
            fps=info.fps,
            timeline_h=LEGEND_H + BAR_H,
            bar_h=BAR_H,
            duration=render_duration,
            font_path=pick_fontfile(),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(
            video_path=video_path,
            output_path=output_path,
            timeline_path=timeline_path,
            filter_complex=filter_complex,
            start=start,
            duration=render_duration,
            fps=info.fps,
            nvenc=nvenc,
            crf=crf,
            preset=preset,
        )
        logger.info("Готово: %s", output_path)
        if keep_temp:
            logger.info("Временные файлы: %s", tmp_dir)
    finally:
        if not keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Цветные субтитры и таймлайн спикеров поверх видео",
    )
    parser.add_argument("video", type=str, help="Путь к исходному видео")
    parser.add_argument(
        "transcript",
        type=str,
        help="Путь к transcript_gigaam.txt или JSON",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Выходной файл (по умолчанию <video>_viz.mp4)",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Старт, секунды")
    parser.add_argument("--end", type=float, default=None, help="Конец, секунды")
    parser.add_argument(
        "--preview",
        type=float,
        default=None,
        help="Короткий превью с начала, секунды (удобно для проверки)",
    )
    parser.add_argument("--nvenc", action="store_true", help="Кодек h264_nvenc")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", type=str, default="veryfast")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    transcript_path = Path(args.transcript)
    output_path = (
        Path(args.output)
        if args.output
        else video_path.with_name(video_path.stem + "_viz.mp4")
    )
    end = args.end
    if args.preview is not None:
        end = args.start + args.preview

    try:
        visualize(
            video_path=video_path,
            transcript_path=transcript_path,
            output_path=output_path,
            start=args.start,
            end=end,
            nvenc=args.nvenc,
            crf=args.crf,
            preset=args.preset,
            keep_temp=args.keep_temp,
        )
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
