#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obs_live_markers_embed.py

- OBS пишет mp4
- во время записи ставим маркеры горячими клавишами
- после Stop Recording:
  - получаем путь к файлу через событие RecordStateChanged (outputPath)
    (fallback: ищем самый новый mp4 в папке записи)
  - переводим секунды в кадры по реальному FPS файла
  - встраиваем маркеры в MP4 через XMP (xmpDM:Tracks) используя ExifTool
  - сохраняем *.markers.json рядом с видео

Hotkeys:
  F8        - segment toggle (IN/OUT) -> Segmentation marker (duration)
  Ctrl+F8   - point marker -> Comment marker

Deps:
  pip install lxml pynput obsws-python
Needs in PATH:
  exiftool, ffprobe
"""

import os
import json
import time
import uuid
import queue
import argparse
import threading
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, List

from lxml import etree
from pynput import keyboard
from obsws_python import ReqClient, EventClient

# ──────────────────────────────────────────────────────────────────────────────
# XMP / Premiere helpers
# ──────────────────────────────────────────────────────────────────────────────

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "xmpDM": "http://ns.adobe.com/xmp/1.0/DynamicMedia/",
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
}


def q(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def exiftool_extract_xmp(video_path: str, exiftool: str = "exiftool") -> bytes:
    r = subprocess.run([exiftool, "-XMP", "-b", video_path], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            r.stderr.decode("utf-8", "ignore").strip() or "ExifTool extract failed"
        )
    return r.stdout


def exiftool_embed_xmp(
    video_path: str, xmp_bytes: bytes, exiftool: str = "exiftool"
) -> None:
    tmp = video_path + ".tmp.xmp"
    with open(tmp, "wb") as f:
        f.write(xmp_bytes)

    r = subprocess.run(
        [exiftool, f"-XMP<={tmp}", "-overwrite_original", video_path],
        capture_output=True,
        text=True,
    )
    os.remove(tmp)

    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "ExifTool embed failed")


def xmp_extract_xmpmeta_only(xmp_packet: bytes) -> Optional[bytes]:
    """
    Возвращает <x:xmpmeta>...</x:xmpmeta> из xpacket.
    Если XMP отсутствует — вернёт None.
    """
    if not xmp_packet or not xmp_packet.strip():
        return None
    start = xmp_packet.find(b"<x:xmpmeta")
    end = xmp_packet.rfind(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        return None
    end += len(b"</x:xmpmeta>")
    return xmp_packet[start:end]


def create_xmp_skeleton() -> etree._Element:
    """
    Если в mp4 нет XMP вообще — создаём минимальный корректный контейнер,
    чтобы Premiere увидел xmpDM:Tracks.
    """
    # Важно: nsmap задаём сразу с нужными префиксами (rdf/xmp/xmpDM/xmpMM),
    # чтобы сериализация получилась "читаемой" и похожей на Adobe.
    xmpmeta = etree.Element(
        q("x", "xmpmeta"),
        nsmap={
            "x": NS["x"],
            "rdf": NS["rdf"],
            "xmp": NS["xmp"],
            "xmpDM": NS["xmpDM"],
            "xmpMM": NS["xmpMM"],
        },
    )
    xmpmeta.set(q("x", "xmptk"), "Adobe XMP Core")

    rdf = etree.SubElement(xmpmeta, q("rdf", "RDF"))
    desc = etree.SubElement(rdf, q("rdf", "Description"))
    desc.set(q("rdf", "about"), "")
    return xmpmeta


def ffprobe_fps_fraction(video_path: str) -> Fraction:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "v:0",
        video_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "ffprobe failed")

    data = json.loads(r.stdout)
    stream = (data.get("streams") or [None])[0]
    if not stream:
        raise RuntimeError("No video stream found")

    raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not raw or raw == "0/0":
        raise RuntimeError("Cannot detect FPS")

    return Fraction(raw)


def adobe_framerate_code(fps: Fraction) -> str:
    mapping = {
        Fraction(24, 1): "f24",
        Fraction(25, 1): "f25",
        Fraction(30, 1): "f30",
        Fraction(50, 1): "f50",
        Fraction(60, 1): "f60",
        Fraction(24000, 1001): "f23976",
        Fraction(30000, 1001): "f2997",
        Fraction(60000, 1001): "f5994",
    }
    return mapping.get(fps, f"f{round(float(fps))}")


def sec_to_frames(seconds: float, fps: Fraction) -> int:
    val = Fraction(str(seconds)) * fps
    n, d = val.numerator, val.denominator
    return int((n + d // 2) // d)


@dataclass
class XmpMarker:
    start_frames: int
    duration_frames: int = 0
    name: str = ""
    comment: str = ""
    marker_type: str = "Comment"  # Comment | Chapter | Segmentation | WebLink
    web_url: Optional[str] = None
    guid: str = field(default_factory=lambda: str(uuid.uuid4()))


def build_tracks_element(
    markers: List[XmpMarker], frame_rate_code: str
) -> etree._Element:
    """
    Один track "Comment" (как часто делает Premiere),
    а тип маркера, если надо, кладём в xmpDM:type.
    """
    tracks = etree.Element(q("xmpDM", "Tracks"))
    bag = etree.SubElement(tracks, q("rdf", "Bag"))
    li = etree.SubElement(bag, q("rdf", "li"))

    track_desc = etree.SubElement(li, q("rdf", "Description"))
    track_desc.set(q("xmpDM", "trackName"), "Comment")
    track_desc.set(q("xmpDM", "trackType"), "Comment")
    track_desc.set(q("xmpDM", "frameRate"), frame_rate_code)

    markers_el = etree.SubElement(track_desc, q("xmpDM", "markers"))
    seq = etree.SubElement(markers_el, q("rdf", "Seq"))

    for m in markers:
        m_li = etree.SubElement(seq, q("rdf", "li"))
        m_desc = etree.SubElement(m_li, q("rdf", "Description"))

        m_desc.set(q("xmpDM", "startTime"), str(m.start_frames))
        if m.duration_frames > 0:
            m_desc.set(q("xmpDM", "duration"), str(m.duration_frames))
        m_desc.set(q("xmpDM", "guid"), m.guid)

        if m.name:
            m_desc.set(q("xmpDM", "name"), m.name)
        if m.comment:
            m_desc.set(q("xmpDM", "comment"), m.comment)
        if m.marker_type and m.marker_type != "Comment":
            m_desc.set(q("xmpDM", "type"), m.marker_type)
        if m.marker_type == "WebLink" and m.web_url:
            m_desc.set(q("xmpDM", "webResourceRef"), m.web_url)

        cpp = etree.SubElement(m_desc, q("xmpDM", "cuePointParams"))
        cpp_seq = etree.SubElement(cpp, q("rdf", "Seq"))
        etree.SubElement(
            cpp_seq,
            q("rdf", "li"),
            {q("xmpDM", "key"): "marker_guid", q("xmpDM", "value"): m.guid},
        )

    return tracks


def replace_tracks_in_xmp(
    xmpmeta: etree._Element, markers: List[XmpMarker], frame_rate_code: str
) -> etree._Element:
    desc = xmpmeta.find(".//rdf:Description", namespaces=NS)
    if desc is None:
        # если вдруг создали скелет криво или получили неожиданную структуру
        raise RuntimeError("No rdf:Description found in XMP")

    old_tracks = desc.find("xmpDM:Tracks", namespaces=NS)
    if old_tracks is not None:
        desc.remove(old_tracks)

    new_tracks = build_tracks_element(markers, frame_rate_code)

    history = desc.find("xmpMM:History", namespaces=NS)
    if history is not None:
        desc.insert(desc.index(history), new_tracks)
    else:
        desc.append(new_tracks)

    return xmpmeta


def load_xmpmeta_or_create(video_path: str, exiftool: str) -> etree._Element:
    """
    1) пытаемся извлечь существующий XMP
    2) если XMP нет — создаём скелет
    """
    pkt = exiftool_extract_xmp(video_path, exiftool=exiftool)
    xmpmeta_bytes = xmp_extract_xmpmeta_only(pkt)

    if xmpmeta_bytes is None:
        return create_xmp_skeleton()

    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    return etree.fromstring(xmpmeta_bytes, parser=parser)


def embed_markers_into_mp4(
    video_path: str, markers: List[XmpMarker], exiftool: str = "exiftool"
) -> None:
    xmpmeta = load_xmpmeta_or_create(video_path, exiftool=exiftool)

    fps = ffprobe_fps_fraction(video_path)
    fr_code = adobe_framerate_code(fps)

    replace_tracks_in_xmp(xmpmeta, markers, fr_code)

    out_bytes = etree.tostring(
        xmpmeta, pretty_print=True, xml_declaration=False, encoding="UTF-8"
    )
    exiftool_embed_xmp(video_path, out_bytes, exiftool=exiftool)


# ──────────────────────────────────────────────────────────────────────────────
# OBS helpers
# ──────────────────────────────────────────────────────────────────────────────


def parse_obs_timecode_to_seconds(tc: str) -> float:
    try:
        h, m, s = tc.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0


def wait_file_stable(path: str, timeout: float = 20.0) -> None:
    t0 = time.time()
    last = -1
    stable = 0
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = -1
            if size > 0 and size == last:
                stable += 1
                if stable >= 3:
                    return
            else:
                stable = 0
                last = size
        time.sleep(0.5)


def get_record_directory(obs_req) -> Optional[str]:
    try:
        r = obs_req.get_record_directory()
        return getattr(r, "record_directory", None)
    except Exception:
        return None


def find_latest_mp4(record_dir: str, after_ts: float) -> Optional[str]:
    if not record_dir or not os.path.isdir(record_dir):
        return None
    candidates = []
    for name in os.listdir(record_dir):
        if not name.lower().endswith(".mp4"):
            continue
        p = os.path.join(record_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            continue
        if mtime >= after_ts:
            candidates.append((mtime, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# ──────────────────────────────────────────────────────────────────────────────
# Logged markers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LoggedMarker:
    kind: str  # "segment" | "point"
    start_sec: float
    duration_sec: float = 0.0
    name: str = ""
    comment: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4455)
    ap.add_argument("--password", default="nikola3325tesla")
    ap.add_argument("--exiftool", default="exiftool")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()

    if not args.password:
        raise SystemExit(
            "Нужно указать --password (или переменную окружения OBS_WS_PASSWORD)."
        )

    obs_req = ReqClient(host=args.host, port=args.port, password=args.password)
    obs_evt = EventClient(host=args.host, port=args.port, password=args.password)

    stop_path_ready = threading.Event()
    last_stop_output_path = {"path": None}

    def on_record_state_changed(data, *rest):
        state = getattr(data, "output_state", None)
        outp = getattr(data, "output_path", None)
        if state == "OBS_WEBSOCKET_OUTPUT_STOPPED" and outp:
            last_stop_output_path["path"] = outp
            stop_path_ready.set()

    try:
        obs_evt.callback.register(on_record_state_changed, "RecordStateChanged")
    except TypeError:
        obs_evt.callback.register(on_record_state_changed)

    events_q: "queue.Queue[tuple[str, float]]" = queue.Queue()
    ctrl_pressed = {"down": False}

    recording_active = False
    rec_start_mono: Optional[float] = None
    rec_start_wall: Optional[float] = None
    record_dir: Optional[str] = None

    seg_open_t: Optional[float] = None
    logged: List[LoggedMarker] = []

    def on_press(key):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            ctrl_pressed["down"] = True
            return
        if key == keyboard.Key.f8 and ctrl_pressed["down"]:
            events_q.put(("point", time.monotonic()))
            return
        if key == keyboard.Key.f8 and not ctrl_pressed["down"]:
            events_q.put(("toggle", time.monotonic()))
            return

    def on_release(key):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            ctrl_pressed["down"] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Горячие клавиши:")
    print("  F8        — начать/закончить сегмент (Segmentation, duration)")
    print("  Ctrl+F8   — точечный маркер (Comment)")
    print("Ожидаю старт записи в OBS... (Ctrl+C чтобы выйти)\n")

    try:
        while True:
            st = obs_req.get_record_status()
            active = bool(st.output_active)
            tc = getattr(st, "output_timecode", "00:00:00.000")

            # START
            if active and not recording_active:
                elapsed = parse_obs_timecode_to_seconds(tc)
                rec_start_mono = time.monotonic() - elapsed
                rec_start_wall = time.time() - elapsed
                record_dir = get_record_directory(obs_req)

                logged = []
                seg_open_t = None

                stop_path_ready.clear()
                last_stop_output_path["path"] = None

                recording_active = True
                print(f"[REC] started. dir={record_dir}")

            # hotkeys
            while not events_q.empty():
                kind, t_mono = events_q.get_nowait()
                if not recording_active or rec_start_mono is None:
                    continue

                t_rel = t_mono - rec_start_mono
                if t_rel < 0:
                    continue

                if kind == "point":
                    idx = 1 + sum(1 for x in logged if x.kind == "point")
                    logged.append(
                        LoggedMarker(kind="point", start_sec=t_rel, name=f"Point {idx}")
                    )
                    print(f"  + point @ {t_rel:.3f}s")

                elif kind == "toggle":
                    if seg_open_t is None:
                        seg_open_t = t_rel
                        print(f"  [seg] IN  @ {seg_open_t:.3f}s")
                    else:
                        start = seg_open_t
                        dur = max(0.0, t_rel - seg_open_t)
                        idx = 1 + sum(1 for x in logged if x.kind == "segment")
                        logged.append(
                            LoggedMarker(
                                kind="segment",
                                start_sec=start,
                                duration_sec=dur,
                                name=f"Seg {idx}",
                            )
                        )
                        print(f"  [seg] OUT @ {t_rel:.3f}s  (dur={dur:.3f}s)")
                        seg_open_t = None

            # STOP
            if not active and recording_active:
                recording_active = False

                stop_rel = (
                    (time.monotonic() - rec_start_mono) if rec_start_mono else None
                )
                if seg_open_t is not None and stop_rel is not None:
                    start = seg_open_t
                    dur = max(0.0, stop_rel - seg_open_t)
                    idx = 1 + sum(1 for x in logged if x.kind == "segment")
                    logged.append(
                        LoggedMarker(
                            kind="segment",
                            start_sec=start,
                            duration_sec=dur,
                            name=f"Seg {idx}",
                        )
                    )
                    print(f"  [seg] auto-close at STOP (dur={dur:.3f}s)")
                    seg_open_t = None

                # 1) event outputPath
                stop_path_ready.wait(timeout=4.0)
                video_path = last_stop_output_path["path"]

                # 2) fallback: latest mp4 in record dir
                if (not video_path) and record_dir and rec_start_wall:
                    video_path = find_latest_mp4(
                        record_dir, after_ts=rec_start_wall - 2
                    )

                if not video_path:
                    print("[STOP] Не удалось определить путь к файлу записи.")
                    print("Проверь OBS: Settings → Output → Recording Path.")
                    break

                video_path = os.path.normpath(video_path)
                print(f"[REC] stopped. file={video_path}")

                # JSON рядом
                json_path = os.path.splitext(video_path)[0] + ".markers.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [lm.__dict__ for lm in logged], f, ensure_ascii=False, indent=2
                    )
                print(f"JSON маркеров: {json_path}")

                if args.no_embed:
                    print("Встраивание отключено (--no-embed).")
                    break

                wait_file_stable(video_path)

                fps = ffprobe_fps_fraction(video_path)
                xmp_markers: List[XmpMarker] = []
                for lm in logged:
                    start_fr = sec_to_frames(lm.start_sec, fps)
                    dur_fr = (
                        sec_to_frames(lm.duration_sec, fps)
                        if lm.kind == "segment"
                        else 0
                    )
                    mtype = "Segmentation" if lm.kind == "segment" else "Comment"
                    xmp_markers.append(
                        XmpMarker(
                            start_frames=start_fr,
                            duration_frames=dur_fr,
                            name=lm.name,
                            comment=lm.comment,
                            marker_type=mtype,
                        )
                    )

                embed_markers_into_mp4(video_path, xmp_markers, exiftool=args.exiftool)
                print("OK: маркеры встроены в MP4. Импортируй файл в Premiere.")
                break

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nExit.")
    finally:
        try:
            listener.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
