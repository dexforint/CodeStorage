#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import uuid
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Iterable, Dict, List
from lxml import etree

# ──────────────────────────────────────────────────────────────────────────────
# XMP namespaces (как в твоём дампе)
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


# ──────────────────────────────────────────────────────────────────────────────
# FPS: берём точную дробь из ffprobe (важно для 30000/1001 и т.п.)
# ──────────────────────────────────────────────────────────────────────────────


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
        raise RuntimeError("Cannot detect FPS from ffprobe")

    return Fraction(raw)  # например 30000/1001


def adobe_framerate_code(fps: Fraction) -> str:
    """
    Premiere/Adobe XMP обычно использует эти коды:
      24 -> f24
      25 -> f25
      30 -> f30
      50 -> f50
      60 -> f60
      24000/1001 -> f23976
      30000/1001 -> f2997
      60000/1001 -> f5994
    """
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
    if fps in mapping:
        return mapping[fps]

    # fallback: ближайшее целое (если вдруг экзотика)
    return f"f{round(float(fps))}"


def sec_to_frames(seconds: float, fps: Fraction) -> int:
    # округление к ближайшему кадру
    val = Fraction(str(seconds)) * fps
    # round для Fraction делаем вручную
    n, d = val.numerator, val.denominator
    return int((n + d // 2) // d)


# ──────────────────────────────────────────────────────────────────────────────
# Маркеры
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Marker:
    start: int  # в кадрах
    duration: int = 0  # в кадрах; 0 => точечный (duration атрибут можно НЕ писать)
    name: str = ""
    comment: str = ""
    track_type: str = "Comment"  # Comment | Chapter | Segmentation | WebLink
    web_url: Optional[str] = None
    guid: str = field(default_factory=lambda: str(uuid.uuid4()))


# ──────────────────────────────────────────────────────────────────────────────
# ExifTool: extract / embed XMP
# ──────────────────────────────────────────────────────────────────────────────


def exiftool_extract_xmp(video_path: str, exiftool: str = "exiftool") -> bytes:
    cmd = [exiftool, "-XMP", "-b", video_path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            r.stderr.decode("utf-8", "ignore").strip() or "ExifTool failed"
        )
    return r.stdout  # это xpacket с padding


def xmp_extract_xmpmeta_only(xmp_packet: bytes) -> bytes:
    """
    Вырезаем ровно <x:xmpmeta>...</x:xmpmeta> чтобы lxml не спотыкался об xpacket begin="﻿"
    и огромные padding-блоки.
    """
    start = xmp_packet.find(b"<x:xmpmeta")
    end = xmp_packet.rfind(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        raise RuntimeError("Cannot find <x:xmpmeta> in extracted XMP")
    end += len(b"</x:xmpmeta>")
    return xmp_packet[start:end]


def exiftool_embed_xmp(
    video_path: str, xmp_bytes: bytes, exiftool: str = "exiftool"
) -> None:
    tmp = video_path + ".tmp.xmp"
    with open(tmp, "wb") as f:
        f.write(xmp_bytes)

    cmd = [exiftool, f"-XMP<={tmp}", "-overwrite_original", video_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tmp)

    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "ExifTool embed failed")


# ──────────────────────────────────────────────────────────────────────────────
# Патчим существующий XMP: заменяем/создаём xmpDM:Tracks
# ──────────────────────────────────────────────────────────────────────────────


def find_main_description(xmpmeta: etree._Element) -> etree._Element:
    # Обычно один rdf:Description
    desc = xmpmeta.find(".//rdf:Description", namespaces=NS)
    if desc is None:
        raise RuntimeError("No rdf:Description found in XMP")
    return desc


def build_tracks_element(
    markers: Iterable[Marker], frame_rate_code: str
) -> etree._Element:
    """
    Строим xmpDM:Tracks в структуре, максимально похожей на Premiere:

    <xmpDM:Tracks>
      <rdf:Bag>
        <rdf:li>
          <rdf:Description xmpDM:trackName="Comment" xmpDM:trackType="Comment" xmpDM:frameRate="f30">
            <xmpDM:markers>
              <rdf:Seq>
                <rdf:li>
                  <rdf:Description xmpDM:startTime="407" xmpDM:duration="843" xmpDM:guid="...">
                    <xmpDM:cuePointParams>...
    """
    # группируем по track_type (чтобы Chapter/Segmentation не смешивать при желании)
    grouped: Dict[str, List[Marker]] = {}
    for m in markers:
        grouped.setdefault(m.track_type or "Comment", []).append(m)

    tracks = etree.Element(q("xmpDM", "Tracks"))
    bag = etree.SubElement(tracks, q("rdf", "Bag"))

    for track_type, ms in grouped.items():
        li = etree.SubElement(bag, q("rdf", "li"))
        track_desc = etree.SubElement(li, q("rdf", "Description"))
        track_desc.set(q("xmpDM", "trackName"), track_type)
        track_desc.set(q("xmpDM", "trackType"), track_type)
        track_desc.set(q("xmpDM", "frameRate"), frame_rate_code)

        markers_el = etree.SubElement(track_desc, q("xmpDM", "markers"))
        seq = etree.SubElement(markers_el, q("rdf", "Seq"))

        for m in ms:
            m_li = etree.SubElement(seq, q("rdf", "li"))
            m_desc = etree.SubElement(m_li, q("rdf", "Description"))
            m_desc.set(q("xmpDM", "startTime"), str(m.start))
            if m.duration > 0:
                m_desc.set(q("xmpDM", "duration"), str(m.duration))
            m_desc.set(q("xmpDM", "guid"), m.guid)

            # Эти поля Premiere пишет не всегда; но они нормально читаются, если нужны
            if m.name:
                m_desc.set(q("xmpDM", "name"), m.name)
            if m.comment:
                m_desc.set(q("xmpDM", "comment"), m.comment)
            if m.track_type and m.track_type != "Comment":
                # можно писать type, но чаще достаточно trackType
                m_desc.set(q("xmpDM", "type"), m.track_type)
            if m.track_type == "WebLink" and m.web_url:
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
    xmpmeta: etree._Element, markers: list[Marker], frame_rate_code: str
) -> etree._Element:
    desc = find_main_description(xmpmeta)

    # удалить старый xmpDM:Tracks (если есть)
    old_tracks = desc.find("xmpDM:Tracks", namespaces=NS)
    if old_tracks is not None:
        desc.remove(old_tracks)

    new_tracks = build_tracks_element(markers, frame_rate_code)

    # вставляем перед xmpMM:History, чтобы порядок совпадал с Premiere (не обязательно, но полезно)
    history = desc.find("xmpMM:History", namespaces=NS)
    if history is not None:
        idx = desc.index(history)
        desc.insert(idx, new_tracks)
    else:
        desc.append(new_tracks)

    return xmpmeta


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def embed_premiere_markers(
    video_path: str, markers: list[Marker], exiftool: str = "exiftool"
) -> None:
    xmp_packet = exiftool_extract_xmp(video_path, exiftool=exiftool)
    xmpmeta_bytes = xmp_extract_xmpmeta_only(xmp_packet)

    parser = etree.XMLParser(recover=False, remove_blank_text=False, huge_tree=True)
    xmpmeta = etree.fromstring(xmpmeta_bytes, parser=parser)

    fps = ffprobe_fps_fraction(video_path)
    fr_code = adobe_framerate_code(fps)

    replace_tracks_in_xmp(xmpmeta, markers, fr_code)

    out_bytes = etree.tostring(
        xmpmeta,
        pretty_print=True,
        xml_declaration=False,  # exiftool сам сделает корректный xpacket
        encoding="UTF-8",
    )
    exiftool_embed_xmp(video_path, out_bytes, exiftool=exiftool)


if __name__ == "__main__":
    VIDEO = r"C:\Users\user\Videos\video.mp4"

    fps = ffprobe_fps_fraction(VIDEO)

    def sec(s: float) -> int:
        return sec_to_frames(s, fps)

    markers = [
        Marker(
            start=sec(0.0),
            name="Открывающий кадр",
            comment="Логотип + заставка",
            track_type="Comment",
        ),
        Marker(
            start=sec(3.0),
            name="Глава 1: Интро",
            comment="Начало",
            track_type="Chapter",
        ),
        Marker(
            start=sec(6.0),
            duration=sec(2.0),
            name="Музыка",
            comment="Поднять громкость",
            track_type="Comment",
        ),
        Marker(
            start=sec(20.0), name="Глава 2", comment="Поворот", track_type="Chapter"
        ),
        Marker(
            start=sec(30.0),
            name="Ad break",
            comment="Точка рекламы",
            track_type="Segmentation",
        ),
    ]

    embed_premiere_markers(VIDEO, markers, exiftool="exiftool")
    print("OK: маркеры встроены.")
