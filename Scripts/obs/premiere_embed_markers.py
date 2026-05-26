#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Iterable, Dict, List

from lxml import etree

# --- namespaces ---
NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "xmpDM": "http://ns.adobe.com/xmp/1.0/DynamicMedia/",
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
    "stDim": "http://ns.adobe.com/xap/1.0/sType/Dimensions#",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "stEvt": "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
}

for p, u in NS.items():
    etree.register_namespace(p, u)


def q(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def run_capture(cmd: list[str]) -> bytes:
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            (r.stderr or b"").decode("utf-8", "ignore").strip() or "Command failed"
        )
    return r.stdout


def ffprobe_fps_fraction(video_path: str, ffprobe: str) -> Fraction:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=nk=1:nw=1",
        video_path,
    ]
    out = run_capture(cmd).decode("utf-8", "ignore").strip()
    if not out or out == "0/0":
        raise RuntimeError("Cannot detect FPS from ffprobe")
    return Fraction(out)


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


def ms_to_frames(ms: int, fps: Fraction) -> int:
    # round(ms/1000 * fps)
    val = Fraction(ms, 1000) * fps
    n, d = val.numerator, val.denominator
    return int((n + d // 2) // d)


@dataclass
class Marker:
    start: int  # frames
    duration: int = 0  # frames
    name: str = ""
    comment: str = ""
    track_type: str = "Comment"
    guid: str = field(default_factory=lambda: str(uuid.uuid4()))


def exiftool_extract_xmp(video_path: str, exiftool: str) -> bytes:
    # -b => raw XMP packet if exists, else empty
    cmd = [exiftool, "-XMP", "-b", video_path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            r.stderr.decode("utf-8", "ignore").strip() or "ExifTool failed"
        )
    return r.stdout


def xmp_extract_xmpmeta_only(xmp_packet: bytes) -> Optional[bytes]:
    start = xmp_packet.find(b"<x:xmpmeta")
    end = xmp_packet.rfind(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        return None
    end += len(b"</x:xmpmeta>")
    return xmp_packet[start:end]


def make_minimal_xmpmeta_like_adobe() -> etree._Element:
    # Минимальный валидный xmpmeta, если в MP4 XMP отсутствовал
    xmpmeta = etree.Element(q("x", "xmpmeta"), nsmap={"x": NS["x"]})
    rdf = etree.SubElement(xmpmeta, q("rdf", "RDF"), nsmap={"rdf": NS["rdf"]})

    # В дампе Adobe namespaces объявлены на rdf:Description — делаем так же.
    desc = etree.SubElement(
        rdf,
        q("rdf", "Description"),
        nsmap={
            "xmp": NS["xmp"],
            "xmpDM": NS["xmpDM"],
            "xmpMM": NS["xmpMM"],
            "stDim": NS["stDim"],
            "tiff": NS["tiff"],
            "stEvt": NS["stEvt"],
        },
    )
    desc.set(q("rdf", "about"), "")
    return xmpmeta


def find_main_description(xmpmeta: etree._Element) -> etree._Element:
    desc = xmpmeta.find(".//rdf:Description", namespaces=NS)
    if desc is None:
        raise RuntimeError("No rdf:Description found in XMP")
    return desc


def build_tracks_element(
    markers: Iterable[Marker], frame_rate_code: str
) -> etree._Element:
    # Группируем по track_type (Comment/Chapter/Segmentation/WebLink)
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

            # ВАЖНО: startTime/duration — атрибуты, в кадрах, как в вашем дампе
            m_desc.set(q("xmpDM", "startTime"), str(int(m.start)))
            if m.duration and m.duration > 0:
                m_desc.set(q("xmpDM", "duration"), str(int(m.duration)))
            m_desc.set(q("xmpDM", "guid"), m.guid)

            # Имя/коммент: Premiere может хранить по-разному, пишем мягко (не обяз.)
            if m.name:
                m_desc.set(q("xmpDM", "name"), m.name)
            if m.comment:
                m_desc.set(q("xmpDM", "comment"), m.comment)

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
) -> None:
    desc = find_main_description(xmpmeta)

    old_tracks = desc.find("xmpDM:Tracks", namespaces=NS)
    if old_tracks is not None:
        desc.remove(old_tracks)

    new_tracks = build_tracks_element(markers, frame_rate_code)

    # как у Premiere: Tracks перед xmpMM:History
    history = desc.find("xmpMM:History", namespaces=NS)
    if history is not None:
        idx = desc.index(history)
        desc.insert(idx, new_tracks)
    else:
        desc.append(new_tracks)


def wrap_xpacket(xmpmeta_bytes: bytes) -> bytes:
    # BOM внутри begin выглядит как "я╗┐" в cp1251-консолях, но это норм.
    header = '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
    footer = "\n" + ("\n" * 20) + '<?xpacket end="w"?>\n'
    return header.encode("utf-8") + xmpmeta_bytes + footer.encode("utf-8")


def exiftool_embed_xmp(
    video_path: str, xmp_packet_or_meta: bytes, exiftool: str
) -> None:
    tmp = video_path + ".tmp.xmp"
    with open(tmp, "wb") as f:
        f.write(xmp_packet_or_meta)

    cmd = [exiftool, f"-XMP<={tmp}", "-overwrite_original", video_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tmp)

    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "ExifTool embed failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument(
        "--markers", required=True, help="JSON from OBS (*.mp4.markers.json)"
    )
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--exiftool", default="exiftool")
    ap.add_argument(
        "--write-sidecar",
        action="store_true",
        help="Also write video.mp4.xmp for debugging",
    )
    args = ap.parse_args()

    with open(args.markers, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_path = args.video
    fps = ffprobe_fps_fraction(video_path, ffprobe=args.ffprobe)
    fr_code = adobe_framerate_code(fps)

    markers: list[Marker] = []
    for m in data.get("markers", []):
        start_f = ms_to_frames(int(m["start_ms"]), fps)
        dur_f = ms_to_frames(int(m.get("duration_ms", 0)), fps)
        markers.append(
            Marker(
                start=start_f,
                duration=dur_f,
                name=m.get("name", ""),
                comment=m.get("comment", ""),
                track_type=m.get("track_type", "Comment"),
            )
        )

    # parse existing XMP if exists, else create minimal
    xmp_packet = exiftool_extract_xmp(video_path, exiftool=args.exiftool)
    xmpmeta_bytes = xmp_extract_xmpmeta_only(xmp_packet)

    if not xmpmeta_bytes:
        xmpmeta = make_minimal_xmpmeta_like_adobe()
    else:
        parser = etree.XMLParser(recover=False, remove_blank_text=False, huge_tree=True)
        xmpmeta = etree.fromstring(xmpmeta_bytes, parser=parser)

    replace_tracks_in_xmp(xmpmeta, markers, fr_code)

    out_xmpmeta = etree.tostring(
        xmpmeta,
        pretty_print=True,
        xml_declaration=False,
        encoding="UTF-8",
    )

    out_packet = wrap_xpacket(out_xmpmeta)

    if args.write_sidecar:
        with open(video_path + ".xmp", "wb") as f:
            f.write(out_packet)

    exiftool_embed_xmp(video_path, out_packet, exiftool=args.exiftool)


if __name__ == "__main__":
    main()
