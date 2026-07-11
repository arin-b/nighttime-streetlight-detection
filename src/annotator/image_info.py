from __future__ import annotations

import struct
from pathlib import Path


def image_size(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".png":
        return _png_size(data)
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_size(data)
    if suffix == ".bmp":
        return _bmp_size(data)
    if suffix == ".webp":
        return _webp_size(data)
    return (0, 0)


def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    return (0, 0)


def _bmp_size(data: bytes) -> tuple[int, int]:
    if len(data) >= 26 and data.startswith(b"BM"):
        width = struct.unpack("<i", data[18:22])[0]
        height = abs(struct.unpack("<i", data[22:26])[0])
        return int(width), int(height)
    return (0, 0)


def _webp_size(data: bytes) -> tuple[int, int]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return (0, 0)
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    return (0, 0)


def _jpeg_size(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return (0, 0)
    index = 2
    while index < len(data):
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            break
        length = struct.unpack(">H", data[index : index + 2])[0]
        if length < 2 or index + length > len(data):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if index + 7 <= len(data):
                height = struct.unpack(">H", data[index + 3 : index + 5])[0]
                width = struct.unpack(">H", data[index + 5 : index + 7])[0]
                return int(width), int(height)
        index += length
    return (0, 0)
