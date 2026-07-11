from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image


def extract_frames(video: Path, out: Path, fps_sample: float, max_frames: int | None = None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"could not open video: {video}")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(source_fps / max(0.1, fps_sample))))
        frame_paths: list[Path] = []
        frame_index = 0
        sampled_index = 1
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if (frame_index - 1) % step != 0:
                    continue
                target = out / f"{sampled_index:06d}.jpg"
                cv2.imwrite(str(target), frame)
                frame_paths.append(target)
                sampled_index += 1
                if max_frames is not None and len(frame_paths) >= max_frames:
                    break
        finally:
            cap.release()
        return frame_paths
    except ImportError:
        return extract_frames_with_ffmpeg(video, out, fps_sample=fps_sample, max_frames=max_frames)


def extract_frames_with_ffmpeg(video: Path, out: Path, fps_sample: float, max_frames: int | None = None) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("frame extraction requires either opencv-python or ffmpeg on PATH")
    pattern = out / "%06d.jpg"
    command = [ffmpeg, "-y", "-i", str(video), "-vf", f"fps={fps_sample}", "-q:v", "2"]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(pattern))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(out.glob("*.jpg"))


def write_mp4(frames: list[Path], path: Path, fps: float) -> Path:
    try:
        import cv2

        first = cv2.imread(str(frames[0]))
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            for frame in frames:
                image = cv2.imread(str(frame))
                if image is not None:
                    writer.write(image)
        finally:
            writer.release()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("MP4 rendering requires either opencv-python or ffmpeg on PATH")
        staging = path.parent / f"_{path.stem}_frames"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            for index, frame in enumerate(frames, start=1):
                shutil.copy2(frame, staging / f"frame_{index:06d}.jpg")
            subprocess.run(
                [ffmpeg, "-y", "-framerate", str(fps), "-i", str(staging / "frame_%06d.jpg"), "-pix_fmt", "yuv420p", str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return path


def write_contact_sheet(frames: list[Path], path: Path, columns: int = 4, rows: int = 4) -> Path:
    selected = frames[:: max(1, len(frames) // (columns * rows))][: columns * rows]
    thumbs = []
    for frame in selected:
        image = Image.open(frame).convert("RGB")
        image.thumbnail((360, 220), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 220), (25, 25, 25))
        canvas.paste(image, ((360 - image.width) // 2, (220 - image.height) // 2))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (columns * 360, rows * 220), (35, 35, 35))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % columns) * 360, (idx // columns) * 220))
    sheet.save(path)
    return path


def write_representative_frames(frames: list[Path], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if not frames:
        return out
    indices = sorted(set([0, len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4, len(frames) - 1]))
    for rank, index in enumerate(indices, start=1):
        shutil.copy2(frames[index], out / f"representative_{rank:02d}_{frames[index].name}")
    return out
