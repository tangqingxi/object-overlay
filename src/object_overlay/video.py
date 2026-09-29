import json
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def read_image(path: Path, flags: int) -> np.ndarray | None:
    """读取图片，并兼容 Windows 中文路径。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None

    if data.size == 0:
        return None

    return cv2.imdecode(data, flags)


def write_image(
    path: Path,
    image: np.ndarray,
    params: list[int] | None = None,
) -> None:
    """写入图片，并兼容 Windows 中文路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image, params or [])

    if not success:
        raise RuntimeError(f"图片编码失败：{path}")

    path.write_bytes(encoded.tobytes())


def cast_pixels(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """把浮点混合结果裁剪并还原为原始像素类型。"""
    target = np.dtype(dtype)

    if np.issubdtype(target, np.integer):
        info = np.iinfo(target)
        values = np.clip(values, info.min, info.max)

    return values.astype(target)


def _read_metadata(metadata_file: Path) -> dict:
    if not metadata_file.exists():
        return {}

    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"抽帧记录无法读取：{metadata_file}") from exc


def _write_metadata(
    metadata_file: Path,
    video_path: Path,
    interval: float,
) -> None:
    metadata_file.write_text(
        json.dumps(
            {
                "video": str(video_path),
                "interval": interval,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def update_metadata(metadata_file: Path, **fields: object) -> None:
    """把本次运行的附加信息合并进抽帧记录。

    调用方不应传入 video 或 interval——那是 prepare_frames 判断能否复用
    已有抽帧的依据，改掉会让缓存校验失效。
    """
    metadata = _read_metadata(metadata_file)
    metadata.update(fields)
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_ffmpeg() -> str:
    """优先使用系统 FFmpeg，否则使用 uv 安装的后备版本。"""
    system_ffmpeg = shutil.which("ffmpeg")

    if system_ffmpeg is not None:
        return system_ffmpeg

    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as exc:
        raise RuntimeError(
            "没有可用的 FFmpeg。请重新执行 uv sync，"
            "或安装系统版 FFmpeg。"
        ) from exc


def prepare_frames(
    video_path: Path,
    frames_dir: Path,
    metadata_file: Path,
    interval: float,
) -> list[Path]:
    """抽取视频帧；已有帧满足条件时直接复用。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(frames_dir.glob("frame_*.png"))
    metadata = _read_metadata(metadata_file)

    if existing_frames:
        old_interval = metadata.get("interval")

        if old_interval is not None and abs(float(old_interval) - interval) > 1e-12:
            raise RuntimeError(
                f"结果目录已有抽帧，记录间隔为 {old_interval} 秒，"
                f"本次请求为 {interval} 秒。请继续使用原间隔，"
                "或指定一个新的 --output-dir。"
            )

        old_video = metadata.get("video")

        if old_video is not None and Path(old_video).resolve() != video_path:
            raise RuntimeError(
                "结果目录属于另一个视频。请为当前视频指定新的 --output-dir。"
            )

        if not metadata:
            _write_metadata(metadata_file, video_path, interval)

        print(f"复用已有抽帧：{len(existing_frames)} 张")
        return existing_frames

    ffmpeg_executable = find_ffmpeg()
    fps = 1.0 / interval
    print(f"开始抽帧：每 {interval:g} 秒一张 PNG")

    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:g}",
        "-compression_level",
        "0",
        str(frames_dir / "frame_%04d.png"),
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg 抽帧失败，返回码：{exc.returncode}") from exc

    frame_files = sorted(frames_dir.glob("frame_*.png"))

    if not frame_files:
        raise RuntimeError("FFmpeg 已执行，但没有生成任何 PNG 帧。")

    _write_metadata(metadata_file, video_path, interval)
    print(f"抽帧完成：{frames_dir}")
    return frame_files


def load_preview_frames(
    frame_files: list[Path],
) -> tuple[list[np.ndarray], int, int]:
    """读取用于检测和界面预览的 8 位彩色帧。"""
    images: list[np.ndarray] = []
    expected_size: tuple[int, int] | None = None

    for path in frame_files:
        image = read_image(path, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"读取抽帧失败：{path}")

        height, width = image.shape[:2]

        if expected_size is None:
            expected_size = (height, width)
        elif expected_size != (height, width):
            raise RuntimeError(f"抽帧尺寸不一致：{path}")

        images.append(image)

    if not images or expected_size is None:
        raise RuntimeError("没有可读取的视频帧。")

    height, width = expected_size
    return images, width, height
