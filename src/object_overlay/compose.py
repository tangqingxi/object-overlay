from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .video import read_image, write_image


@dataclass(frozen=True)
class CompositionResult:
    png_path: Path
    jpg_path: Path
    mask_path: Path
    width: int
    height: int
    dtype: str
    background_max_difference: int


def _read_source_frame(
    frame_path: Path,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> np.ndarray:
    frame = read_image(frame_path, cv2.IMREAD_UNCHANGED)

    if frame is None:
        raise RuntimeError(f"无法按原始位深读取：{frame_path}")

    height, width = frame.shape[:2]

    if expected_width is not None and expected_height is not None:
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError(
                f"帧尺寸发生变化：{frame_path} 得到 {width}×{height}，"
                f"预期 {expected_width}×{expected_height}"
            )

    return frame


def _make_jpg_preview(image: np.ndarray) -> np.ndarray:
    preview = image

    if preview.dtype == np.uint16:
        preview = (
            preview.astype(np.float32) / 65535.0 * 255.0
        ).clip(0, 255).astype(np.uint8)
    elif preview.dtype != np.uint8:
        value_min = float(preview.min())
        value_max = float(preview.max())

        if value_max > value_min:
            preview = (
                (preview.astype(np.float32) - value_min)
                / (value_max - value_min)
                * 255.0
            ).clip(0, 255).astype(np.uint8)
        else:
            preview = np.zeros_like(preview, dtype=np.uint8)

    if preview.ndim == 3 and preview.shape[2] == 4:
        return cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR)

    return preview


def compose_overlay(
    frame_files: list[Path],
    masks: list[np.ndarray],
    base_index: int,
    output_dir: Path,
    mode: str,
) -> CompositionResult:
    """使用原始 PNG 像素生成无损多时刻目标叠加图。"""
    if len(frame_files) != len(masks):
        raise RuntimeError("视频帧数量与 Mask 数量不一致。")

    base = _read_source_frame(frame_files[base_index])
    height, width = base.shape[:2]
    output = base.copy()
    source_dtype = output.dtype
    union_mask = np.zeros((height, width), dtype=np.uint8)

    print()
    print(
        f"最终合成像素格式：dtype={source_dtype}, "
        f"channels={1 if output.ndim == 2 else output.shape[2]}, "
        f"size={width}×{height}"
    )

    for frame_index, frame_path in enumerate(frame_files):
        if frame_index == base_index:
            continue

        mask = masks[frame_index]

        if mask.shape != (height, width):
            raise RuntimeError(f"Mask 尺寸不一致：frame_{frame_index + 1:04d}")

        selected = mask > 0

        if not np.any(selected):
            continue

        source = _read_source_frame(frame_path, width, height)

        if source.dtype != source_dtype:
            raise RuntimeError(
                f"位深或数据类型不一致：{frame_path} 为 {source.dtype}，"
                f"背景为 {source_dtype}"
            )

        if source.ndim != output.ndim:
            raise RuntimeError(f"通道结构不一致：{frame_path}")

        if source.ndim == 3 and source.shape[2] != output.shape[2]:
            raise RuntimeError(f"通道数不一致：{frame_path}")

        # Mask 内直接复制对应时刻的原始像素，不进行缩放和混合。
        output[selected] = source[selected]
        union_mask[selected] = 255

    png_path = output_dir / f"overlay_{mode}.png"
    jpg_path = output_dir / f"overlay_{mode}.jpg"
    mask_path = output_dir / f"masks_{mode}.png"

    write_image(
        png_path,
        output,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    )
    write_image(
        jpg_path,
        _make_jpg_preview(output),
        [cv2.IMWRITE_JPEG_QUALITY, 98],
    )
    write_image(
        mask_path,
        union_mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    )

    outside_mask = union_mask == 0
    difference = cv2.absdiff(output, base)
    background_difference = difference[outside_mask]
    max_difference = (
        int(background_difference.max())
        if background_difference.size > 0
        else 0
    )

    return CompositionResult(
        png_path=png_path,
        jpg_path=jpg_path,
        mask_path=mask_path,
        width=width,
        height=height,
        dtype=str(source_dtype),
        background_max_difference=max_difference,
    )
