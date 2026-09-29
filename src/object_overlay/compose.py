from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .trajectory import (
    TRAJECTORY_COLOR,
    draw_trajectory,
    mask_center,
    smooth_path,
)
from .video import cast_pixels, read_image, write_image


# 残影版中最早时刻的不透明度，最新时刻始终为 1.0。
GHOST_MIN_ALPHA = 0.15


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

    png_path = output_dir / "overlay.png"
    jpg_path = output_dir / "overlay.jpg"
    mask_path = output_dir / "masks.png"

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


def _read_matching_source(
    frame_path: Path,
    canvas: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """读取位深与通道结构都与画布一致的原始帧。"""
    source = _read_source_frame(frame_path, width, height)

    if source.dtype != canvas.dtype:
        raise RuntimeError(
            f"位深或数据类型不一致：{frame_path} 为 {source.dtype}，"
            f"背景为 {canvas.dtype}"
        )

    if source.ndim != canvas.ndim:
        raise RuntimeError(f"通道结构不一致：{frame_path}")

    if source.ndim == 3 and source.shape[2] != canvas.shape[2]:
        raise RuntimeError(f"通道数不一致：{frame_path}")

    return source


def _normalized_times(frame_indices: list[int]) -> list[float]:
    """把各时刻按实际帧序号归一化到 0..1。

    用真实序号而不是选中次序：人工跳帧造成的时间间隔不均匀，应该如实
    反映在渐隐上，这符合多次曝光的物理含义。
    """
    if len(frame_indices) < 2:
        return [1.0 for _ in frame_indices]

    span = frame_indices[-1] - frame_indices[0]

    if span <= 0:
        return [1.0 for _ in frame_indices]

    return [
        (frame_index - frame_indices[0]) / span
        for frame_index in frame_indices
    ]


def _erase_background_object(
    canvas: np.ndarray,
    mask: np.ndarray,
    frame_files: list[Path],
    reference_indices: list[int],
) -> None:
    """用其余时刻在该区域的时域中值替换画布上固定的目标（就地修改）。

    中值对少数派不敏感：目标只在少数帧里覆盖这些像素，其余帧给出的是干净
    背景，因此中值就是背景本身。边缘做羽化过渡，避免替换边界出现接缝。
    """
    rows, columns = np.nonzero(mask)

    if columns.size == 0 or not reference_indices:
        return

    height, width = canvas.shape[:2]
    padding = max(16, int(round(width / 100)))
    x0 = max(0, int(columns.min()) - padding)
    y0 = max(0, int(rows.min()) - padding)
    x1 = min(width, int(columns.max()) + padding + 1)
    y1 = min(height, int(rows.max()) + padding + 1)

    stack = []

    for frame_index in reference_indices:
        frame = _read_matching_source(
            frame_files[frame_index],
            canvas,
            width,
            height,
        )
        stack.append(frame[y0:y1, x0:x1])

    replacement = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)

    feather = cv2.GaussianBlur(
        mask[y0:y1, x0:x1],
        (0, 0),
        max(1.5, width / 800),
    ).astype(np.float32) / 255.0
    feather = feather.reshape(feather.shape + (1,) * (canvas.ndim - 2))

    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = cast_pixels(
        region + (replacement - region) * feather,
        canvas.dtype,
    )


def _draw_path(
    canvas: np.ndarray,
    moments: list[tuple[int, tuple[float, float]]],
    taus: list[float],
    min_alpha: float,
    color: tuple[int, int, int],
    outline: bool,
) -> None:
    """在画布上绘制轨迹线，有效目标不足两个时跳过。"""
    if len(moments) < 2:
        print("有效目标不足两个，跳过轨迹线绘制。")
        return

    samples, sample_taus = smooth_path(
        [center for _, center in moments],
        taus,
    )
    draw_trajectory(canvas, samples, sample_taus, min_alpha, color, outline)


def compose_ghost_overlay(
    frame_files: list[Path],
    masks: list[np.ndarray],
    base_index: int,
    output_dir: Path,
    min_alpha: float = GHOST_MIN_ALPHA,
    background_mask: np.ndarray | None = None,
    show_trajectory: bool = True,
    trajectory_color: tuple[int, int, int] = TRAJECTORY_COLOR,
    trajectory_outline: bool = True,
) -> Path:
    """生成带轨迹线与时间渐隐的叠加图。

    与 compose_overlay 不同，这里会按时间混合像素，因此产出的不是无损
    图像：越早的时刻越透明，最新的时刻完全不透明。

    background_mask 给出背景帧里目标的位置。背景帧并非纯背景，它里面的
    目标同样是序列中的一个时刻：先把它从画布上擦掉，再让它作为第一个
    时刻参与渐隐，否则它会以完整不透明留在画面里、脱离轨迹线。
    """
    if len(frame_files) != len(masks):
        raise RuntimeError("视频帧数量与 Mask 数量不一致。")

    base = _read_source_frame(frame_files[base_index])
    height, width = base.shape[:2]
    canvas = base.copy()

    print()
    print("正在生成残影版叠加图...")

    effective_masks = list(masks)

    if background_mask is not None:
        effective_masks[base_index] = background_mask

    moments: list[tuple[int, tuple[float, float]]] = []

    for frame_index, mask in enumerate(effective_masks):
        if mask.shape != (height, width):
            raise RuntimeError(f"Mask 尺寸不一致：frame_{frame_index + 1:04d}")

        center = mask_center(mask)

        if center is not None:
            moments.append((frame_index, center))

    taus = _normalized_times([frame_index for frame_index, _ in moments])

    if np.any(effective_masks[base_index] > 0):
        _erase_background_object(
            canvas,
            effective_masks[base_index],
            frame_files,
            [
                frame_index
                for frame_index, _ in moments
                if frame_index != base_index
            ],
        )
        print(
            "背景帧里的目标已按其余时刻的时域中值擦除："
            f"frame_{base_index + 1:04d}"
        )

    # 关掉轨迹线不影响上面那步擦除：擦除属于渐隐语义，背景帧里的目标仍要
    # 作为最淡的第一个时刻参与合成。
    if show_trajectory:
        _draw_path(
            canvas,
            moments,
            taus,
            min_alpha,
            trajectory_color,
            trajectory_outline,
        )

    for (frame_index, _center), tau in zip(moments, taus):
        selected = effective_masks[frame_index] > 0

        if not np.any(selected):
            continue

        alpha = min_alpha + (1.0 - min_alpha) * tau
        source = _read_matching_source(
            frame_files[frame_index],
            canvas,
            width,
            height,
        )

        # 顺序累积而非各自与背景混合：这是多次曝光的物理模型，
        # 最新时刻 alpha=1，最终完全实心。
        original = canvas[selected].astype(np.float32)
        target = source[selected].astype(np.float32)
        canvas[selected] = cast_pixels(
            original + (target - original) * alpha,
            canvas.dtype,
        )

    ghost_path = output_dir / "overlay_ghost.png"
    write_image(ghost_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return ghost_path
