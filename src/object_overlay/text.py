"""在 OpenCV 窗口中绘制中文文字。

OpenCV 的 putText 不支持中文，这里用 Pillow 画字后再转回 OpenCV 图像。
字体优先使用 Windows 自带的中文字体；找不到时回退到 Pillow 默认字体，
至少不会因为缺字体而中断整个流程。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font_candidates() -> tuple[Path, ...]:
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    fonts_dir = windows_dir / "Fonts"
    return (
        fonts_dir / "msyh.ttc",
        fonts_dir / "msyhbd.ttc",
        fonts_dir / "simhei.ttf",
        fonts_dir / "simsun.ttc",
    )


@lru_cache(maxsize=16)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _font_candidates():
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue

    return ImageFont.load_default()


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    size: int = 24,
    color: tuple[int, int, int] = (0, 255, 255),
    stroke: int = 2,
) -> None:
    """在 BGR 图像上绘制一行中文，origin 是左上角坐标。"""
    font = _load_font(size)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    rgb_color = (int(color[2]), int(color[1]), int(color[0]))
    draw.text(
        origin,
        text,
        font=font,
        fill=rgb_color,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0),
    )
    image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
