from dataclasses import dataclass
from pathlib import Path

from .compose import CompositionResult, compose_overlay
from .masking import (
    generate_motion_masks,
    select_motion_masks,
)
from .video import load_preview_frames, prepare_frames


@dataclass(frozen=True)
class RunConfig:
    video_path: Path
    mode: str
    interval: float = 0.5
    base_frame: int = 1
    output_dir: Path | None = None
    force: bool = False
    select: bool = False


def _resolve_config(config: RunConfig) -> RunConfig:
    video_path = config.video_path.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    if config.interval <= 0:
        raise ValueError("--interval 必须大于 0")

    if config.base_frame <= 0:
        raise ValueError("--base-frame 必须大于 0")

    if config.mode != "motion":
        raise ValueError(f"不支持的模式：{config.mode}")

    if config.output_dir is None:
        output_dir = video_path.with_name(f"{video_path.stem}_overlay")
    else:
        output_dir = config.output_dir.expanduser().resolve()

    return RunConfig(
        video_path=video_path,
        mode=config.mode,
        interval=config.interval,
        base_frame=config.base_frame,
        output_dir=output_dir,
        force=config.force,
        select=config.select,
    )


def run_pipeline(config: RunConfig) -> CompositionResult:
    """执行抽帧、目标 Mask 生成和无损叠加。"""
    config = _resolve_config(config)
    output_dir = config.output_dir

    if output_dir is None:
        raise RuntimeError("未能确定输出目录。")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    metadata_file = output_dir / "metadata.json"
    motion_mask_dir = output_dir / "masks_motion"

    print()
    print("=" * 60)
    print(f"视频：{config.video_path}")
    print(f"输出目录：{output_dir}")
    print(f"模式：{config.mode}")
    print(f"抽帧间隔：{config.interval:g} 秒")
    print(f"背景帧：第 {config.base_frame} 张抽帧")
    print(f"人工筛选：{'是' if config.select else '否'}")
    print("=" * 60)

    frame_files = prepare_frames(
        config.video_path,
        frames_dir,
        metadata_file,
        config.interval,
    )
    images, width, height = load_preview_frames(frame_files)
    print(f"共找到 {len(images)} 帧")
    print(f"分辨率：{width} × {height}")

    base_index = config.base_frame - 1

    if base_index >= len(images):
        raise ValueError(
            f"--base-frame 为 {config.base_frame}，"
            f"但当前只有 {len(images)} 张抽帧。"
        )

    masks = generate_motion_masks(
        images,
        motion_mask_dir,
        output_dir,
        config.force,
        base_index,
    )

    if config.select:
        masks = select_motion_masks(
            images,
            masks,
            output_dir / "masks_corrected",
            base_index,
            config.interval,
        )
        output_mode = "selected"
    else:
        output_mode = "motion"

    result = compose_overlay(
        frame_files,
        masks,
        base_index,
        output_dir,
        output_mode,
    )

    print()
    print("=" * 60)
    print("全部完成")
    print(f"模式：{output_mode}")
    print(f"PNG：{result.png_path}")
    print(f"JPG（仅预览）：{result.jpg_path}")
    print(f"Mask：{result.mask_path}")
    print(f"最终 PNG dtype：{result.dtype}")
    print(f"最终 PNG 分辨率：{result.width} × {result.height}")
    print(
        "Mask 外背景最大像素差："
        f"{result.background_max_difference}"
    )
    print("=" * 60)

    if output_mode == "motion":
        print()
        print("请同时检查最终 PNG 和 masks_motion.png。")
        print("运动目标模式适用于固定或基本固定的镜头。")
    elif output_mode == "selected":
        print()
        print("最终结果只包含人工选择纳入的自动识别目标。")

    return result
