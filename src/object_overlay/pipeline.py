from dataclasses import dataclass
from pathlib import Path

from .compose import (
    GHOST_MIN_ALPHA,
    CompositionResult,
    compose_ghost_overlay,
    compose_overlay,
)
from .masking import (
    detect_background_object,
    generate_motion_masks,
    select_motion_masks,
)
from .trajectory import TRAJECTORY_COLOR, format_hex_color
from .video import (
    load_preview_frames,
    prepare_frames,
    update_metadata,
)


# --mode 的取值：只出无损版、只出残影版、两个都出。
OUTPUT_MODES = ("overlay", "ghost", "both")

# 视频放在这个目录名下时，产物默认落到它的兄弟目录 output 里。
INPUT_DIR_NAME = "input"

OUTPUT_DIR_NAME = "output"


def default_output_dir(video_path: Path) -> Path:
    """未指定 --output-dir 时的默认输出目录。

    视频放在名为 input 的目录里时（本仓库的样例就是这种布局），输出到它
    的兄弟目录 output 下，免得产物和源视频混在一起；其他情况放在视频旁边。
    """
    parent = video_path.parent

    if parent.name.lower() == INPUT_DIR_NAME:
        parent = parent.parent / OUTPUT_DIR_NAME

    return parent / f"{video_path.stem}_overlay"


@dataclass(frozen=True)
class RunConfig:
    video_path: Path
    mode: str = "both"
    interval: float = 0.5
    base_frame: int = 1
    output_dir: Path | None = None
    force: bool = False
    select: bool = False
    ghost_min_alpha: float = GHOST_MIN_ALPHA
    show_trajectory: bool = True
    trajectory_color: tuple[int, int, int] = TRAJECTORY_COLOR
    trajectory_outline: bool = True


@dataclass(frozen=True)
class RunResult:
    """一次运行的产物，未生成的为 None。"""

    overlay: CompositionResult | None = None
    ghost_png: Path | None = None


def _resolve_config(config: RunConfig) -> RunConfig:
    video_path = config.video_path.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    if config.mode not in OUTPUT_MODES:
        raise ValueError(
            f"不支持的模式：{config.mode}，"
            f"可选 {'、'.join(OUTPUT_MODES)}"
        )

    if config.interval <= 0:
        raise ValueError("--interval 必须大于 0")

    if config.base_frame <= 0:
        raise ValueError("--base-frame 必须大于 0")

    if not 0.0 < config.ghost_min_alpha <= 1.0:
        raise ValueError("--ghost-min-alpha 必须落在 (0, 1] 区间内")

    if len(config.trajectory_color) != 3 or not all(
        0 <= channel <= 255 for channel in config.trajectory_color
    ):
        raise ValueError("--trajectory-color 需要三个 0..255 的通道值")

    if config.output_dir is None:
        output_dir = default_output_dir(video_path)
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
        ghost_min_alpha=config.ghost_min_alpha,
        show_trajectory=config.show_trajectory,
        trajectory_color=config.trajectory_color,
        trajectory_outline=config.trajectory_outline,
    )


def _print_header(config: RunConfig, output_dir: Path, want_ghost: bool) -> None:
    print()
    print("=" * 60)
    print(f"视频：{config.video_path}")
    print(f"输出目录：{output_dir}")
    print(f"输出内容：{config.mode}")
    print(f"抽帧间隔：{config.interval:g} 秒")
    print(f"背景帧：第 {config.base_frame} 张抽帧")
    print(f"人工筛选：{'是' if config.select else '否'}")

    if want_ghost:
        print(f"残影最早时刻不透明度：{config.ghost_min_alpha:g}")
        print(f"轨迹线：{'开' if config.show_trajectory else '关'}")

        if config.show_trajectory:
            print(f"轨迹线颜色：{format_hex_color(config.trajectory_color)}")
            print(
                f"轨迹线描边：{'开' if config.trajectory_outline else '关'}"
            )

    print("=" * 60)


def _print_summary(
    config: RunConfig,
    overlay: CompositionResult | None,
    ghost_png: Path | None,
) -> None:
    print()
    print("=" * 60)
    print("全部完成")

    if overlay is not None:
        print(f"无损叠加图：{overlay.png_path}")
        print(f"预览 JPG：{overlay.jpg_path}")
        print(f"Mask：{overlay.mask_path}")
        print(f"最终 PNG dtype：{overlay.dtype}")
        print(f"最终 PNG 分辨率：{overlay.width} × {overlay.height}")
        print(f"Mask 外背景最大像素差：{overlay.background_max_difference}")

    if ghost_png is not None:
        print(f"残影叠加图：{ghost_png}")

    print("=" * 60)

    print()
    if config.select:
        print("结果已按逐帧复核的结果筛选过。")
    else:
        print(
            "结果包含自动识别的全部目标，识别错误也会照收；"
            "需要复核时加 --select。"
        )

    print("自动提取运动目标适用于固定或基本固定的镜头。")


def run_pipeline(config: RunConfig) -> RunResult:
    """执行抽帧、目标 Mask 生成，并按 --mode 生成叠加图。"""
    config = _resolve_config(config)
    output_dir = config.output_dir

    if output_dir is None:
        raise RuntimeError("未能确定输出目录。")

    want_overlay = config.mode in ("overlay", "both")
    want_ghost = config.mode in ("ghost", "both")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    metadata_file = output_dir / "metadata.json"
    motion_mask_dir = output_dir / "masks_motion"

    _print_header(config, output_dir, want_ghost)

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

    # 产物文件名不再随 --select 变化，人工筛选的痕迹记进抽帧记录。
    update_metadata(
        metadata_file,
        mode=config.mode,
        curated=config.select,
    )

    overlay = None
    ghost_png = None

    if want_overlay:
        overlay = compose_overlay(
            frame_files,
            masks,
            base_index,
            output_dir,
        )

    if want_ghost:
        # 只有残影版需要知道背景帧里的目标在哪。
        background_mask = detect_background_object(images, base_index)

        if not background_mask.any():
            print("背景帧里没有检测到目标。")

        ghost_png = compose_ghost_overlay(
            frame_files,
            masks,
            base_index,
            output_dir,
            min_alpha=config.ghost_min_alpha,
            background_mask=background_mask,
            show_trajectory=config.show_trajectory,
            trajectory_color=config.trajectory_color,
            trajectory_outline=config.trajectory_outline,
        )

    _print_summary(config, overlay, ghost_png)

    return RunResult(overlay=overlay, ghost_png=ghost_png)
