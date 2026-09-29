import argparse
import sys
from pathlib import Path

from .compose import GHOST_MIN_ALPHA
from .pipeline import OUTPUT_MODES, RunConfig, run_pipeline
from .trajectory import (
    TRAJECTORY_COLOR,
    format_hex_color,
    parse_hex_color,
)
from .wizard import configure


def _trajectory_color(text: str) -> tuple[int, int, int]:
    """把解析错误转成 argparse 自己的类型，让报错显示具体原因。"""
    try:
        return parse_hex_color(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="object-overlay",
        description=(
            "从视频中提取目标对象，并生成目标在多个时刻的叠加图。"
        ),
    )
    parser.add_argument(
        "video",
        type=Path,
        help="视频文件路径，例如 .\\drone.mp4",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="跳过交互式设置，直接按当前参数开始（脚本用）",
    )

    framing = parser.add_argument_group("抽帧")
    framing.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="抽帧时间间隔（秒），默认 0.5",
    )
    framing.add_argument(
        "--base-frame",
        type=int,
        default=1,
        help="作为固定背景的抽帧编号，默认 1",
    )

    targets = parser.add_argument_group("目标 Mask")
    targets.add_argument(
        "--force",
        action="store_true",
        help="忽略已有缓存，重新生成全部 Mask",
    )
    targets.add_argument(
        "--select",
        action="store_true",
        help=(
            "逐帧复核自动识别的结果：剔除认错的对象，"
            "Mask 不准的还能手动修正"
        ),
    )

    output = parser.add_argument_group("输出")
    output.add_argument(
        "--mode",
        choices=OUTPUT_MODES,
        default="both",
        help=(
            "要生成哪种叠加图：overlay=只出无损版，"
            "ghost=只出残影版，both=两个都出（默认）"
        ),
    )
    output.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果目录；默认在视频旁创建 <视频名>_overlay",
    )

    ghost = parser.add_argument_group("残影版")
    ghost.add_argument(
        "--ghost-min-alpha",
        type=float,
        default=GHOST_MIN_ALPHA,
        help=(
            "最早时刻的不透明度，范围 (0, 1]，"
            f"默认 {GHOST_MIN_ALPHA:g}"
        ),
    )
    ghost.add_argument(
        "--trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否绘制连接各时刻的轨迹线，默认绘制",
    )
    ghost.add_argument(
        "--trajectory-outline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "是否在轨迹线下方加一条深色描边，默认加。"
            "描边让浅色线条在浅色背景上也可读，关掉则只剩彩色线本身"
        ),
    )
    ghost.add_argument(
        "--trajectory-color",
        type=_trajectory_color,
        default=TRAJECTORY_COLOR,
        metavar="#RRGGBB",
        help=(
            "轨迹线颜色，十六进制，也接受 #RGB 简写；"
            f"默认 {format_hex_color(TRAJECTORY_COLOR)}"
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = RunConfig(
        video_path=args.video,
        mode=args.mode,
        interval=args.interval,
        base_frame=args.base_frame,
        output_dir=args.output_dir,
        force=args.force,
        select=args.select,
        ghost_min_alpha=args.ghost_min_alpha,
        show_trajectory=args.trajectory,
        trajectory_color=args.trajectory_color,
        trajectory_outline=args.trajectory_outline,
    )

    try:
        # 先确认视频存在，免得用户在菜单里选完一串才发现路径写错。
        video_path = args.video.expanduser()

        if not video_path.is_file():
            print(f"错误：视频文件不存在：{video_path}", file=sys.stderr)
            return 1

        # 非终端环境无法弹菜单，只能直接按参数跑。
        if not args.yes and sys.stdin.isatty():
            configured = configure(config)

            if configured is None:
                print("\n已取消。", file=sys.stderr)
                return 130

            config = configured

        run_pipeline(config)
    except KeyboardInterrupt:
        print("\n处理已由用户中止。", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    return 0
