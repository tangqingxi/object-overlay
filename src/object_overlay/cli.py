import argparse
import sys
from pathlib import Path

from .pipeline import RunConfig, run_pipeline


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
        "--mode",
        choices=["motion", "auto"],
        default="motion",
        help=(
            "motion=自动提取运动目标；auto 是 motion 的兼容别名"
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="抽帧时间间隔（秒），默认 0.5",
    )
    parser.add_argument(
        "--base-frame",
        type=int,
        default=1,
        help="作为固定背景的抽帧编号，默认 1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果目录；默认在视频旁创建 <视频名>_overlay",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新生成当前模式的全部 Mask",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="逐帧预览运动 Mask，人工选择是否纳入叠加图",
    )
    return parser


def _choose_mode(mode: str) -> str:
    if mode == "auto":
        return "motion"
    return mode


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = _choose_mode(args.mode)
    config = RunConfig(
        video_path=args.video,
        mode=mode,
        interval=args.interval,
        base_frame=args.base_frame,
        output_dir=args.output_dir,
        force=args.force,
        select=args.select,
    )

    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        print("\n处理已由用户中止。", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    return 0
