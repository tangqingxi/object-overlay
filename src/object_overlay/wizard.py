"""交互式设置引导：逐题询问，方向键选择，Enter 确认。

命令行参数先填进 RunConfig 当初始值，本模块在它上面修改，所以两种用法是
叠加而不是互斥的。

提问顺序按"先定产物、再定处理"排：输出内容 → 输出目录 → 轨迹线样式 →
轨迹线颜色 → 抽帧间隔 → 是否逐帧复核。
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .menu import choose, prompt_text
from .pipeline import RunConfig, default_output_dir
from .trajectory import format_hex_color, parse_hex_color


@dataclass(frozen=True)
class Question:
    title: str
    options: Callable[[RunConfig], tuple[tuple[str, str], ...]]
    current: Callable[[RunConfig], int]
    apply: Callable[[RunConfig, int], RunConfig]
    applies: Callable[[RunConfig], bool] = lambda _config: True


_MODE_OPTIONS = (
    ("overlay", "只出无损叠加图"),
    ("ghost", "只出残影版"),
    ("both", "两个都出"),
)
_INTERVAL_PRESETS = (0.5, 1.0, 2.0)
_COLOR_PRESETS = (
    ("#FFA500", "琥珀"),
    ("#00FF00", "绿"),
    ("#FF3B30", "红"),
)
_CUSTOM = "自定义"

_KEYS_HINT = (
    "↑/↓ 选择   Enter 或 → 确认   ← 或 Esc 返回上一步（在第一题时取消）"
)

_GHOST_MODES = ("ghost", "both")


def _mode_index(config: RunConfig) -> int:
    return [value for value, _ in _MODE_OPTIONS].index(config.mode)


def _apply_mode(config: RunConfig, index: int) -> RunConfig:
    return replace(config, mode=_MODE_OPTIONS[index][0])


def _resolved_output_dir(config: RunConfig) -> Path:
    if config.output_dir is not None:
        return config.output_dir.expanduser().resolve()

    return default_output_dir(config.video_path.expanduser().resolve())


def _output_options(config: RunConfig) -> tuple[tuple[str, str], ...]:
    return (
        ("默认", str(_resolved_output_dir(config))),
        (_CUSTOM, "手动输入路径"),
    )


def _output_index(config: RunConfig) -> int:
    # 命令行给了 --output-dir 就落在"自定义"上，否则是"默认"。
    return 1 if config.output_dir is not None else 0


def _apply_output(config: RunConfig, index: int) -> RunConfig:
    if index == 0:
        return replace(config, output_dir=None)

    answer = prompt_text("输出目录", str(_resolved_output_dir(config)))

    if answer is None:
        return config

    return replace(config, output_dir=Path(answer))


_TRAJECTORY_OPTIONS = (
    ("绘制，带描边", "线下方铺一条深色描边，浅色背景上也读得出来"),
    ("绘制，不带描边", "只留彩色线本身"),
    ("不绘制", "只做时间渐隐，不连线"),
)


def _trajectory_index(config: RunConfig) -> int:
    if not config.show_trajectory:
        return 2

    return 0 if config.trajectory_outline else 1


def _apply_trajectory(config: RunConfig, index: int) -> RunConfig:
    return replace(
        config,
        show_trajectory=index < 2,
        trajectory_outline=index == 0,
    )


def _color_options(_config: RunConfig) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, hex_value) for hex_value, name in _COLOR_PRESETS
    ) + ((_CUSTOM, "输入十六进制色值"),)


def _color_index(config: RunConfig) -> int:
    current = format_hex_color(config.trajectory_color)

    for index, (hex_value, _name) in enumerate(_COLOR_PRESETS):
        if hex_value.upper() == current:
            return index

    return len(_COLOR_PRESETS)


def _apply_color(config: RunConfig, index: int) -> RunConfig:
    if index < len(_COLOR_PRESETS):
        return replace(
            config,
            trajectory_color=parse_hex_color(_COLOR_PRESETS[index][0]),
        )

    answer = prompt_text(
        "轨迹线颜色（十六进制）",
        format_hex_color(config.trajectory_color),
    )

    if answer is None:
        return config

    try:
        color = parse_hex_color(answer)
    except ValueError as exc:
        print(f"  {exc}，保持原颜色")
        return config

    return replace(config, trajectory_color=color)


def _interval_options(_config: RunConfig) -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"{value:g} 秒", "每个候选位置之间的时间间隔")
        for value in _INTERVAL_PRESETS
    ) + ((_CUSTOM, "手动输入秒数"),)


def _interval_index(config: RunConfig) -> int:
    if config.interval in _INTERVAL_PRESETS:
        return _INTERVAL_PRESETS.index(config.interval)

    return len(_INTERVAL_PRESETS)


def _apply_interval(config: RunConfig, index: int) -> RunConfig:
    if index < len(_INTERVAL_PRESETS):
        return replace(config, interval=_INTERVAL_PRESETS[index])

    answer = prompt_text("抽帧间隔（秒）", f"{config.interval:g}")

    if answer is None:
        return config

    try:
        value = float(answer)
    except ValueError:
        print(f"  无法识别为数字：{answer}，保持 {config.interval:g} 秒")
        return config

    if value <= 0:
        print(f"  间隔必须大于 0，保持 {config.interval:g} 秒")
        return config

    return replace(config, interval=value)


_SELECT_OPTIONS = (
    ("否", "直接用自动识别的全部目标，认错了也照收"),
    ("是", "逐帧看：剔除认错的对象，Mask 不准的还能手动修"),
)


def _select_index(config: RunConfig) -> int:
    return 1 if config.select else 0


def _apply_select(config: RunConfig, index: int) -> RunConfig:
    return replace(config, select=index == 1)


def _is_ghost(config: RunConfig) -> bool:
    return config.mode in _GHOST_MODES


def _build_questions() -> tuple[Question, ...]:
    return (
        Question(
            title="要生成哪种叠加图",
            options=lambda _config: _MODE_OPTIONS,
            current=_mode_index,
            apply=_apply_mode,
        ),
        Question(
            title="输出目录",
            options=_output_options,
            current=_output_index,
            apply=_apply_output,
        ),
        Question(
            title="轨迹线",
            options=lambda _config: _TRAJECTORY_OPTIONS,
            current=_trajectory_index,
            apply=_apply_trajectory,
            applies=_is_ghost,
        ),
        Question(
            title="轨迹线颜色",
            options=_color_options,
            current=_color_index,
            apply=_apply_color,
            applies=lambda config: _is_ghost(config) and config.show_trajectory,
        ),
        Question(
            title="抽帧间隔",
            options=_interval_options,
            current=_interval_index,
            apply=_apply_interval,
        ),
        Question(
            # 自动识别有时会把不是目标的东西也抠进来，或者 Mask 切不准。
            # 这一步的意义就是复核这些错误，而不是单纯地挑选时刻。
            title="是否逐帧复核识别结果",
            options=lambda _config: _SELECT_OPTIONS,
            current=_select_index,
            apply=_apply_select,
        ),
    )


def _summarize(config: RunConfig) -> None:
    if config.show_trajectory:
        trajectory = (
            "绘制，带描边" if config.trajectory_outline else "绘制，不带描边"
        )
    else:
        trajectory = "不绘制"

    print()
    print("将按以下设置生成：")
    print(f"  输出内容    {config.mode}")
    print(f"  输出目录    {_resolved_output_dir(config)}")

    if _is_ghost(config):
        print(f"  轨迹线      {trajectory}")

        if config.show_trajectory:
            print(f"  轨迹线颜色  {format_hex_color(config.trajectory_color)}")

    print(f"  抽帧间隔    {config.interval:g} 秒")
    print(f"  逐帧复核    {'是' if config.select else '否'}")


def configure(config: RunConfig) -> RunConfig | None:
    """逐题询问并返回改好的配置；用户取消时返回 None。"""
    questions = _build_questions()
    current = config
    index = 0

    print()
    print(_KEYS_HINT)

    while True:
        # 每次作答后重新筛题：第一题选 overlay 之后，轨迹线那两题会消失；
        # 回退时也自动跟着当前状态走。
        active = [
            question for question in questions if question.applies(current)
        ]

        if index >= len(active):
            _summarize(current)
            confirmed = choose(
                "确认开始？",
                (("开始处理", ""), ("返回修改", "")),
            )

            if confirmed is None:
                return None

            if confirmed == 0:
                return current

            index = len(active) - 1
            continue

        question = active[index]
        answer = choose(
            question.title,
            question.options(current),
            question.current(current),
        )

        if answer is None:
            # 第一题再往回就是取消整轮。
            if index == 0:
                return None

            index -= 1
            continue

        current = question.apply(current, answer)
        index += 1
