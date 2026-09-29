"""终端菜单：方向键移动、Enter 确认。

Windows 上 curses 和 termios 都不可用，只能走 msvcrt；POSIX 上走 termios。
两条路径都封装在本模块里，上层只需要 read_key / choose / prompt_text。

光标标记用 ">" 而不是 "❯"——GBK 里没有后者，Windows 控制台显示不出来。
"""

import os
import sys
from collections.abc import Callable, Sequence


WINDOWS = sys.platform == "win32"

_ESCAPE = "\x1b"
# 单独的 Esc 与 "ESC [ A" 这类序列前缀相同，只能靠超时区分。
_ESCAPE_TIMEOUT = 0.05
_ENTER_KEYS = ("\r", "\n")
_INTERRUPT = "\x03"

# Windows 上方向键是前缀字符加一个字母。
_WINDOWS_ARROWS = {"H": "up", "P": "down", "K": "left", "M": "right"}
# POSIX 上是 ESC [ 加一个字母。
_POSIX_ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

_ansi_ready = False


def _enable_ansi() -> None:
    """Windows 控制台默认不解析 ANSI 转义，需要显式打开。"""
    global _ansi_ready

    if WINDOWS and not _ansi_ready:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()

        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)

    _ansi_ready = True


def _normalize(character: str) -> str:
    if character in _ENTER_KEYS:
        return "enter"
    if character == _ESCAPE:
        return "esc"
    if character == _INTERRUPT:
        # raw 模式关掉了 ISIG，Ctrl+C 不会自己抛异常，得手动转。
        raise KeyboardInterrupt

    return character.lower()


def _read_key_windows() -> str:
    import msvcrt

    character = msvcrt.getwch()

    if character in ("\x00", "\xe0"):
        return _WINDOWS_ARROWS.get(msvcrt.getwch(), "")

    return _normalize(character)


def _decode_escape(
    read_byte: Callable[[], str],
    has_pending: Callable[[], bool],
) -> str:
    """解码 ESC 开头的序列。

    单独的 Esc 与 "ESC [ A" 这类序列前缀相同，只能靠"后面还有没有字符"
    区分。抽成纯函数是为了能脱离终端单独测试。
    """
    if not has_pending():
        return "esc"

    if read_byte() != "[":
        return "esc"

    return _POSIX_ARROWS.get(read_byte(), "")


def _read_key_posix() -> str:
    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)

    def read_byte() -> str:
        # 用 os.read 而不是 sys.stdin.read：后者在 raw 模式下仍然会缓冲。
        return os.read(descriptor, 1).decode("utf-8", "ignore")

    def has_pending() -> bool:
        return bool(select.select([descriptor], [], [], _ESCAPE_TIMEOUT)[0])

    try:
        tty.setraw(descriptor)
        character = read_byte()

        if character == _ESCAPE:
            return _decode_escape(read_byte, has_pending)

        return _normalize(character)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def read_key() -> str:
    """读一个按键，方向键归一成 up / down / left / right。

    无法识别的特殊键返回空串，调用方应当忽略。
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "交互式菜单需要终端。非交互环境请直接使用命令行参数。"
        )

    if WINDOWS:
        return _read_key_windows()

    return _read_key_posix()


def _render(
    title: str,
    options: Sequence[tuple[str, str]],
    index: int,
    label_width: int,
) -> list[str]:
    lines = [f"? {title}"]

    for position, (label, note) in enumerate(options):
        marker = ">" if position == index else " "
        line = f"{marker} {label.ljust(label_width)}"

        if note:
            line = f"{line}   {note}"

        lines.append(line)

    return lines


def choose(
    title: str,
    options: Sequence[tuple[str, str]],
    default_index: int = 0,
) -> int | None:
    """显示单选菜单，返回选中项下标；按 Esc 或 ← 返回 None。

    options 是 (标签, 说明) 序列，说明可以为空串。→ 与 Enter 等价。
    """
    if not options:
        return None

    _enable_ansi()

    index = max(0, min(default_index, len(options) - 1))
    label_width = max(len(label) for label, _ in options)
    lines = _render(title, options, index, label_width)

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()

    while True:
        key = read_key()

        if key in ("up", "down"):
            step = -1 if key == "up" else 1
            index = max(0, min(index + step, len(options) - 1))
            lines = _render(title, options, index, label_width)
            # 光标此刻在菜单下方，上移这么多行就能原地重绘。
            sys.stdout.write(f"\x1b[{len(lines)}A")
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
        elif key in ("enter", "right"):
            return index
        elif key in ("esc", "left"):
            return None


def prompt_text(title: str, default: str) -> str | None:
    """让用户输入一行文本；直接回车取默认值，Ctrl+C 返回 None。"""
    print(f"? {title} [{default}]")

    try:
        answer = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    return answer or default
