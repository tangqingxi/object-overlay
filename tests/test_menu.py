import io
import unittest
from pathlib import Path
from unittest.mock import patch

from object_overlay import menu, wizard
from object_overlay.pipeline import RunConfig


class KeyDecodeTest(unittest.TestCase):
    def test_普通按键归一化(self) -> None:
        self.assertEqual(menu._normalize("\r"), "enter")
        self.assertEqual(menu._normalize("\n"), "enter")
        self.assertEqual(menu._normalize("\x1b"), "esc")
        self.assertEqual(menu._normalize("Q"), "q")

    def test_CtrlC转成键盘中断(self) -> None:
        # raw 模式关掉了 ISIG，Ctrl+C 不会自己抛异常，必须手动转。
        with self.assertRaises(KeyboardInterrupt):
            menu._normalize("\x03")

    def test_非终端环境抛错(self) -> None:
        with patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(RuntimeError):
                menu.read_key()


class EscapeSequenceTest(unittest.TestCase):
    def _decode(self, sequence: str) -> str:
        pending = list(sequence)

        def read_byte() -> str:
            return pending.pop(0) if pending else ""

        return menu._decode_escape(read_byte, lambda: bool(pending))

    def test_单独Esc(self) -> None:
        self.assertEqual(self._decode(""), "esc")

    def test_方向键序列(self) -> None:
        self.assertEqual(self._decode("[A"), "up")
        self.assertEqual(self._decode("[B"), "down")
        self.assertEqual(self._decode("[C"), "right")
        self.assertEqual(self._decode("[D"), "left")

    def test_未知序列返回空串(self) -> None:
        # 空串表示无法识别，调用方应当忽略。
        self.assertEqual(self._decode("[Z"), "")

    def test_Esc后面不是方括号(self) -> None:
        self.assertEqual(self._decode("x"), "esc")


@unittest.skipUnless(menu.WINDOWS, "msvcrt 只在 Windows 上可用")
class WindowsKeyTest(unittest.TestCase):
    def _read(self, sequence: str) -> str:
        keys = iter(sequence)

        with patch("msvcrt.getwch", side_effect=lambda: next(keys)):
            return menu._read_key_windows()

    def test_方向键前缀解码(self) -> None:
        # Windows 上方向键是 \xe0 前缀加一个字母。
        self.assertEqual(self._read("\xe0H"), "up")
        self.assertEqual(self._read("\xe0P"), "down")
        self.assertEqual(self._read("\xe0K"), "left")
        self.assertEqual(self._read("\xe0M"), "right")

    def test_普通键不受影响(self) -> None:
        self.assertEqual(self._read("\r"), "enter")
        self.assertEqual(self._read("s"), "s")

    def test_未知特殊键返回空串(self) -> None:
        self.assertEqual(self._read("\xe0X"), "")


class ChooseTest(unittest.TestCase):
    OPTIONS = (("a", "第一项"), ("b", "第二项"), ("c", "第三项"))

    def _choose(
        self,
        keys: list[str],
        default_index: int = 0,
    ) -> int | None:
        with (
            patch("object_overlay.menu.read_key", side_effect=keys),
            patch("object_overlay.menu._enable_ansi"),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            return menu.choose("标题", self.OPTIONS, default_index)

    def test_回车取默认项(self) -> None:
        self.assertEqual(self._choose(["enter"]), 0)

    def test_下移后回车(self) -> None:
        self.assertEqual(self._choose(["down", "down", "enter"]), 2)

    def test_到顶继续上移应停住(self) -> None:
        self.assertEqual(self._choose(["up", "up", "enter"]), 0)

    def test_到底继续下移应停住(self) -> None:
        self.assertEqual(
            self._choose(["down"] * 5 + ["enter"]),
            len(self.OPTIONS) - 1,
        )

    def test_默认项越界时收敛到合法范围(self) -> None:
        self.assertEqual(self._choose(["enter"], 99), len(self.OPTIONS) - 1)

    def test_Esc与左键都表示返回(self) -> None:
        self.assertIsNone(self._choose(["esc"]))
        self.assertIsNone(self._choose(["left"]))

    def test_右键等价于确认(self) -> None:
        self.assertEqual(self._choose(["down", "right"]), 1)

    def test_无法识别的按键被忽略(self) -> None:
        self.assertEqual(self._choose(["", "", "enter"]), 0)

    def test_没有选项时返回None(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertIsNone(menu.choose("标题", ()))


class WizardTest(unittest.TestCase):
    def _config(self, **overrides: object) -> RunConfig:
        return RunConfig(video_path=Path("clip.mp4"), **overrides)

    def _run_with_answers(
        self,
        answers: list[int | None],
        config: RunConfig,
    ) -> tuple[RunConfig | None, list[str]]:
        titles: list[str] = []
        remaining = list(answers)

        def fake_choose(title, options, default_index=0):
            titles.append(title)
            return remaining.pop(0) if remaining else 0

        with (
            patch("object_overlay.wizard.choose", side_effect=fake_choose),
            patch("object_overlay.wizard.prompt_text", return_value=None),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            return wizard.configure(config), titles

    def test_全部确认默认项时配置不变(self) -> None:
        config = self._config()
        # 各题的默认选中项依次是：both、默认输出目录、绘制带描边、琥珀、
        # 0.5 秒、不复核，最后确认开始。
        result, _ = self._run_with_answers([2, 0, 0, 0, 0, 0, 0], config)

        self.assertEqual(result, config)

    def test_题目顺序(self) -> None:
        # 先定产物（输出内容、输出目录、轨迹线样式），再定处理（抽帧、复核）。
        _, titles = self._run_with_answers(
            [1, 0, 0, 0, 0, 0, 0],
            self._config(mode="both"),
        )

        self.assertEqual(
            titles,
            [
                "要生成哪种叠加图",
                "输出目录",
                "轨迹线",
                "轨迹线颜色",
                "抽帧间隔",
                "是否逐帧复核识别结果",
                "确认开始？",
            ],
        )

    def test_选overlay后不再询问轨迹线(self) -> None:
        _, titles = self._run_with_answers(
            [0, 0, 0, 0, 0],
            self._config(mode="both"),
        )

        self.assertEqual(
            titles,
            [
                "要生成哪种叠加图",
                "输出目录",
                "抽帧间隔",
                "是否逐帧复核识别结果",
                "确认开始？",
            ],
        )

    def test_不绘制轨迹线时不再询问颜色(self) -> None:
        config = self._config(mode="ghost", show_trajectory=True)
        # 轨迹线那题选第三个选项＝不绘制
        _, titles = self._run_with_answers([1, 0, 2, 0, 0, 0], config)

        self.assertIn("轨迹线", titles)
        self.assertNotIn("轨迹线颜色", titles)

    def test_可以把描边关掉(self) -> None:
        result, _ = self._run_with_answers(
            [1, 0, 1, 0, 0, 0, 0],
            self._config(mode="ghost"),
        )

        assert result is not None
        self.assertTrue(result.show_trajectory)
        self.assertFalse(result.trajectory_outline)

    def test_可以自定义输出目录(self) -> None:
        config = self._config(mode="overlay")

        def fake_choose(title, options, default_index=0):
            return 1 if title == "输出目录" else 0

        with (
            patch("object_overlay.wizard.choose", side_effect=fake_choose),
            patch(
                "object_overlay.wizard.prompt_text",
                return_value="D:/renders",
            ),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            result = wizard.configure(config)

        assert result is not None
        self.assertEqual(result.output_dir, Path("D:/renders"))

    def test_选默认输出目录会清掉命令行给的值(self) -> None:
        config = self._config(mode="overlay", output_dir=Path("/tmp/given"))
        result, _ = self._run_with_answers([0, 0, 0, 0, 0], config)

        assert result is not None
        self.assertIsNone(result.output_dir)

    def test_第一题取消时返回None(self) -> None:
        result, _ = self._run_with_answers([None], self._config())
        self.assertIsNone(result)

    def test_确认页选返回修改会退回最后一题(self) -> None:
        # 四题全确认后进入确认页，选“返回修改”，再走一遍。
        _, titles = self._run_with_answers(
            [0, 0, 0, 0, 1, 0, 0],
            self._config(mode="overlay"),
        )

        self.assertEqual(titles.count("确认开始？"), 2)
        self.assertEqual(titles[-2], "是否逐帧复核识别结果")

    def test_取消确认页返回None(self) -> None:
        result, _ = self._run_with_answers(
            [0, 0, 0, 0, None],
            self._config(mode="overlay"),
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
