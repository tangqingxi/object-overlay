import tempfile
import unittest
from pathlib import Path

from object_overlay.pipeline import (
    OUTPUT_MODES,
    RunConfig,
    _resolve_config,
    default_output_dir,
)


class DefaultOutputDirTest(unittest.TestCase):
    def test_视频在input目录下时输出到兄弟output目录(self) -> None:
        self.assertEqual(
            default_output_dir(Path("/data/drone/input/clip.mp4")),
            Path("/data/drone/output/clip_overlay"),
        )

    def test_其他位置放在视频旁边(self) -> None:
        self.assertEqual(
            default_output_dir(Path("/data/movies/clip.mp4")),
            Path("/data/movies/clip_overlay"),
        )

    def test_目录名大小写不敏感(self) -> None:
        self.assertEqual(
            default_output_dir(Path("/data/drone/Input/clip.mp4")),
            Path("/data/drone/output/clip_overlay"),
        )



class ResolveConfigTest(unittest.TestCase):
    def _video(self, root: Path) -> Path:
        video = root / "clip.mp4"
        video.write_bytes(b"placeholder")
        return video

    def test_默认生成两种叠加图(self) -> None:
        self.assertEqual(RunConfig(video_path=Path("clip.mp4")).mode, "both")

    def test_接受全部合法模式(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            video = self._video(Path(temporary_dir))

            for mode in OUTPUT_MODES:
                resolved = _resolve_config(
                    RunConfig(video_path=video, mode=mode)
                )
                self.assertEqual(resolved.mode, mode)

    def test_拒绝不支持的模式(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            video = self._video(Path(temporary_dir))

            # motion 是旧版的取值，现在必须明确报错而不是被静默接受。
            for mode in ("motion", "auto", "manual", ""):
                with self.assertRaises(ValueError):
                    _resolve_config(RunConfig(video_path=video, mode=mode))

    def test_视频在input目录时默认输出到兄弟output目录(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir) / "input"
            input_dir.mkdir()
            video = input_dir / "clip.mp4"
            video.write_bytes(b"placeholder")

            resolved = _resolve_config(RunConfig(video_path=video))

            self.assertEqual(
                resolved.output_dir,
                resolved.video_path.parent.parent / "output" / "clip_overlay",
            )
            self.assertEqual(resolved.output_dir.parent.name, "output")

    def test_输出目录默认取自视频名(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            video = self._video(Path(temporary_dir))
            resolved = _resolve_config(RunConfig(video_path=video))

            self.assertEqual(resolved.output_dir, video.with_name("clip_overlay"))


if __name__ == "__main__":
    unittest.main()
