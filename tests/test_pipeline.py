import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from object_overlay.masking import SelectionResult
from object_overlay.pipeline import (
    OUTPUT_MODES,
    RunConfig,
    _resolve_config,
    default_output_dir,
    run_pipeline,
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


class BackgroundSelectionPipelineTest(unittest.TestCase):
    def _run(self, mode: str, select: bool, background_mask: np.ndarray | None):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            images = [
                np.zeros((80, 80, 3), dtype=np.uint8)
                for _ in range(3)
            ]
            motion_masks = [
                np.zeros((80, 80), dtype=np.uint8)
                for _ in range(3)
            ]
            selected_masks = [
                np.zeros((80, 80), dtype=np.uint8)
                for _ in range(3)
            ]
            selected_masks[1][10:20, 10:20] = 255
            auto_background = np.zeros((80, 80), dtype=np.uint8)
            auto_background[30:40, 30:40] = 255
            selected = SelectionResult(
                selected_masks=selected_masks,
                background_mask=background_mask,
            )
            frame_files = [
                output_dir / f"frame_{index:04d}.png"
                for index in range(1, 4)
            ]
            config = RunConfig(
                video_path=Path("clip.mp4"),
                mode=mode,
                select=select,
                base_frame=1,
                output_dir=output_dir,
            )

            with (
                patch(
                    "object_overlay.pipeline._resolve_config",
                    return_value=config,
                ),
                patch(
                    "object_overlay.pipeline.prepare_frames",
                    return_value=frame_files,
                ),
                patch(
                    "object_overlay.pipeline.load_preview_frames",
                    return_value=(images, 80, 80),
                ),
                patch(
                    "object_overlay.pipeline.generate_motion_masks",
                    return_value=motion_masks,
                ),
                patch(
                    "object_overlay.pipeline.detect_background_object",
                    return_value=auto_background,
                ) as detect,
                patch(
                    "object_overlay.pipeline.select_motion_masks",
                    return_value=selected,
                ) as select_masks,
                patch(
                    "object_overlay.pipeline.compose_overlay",
                    return_value=None,
                ) as compose_overlay,
                patch(
                    "object_overlay.pipeline.compose_ghost_overlay",
                    return_value=Path("ghost.png"),
                ) as compose_ghost,
                patch("object_overlay.pipeline.update_metadata"),
            ):
                run_pipeline(config)

            return (
                detect,
                select_masks,
                compose_overlay,
                compose_ghost,
                auto_background,
                selected_masks,
                selected,
                images,
                output_dir,
            )

    def test_both加select时人工背景Mask只进入ghost合成(self) -> None:
        manual = np.zeros((80, 80), dtype=np.uint8)
        manual[50:60, 50:60] = 255
        (
            detect,
            select_masks,
            compose_overlay,
            compose_ghost,
            auto_background,
            _selected_masks,
            selected,
            images,
            _output_dir,
        ) = self._run("both", True, manual)

        # 自动检测只发生在复核之前，作为初值传给复核界面。
        detect.assert_called_once_with(images, 0)
        self.assertIs(
            select_masks.call_args.kwargs["background_auto_mask"],
            auto_background,
        )
        compose_overlay.assert_called_once()
        self.assertIs(
            compose_overlay.call_args.args[1],
            selected.selected_masks,
        )
        self.assertIs(
            compose_ghost.call_args.kwargs["background_mask"],
            manual,
        )

    def test_overlay加select不弹背景帧复核(self) -> None:
        (
            detect,
            select_masks,
            compose_overlay,
            compose_ghost,
            _auto_background,
            _selected_masks,
            _selected,
            _images,
            _output_dir,
        ) = self._run("overlay", True, None)

        detect.assert_not_called()
        self.assertIsNone(
            select_masks.call_args.kwargs["background_auto_mask"]
        )
        compose_overlay.assert_called_once()
        compose_ghost.assert_not_called()

    def test_ghost加select跳过背景时不回退到自动检测(self) -> None:
        (
            detect,
            _select_masks,
            compose_overlay,
            compose_ghost,
            _auto_background,
            _selected_masks,
            _selected,
            _images,
            _output_dir,
        ) = self._run("ghost", True, None)

        detect.assert_called_once()
        compose_overlay.assert_not_called()
        self.assertIsNone(
            compose_ghost.call_args.kwargs["background_mask"]
        )

    def test_不使用select保持原有自动流程(self) -> None:
        (
            detect,
            select_masks,
            compose_overlay,
            compose_ghost,
            auto_background,
            _selected_masks,
            _selected,
            images,
            _output_dir,
        ) = self._run("both", False, None)

        select_masks.assert_not_called()
        detect.assert_called_once_with(images, 0)
        compose_overlay.assert_called_once()
        self.assertIs(
            compose_ghost.call_args.kwargs["background_mask"],
            auto_background,
        )


if __name__ == "__main__":
    unittest.main()
