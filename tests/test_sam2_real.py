import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from object_overlay.sam2_tracker import Sam2Tracker, resolve_checkpoint
from object_overlay.video import write_image


RUN_REAL = os.environ.get("RUN_SAM2_REAL") == "1"


@unittest.skipUnless(
    RUN_REAL,
    "设置 RUN_SAM2_REAL=1 才运行真实 SAM 2.1 GPU 推理",
)
class Sam2RealInferenceTest(unittest.TestCase):
    def test_真实模型传播与人工修正重传播(self) -> None:
        import torch

        self.assertTrue(torch.cuda.is_available())
        checkpoint = resolve_checkpoint(None)
        self.assertTrue(
            checkpoint.is_file(),
            f"缺少 SAM 2.1 权重：{checkpoint}",
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            video_path = root / "synthetic.mp4"
            self._make_video(video_path)
            frame_files, images = self._make_output_frames(root, 3.0, 0.5)
            tracker = Sam2Tracker(
                video_path,
                frame_files,
                images,
                root,
                0,
                0.5,
                target_track_fps=15.0,
                initial_box=(58, 218, 44, 44),
                checkpoint=checkpoint,
            )

            def accept_prompt(
                _self: Sam2Tracker,
                mask: np.ndarray,
                *,
                source: str = "AUTO",
            ) -> tuple[str, np.ndarray]:
                del source
                return "accept", mask

            with patch.object(
                Sam2Tracker,
                "_review_initial_mask",
                accept_prompt,
            ):
                started = time.perf_counter()
                tracker.prepare()
                propagation_seconds = time.perf_counter() - started

            self.assertTrue(np.any(tracker.masks[0] > 0))
            self.assertEqual(len(tracker.masks), len(images))
            self.assertTrue(np.any(tracker.background_mask > 0))

            correction_index = 2
            correction = self._target_mask(correction_index, 0.5, shift_x=20)
            before = [mask.copy() for mask in tracker.masks]
            tracker.correct(correction_index, correction)
            after = tracker.masks

            track_index = tracker._track_index(correction_index)
            self.assertIn(
                track_index,
                tracker._state["output_dict_per_obj"][0][
                    "cond_frame_outputs"
                ],
            )
            self.assertTrue(
                any(
                    start == track_index and not reverse
                    for start, reverse, _seen in tracker.propagation_log
                )
            )
            changed = [
                index
                for index in range(correction_index + 1, len(after))
                if not np.array_equal(before[index], after[index])
            ]
            self.assertTrue(
                changed,
                "人工修正后没有观察到任何后续帧被重新计算/改变",
            )

            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            print(
                "\nSAM2 real inference:"
                f" frames={len(images)}"
                f" track_frames={tracker._state['num_frames']}"
                f" init+propagate={propagation_seconds:.1f}s"
                f" peak_vram={peak_memory:.2f}GiB"
                f" changed_after_correction={changed}"
            )

    def _make_video(self, path: Path) -> None:
        width, height, fps, seconds = 640, 480, 30, 3
        rng = np.random.default_rng(7)
        background = np.full((height, width, 3), 35, np.uint8)
        background += rng.integers(0, 18, (height, width, 3), dtype=np.uint8)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        for index in range(fps * seconds):
            frame = background.copy()
            center_x = int(80 + (width - 180) * index / (fps * seconds - 1))
            center_y = int(240 + 20 * np.sin(index / 12.0))
            cv2.rectangle(
                frame,
                (center_x - 22, center_y - 22),
                (center_x + 22, center_y + 22),
                (245, 245, 245),
                -1,
            )
            writer.write(frame)

        writer.release()

    def _make_output_frames(
        self,
        root: Path,
        seconds: float,
        interval: float,
    ) -> tuple[list[Path], list[np.ndarray]]:
        video_path = root / "synthetic.mp4"
        capture = cv2.VideoCapture(str(video_path))
        source = []

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source.append(frame)

        capture.release()
        step = int(round(interval * 30))
        frame_files: list[Path] = []
        images: list[np.ndarray] = []

        for index in range(int(seconds / interval)):
            frame = source[index * step]
            path = root / f"frame_{index + 1:04d}.png"
            write_image(path, frame)
            frame_files.append(path)
            images.append(frame)

        return frame_files, images

    def _target_mask(
        self,
        output_index: int,
        interval: float,
        *,
        shift_x: int = 0,
    ) -> np.ndarray:
        source_index = int(round(output_index * interval * 30))
        center_x = int(80 + (640 - 180) * source_index / (30 * 3 - 1))
        center_y = int(240 + 20 * np.sin(source_index / 12.0))
        mask = np.zeros((480, 640), dtype=np.uint8)
        x = center_x + shift_x
        cv2.rectangle(
            mask,
            (x - 26, center_y - 26),
            (x + 26, center_y + 26),
            255,
            -1,
        )
        return mask


if __name__ == "__main__":
    unittest.main()
