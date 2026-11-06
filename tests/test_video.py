import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from object_overlay.video import prepare_frames, read_image


class PrepareFramesFallbackTest(unittest.TestCase):
    def test_ffmpeg写帧失败时回退OpenCV(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            video_path = root / "clip.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (64, 64),
            )

            if not writer.isOpened():
                self.skipTest("当前 OpenCV 后端无法写入测试视频")

            for index in range(10):
                frame = np.full(
                    (64, 64, 3),
                    index * 20,
                    dtype=np.uint8,
                )
                writer.write(frame)

            writer.release()
            frames_dir = root / "frames"
            metadata_file = root / "metadata.json"
            failure = subprocess.CalledProcessError(
                1,
                ["ffmpeg"],
            )

            with patch(
                "object_overlay.video.subprocess.run",
                side_effect=failure,
            ):
                frame_files = prepare_frames(
                    video_path,
                    frames_dir,
                    metadata_file,
                    0.5,
                )

            self.assertEqual(len(frame_files), 2)
            first = read_image(frame_files[0], cv2.IMREAD_COLOR)
            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first.shape, (64, 64, 3))

            metadata = json.loads(
                metadata_file.read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["interval"], 0.5)


if __name__ == "__main__":
    unittest.main()
