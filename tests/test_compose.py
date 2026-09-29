import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from object_overlay.compose import compose_overlay
from object_overlay.masking import select_motion_masks
from object_overlay.video import read_image, write_image


class ComposeOverlayTest(unittest.TestCase):
    def test_mask内复制原始像素且背景不变(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            frame_1 = np.full((4, 4, 3), 100, dtype=np.uint16)
            frame_2 = np.full((4, 4, 3), 500, dtype=np.uint16)
            frame_2[1, 2] = (1000, 2000, 3000)

            frame_paths = [
                root / "frame_0001.png",
                root / "frame_0002.png",
            ]
            write_image(frame_paths[0], frame_1)
            write_image(frame_paths[1], frame_2)

            empty_mask = np.zeros((4, 4), dtype=np.uint8)
            object_mask = np.zeros((4, 4), dtype=np.uint8)
            object_mask[1, 2] = 255

            result = compose_overlay(
                frame_paths,
                [empty_mask, object_mask],
                base_index=0,
                output_dir=root,            )
            output = read_image(result.png_path, cv2.IMREAD_UNCHANGED)

            self.assertIsNotNone(output)
            assert output is not None
            self.assertEqual(output.dtype, np.uint16)
            np.testing.assert_array_equal(output[1, 2], frame_2[1, 2])
            np.testing.assert_array_equal(output[0, 0], frame_1[0, 0])
            self.assertEqual(result.background_max_difference, 0)

    def test_跳过后可回退且已纳入目标保留在预览中(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            images = [
                np.zeros((120, 120, 3), dtype=np.uint8)
                for _ in range(3)
            ]
            images[1][119, 119] = (10, 20, 30)
            images[2][119, 100] = (40, 50, 60)
            masks = [
                np.zeros((120, 120), dtype=np.uint8)
                for _ in range(3)
            ]
            masks[1][119, 119] = 255
            masks[2][119, 100] = 255
            previews: list[np.ndarray] = []
            keys = [ord("s"), ord("b"), 13, 13, 13]

            with (
                patch("cv2.namedWindow"),
                patch(
                    "cv2.imshow",
                    side_effect=lambda _name, image: previews.append(
                        image.copy()
                    ),
                ),
                patch("cv2.waitKey", side_effect=keys),
                patch("cv2.destroyWindow"),
            ):
                selected = select_motion_masks(
                    images,
                    masks,
                    Path(temporary_dir) / "masks_corrected",
                    base_index=0,
                    interval=0.5,
                )

            self.assertEqual(cv2.countNonZero(selected[1]), 1)
            self.assertEqual(cv2.countNonZero(selected[2]), 1)
            np.testing.assert_array_equal(
                previews[3][119, 119],
                images[1][119, 119],
            )

    def test_自动Mask不佳时可切换为人工修正Mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            images = [
                np.zeros((120, 120, 3), dtype=np.uint8)
                for _ in range(2)
            ]
            automatic_masks = [
                np.zeros((120, 120), dtype=np.uint8)
                for _ in range(2)
            ]
            automatic_masks[1][10, 10] = 255
            corrected_mask = np.zeros((120, 120), dtype=np.uint8)
            corrected_mask[50, 50] = 255

            with (
                patch("cv2.namedWindow"),
                patch("cv2.imshow"),
                patch("cv2.waitKey", side_effect=[ord("m"), 13, 13]),
                patch("cv2.destroyWindow"),
                patch(
                    "object_overlay.masking._edit_manual_frame",
                    return_value=corrected_mask,
                ) as edit_manual,
            ):
                selected = select_motion_masks(
                    images,
                    automatic_masks,
                    Path(temporary_dir) / "masks_corrected",
                    base_index=0,
                    interval=0.5,
                )

            edit_manual.assert_called_once()
            self.assertEqual(selected[1][10, 10], 0)
            self.assertEqual(selected[1][50, 50], 255)


if __name__ == "__main__":
    unittest.main()
