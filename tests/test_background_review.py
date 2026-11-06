import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from object_overlay.masking import (
    BACKGROUND_CACHE_MASK_NAME,
    BACKGROUND_CACHE_STATE_NAME,
    BACKGROUND_SOURCE_MANUAL,
    BACKGROUND_SOURCE_SKIPPED,
    _save_background_selection,
    select_motion_masks,
)


SIZE = 80


def _make_images(count: int = 3) -> list[np.ndarray]:
    images = [
        np.full((SIZE, SIZE, 3), 40, dtype=np.uint8)
        for _ in range(count)
    ]
    images[0][10:18, 10:18] = 200

    for index in range(1, count):
        images[index][10 + index * 8 : 18 + index * 8, 55:63] = 200

    return images


def _make_masks(count: int = 3) -> list[np.ndarray]:
    masks = [
        np.zeros((SIZE, SIZE), dtype=np.uint8)
        for _ in range(count)
    ]

    for index in range(1, count):
        masks[index][10 + index * 8 : 18 + index * 8, 55:63] = 255

    return masks


def _state(corrected_dir: Path) -> dict:
    state_path = corrected_dir / BACKGROUND_CACHE_STATE_NAME
    return json.loads(state_path.read_text(encoding="utf-8"))


class BackgroundReviewTest(unittest.TestCase):
    def _run(
        self,
        keys: list[int],
        *,
        base_index: int = 0,
        image_count: int = 3,
        auto_mask: np.ndarray | None = None,
        force: bool = False,
        edit_mask: np.ndarray | None = None,
        background_review: bool = True,
        corrected_dir: Path,
    ) -> tuple[object, list[np.ndarray], list[str], object | None]:
        images = _make_images(image_count)
        masks = _make_masks(image_count)
        auto_mask = (
            np.zeros((SIZE, SIZE), dtype=np.uint8)
            if background_review and auto_mask is None
            else auto_mask
        )
        previews: list[np.ndarray] = []
        texts: list[str] = []

        def capture_text(
            _image: np.ndarray,
            text: str,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            texts.append(text)

        def capture_preview(_name: str, image: np.ndarray) -> None:
            previews.append(image.copy())

        context = [
            patch("cv2.namedWindow"),
            patch("cv2.imshow", side_effect=capture_preview),
            patch("cv2.waitKey", side_effect=keys),
            patch("cv2.destroyWindow"),
            patch("cv2.putText", side_effect=capture_text),
        ]

        if edit_mask is None:
            with context[0], context[1], context[2], context[3], context[4]:
                result = select_motion_masks(
                    images,
                    masks,
                    corrected_dir,
                    base_index=base_index,
                    interval=0.5,
                    background_auto_mask=auto_mask,
                    force=force,
                )
            edit_manual = None
        else:
            with (
                context[0],
                context[1],
                context[2],
                context[3],
                context[4],
                patch(
                    "object_overlay.masking._edit_manual_frame",
                    return_value=edit_mask,
                ) as edit_manual,
            ):
                result = select_motion_masks(
                    images,
                    masks,
                    corrected_dir,
                    base_index=base_index,
                    interval=0.5,
                    background_auto_mask=auto_mask,
                    force=force,
                )

        return result, previews, texts, edit_manual

    def test_ghost_select先复核背景帧并返回确认Mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"
            auto_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            auto_mask[10:18, 10:18] = 255

            result, _previews, texts, _edit = self._run(
                [13, 13, 13, 13],
                auto_mask=auto_mask,
                corrected_dir=corrected_dir,
            )

            self.assertTrue(texts[0].startswith("BASE FRAME"))
            self.assertIn("frame_0001", texts[0])
            self.assertIsNotNone(result.background_mask)
            np.testing.assert_array_equal(
                result.background_mask,
                auto_mask,
            )

            candidate_texts = [
                text
                for text in texts
                if text.startswith("frame_") and "BASE FRAME" not in text
            ]
            self.assertTrue(candidate_texts[0].startswith("frame_0002"))
            self.assertFalse(
                any("frame_0001" in text for text in candidate_texts)
            )
            self.assertEqual(
                cv2.countNonZero(result.selected_masks[0]),
                0,
            )

    def test_背景帧按M可手动修正并参与返回结果(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"
            auto_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            auto_mask[10:18, 10:18] = 255
            manual_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            manual_mask[35:45, 35:45] = 255

            result, _previews, _texts, edit_manual = self._run(
                [ord("m"), 13, 13, 13, 13],
                auto_mask=auto_mask,
                edit_mask=manual_mask,
                corrected_dir=corrected_dir,
            )

            self.assertIsNotNone(edit_manual)
            edit_manual.assert_called_once()
            call_args, call_kwargs = edit_manual.call_args
            self.assertEqual(call_args[0], 0)
            self.assertEqual(
                call_kwargs["mask_path"].name,
                BACKGROUND_CACHE_MASK_NAME,
            )
            np.testing.assert_array_equal(
                result.background_mask,
                manual_mask,
            )
            self.assertTrue(
                (corrected_dir / BACKGROUND_CACHE_MASK_NAME).exists()
            )
            state = _state(corrected_dir)
            self.assertEqual(state["source"], BACKGROUND_SOURCE_MANUAL)
            self.assertEqual(state["base_frame"], 1)

    def test_背景帧按R恢复自动Mask并删除人工缓存(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"
            auto_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            auto_mask[10:18, 10:18] = 255
            manual_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            manual_mask[35:45, 35:45] = 255
            _save_background_selection(
                corrected_dir,
                0,
                BACKGROUND_SOURCE_MANUAL,
                manual_mask,
            )

            result, _previews, texts, _edit = self._run(
                [ord("r"), 13, 13, 13, 13],
                auto_mask=auto_mask,
                corrected_dir=corrected_dir,
            )

            self.assertIn("Mask: MANUAL", texts[0])
            self.assertIn("Mask: AUTO", texts[3])
            np.testing.assert_array_equal(
                result.background_mask,
                auto_mask,
            )
            self.assertFalse(
                (corrected_dir / BACKGROUND_CACHE_MASK_NAME).exists()
            )
            self.assertFalse(
                (corrected_dir / BACKGROUND_CACHE_STATE_NAME).exists()
            )

    def test_背景帧按S跳过并记录明确状态(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"
            auto_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            auto_mask[10:18, 10:18] = 255

            result, _previews, _texts, _edit = self._run(
                [ord("s"), 13, 13, 13],
                auto_mask=auto_mask,
                corrected_dir=corrected_dir,
            )

            self.assertIsNone(result.background_mask)
            state = _state(corrected_dir)
            self.assertEqual(state["source"], BACKGROUND_SOURCE_SKIPPED)
            self.assertEqual(state["base_frame"], 1)
            self.assertFalse(
                (corrected_dir / BACKGROUND_CACHE_MASK_NAME).exists()
            )

    def test_空的人工Mask不等于跳过(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"
            empty_manual = np.zeros((SIZE, SIZE), dtype=np.uint8)
            _save_background_selection(
                corrected_dir,
                0,
                BACKGROUND_SOURCE_MANUAL,
                empty_manual,
            )

            result, _previews, texts, _edit = self._run(
                [13, 13, 13, 13],
                auto_mask=np.full((SIZE, SIZE), 255, dtype=np.uint8),
                corrected_dir=corrected_dir,
            )

            self.assertIsNotNone(result.background_mask)
            self.assertEqual(cv2.countNonZero(result.background_mask), 0)
            self.assertEqual(
                _state(corrected_dir)["source"],
                BACKGROUND_SOURCE_MANUAL,
            )
            self.assertIn("Mask: MANUAL", texts[0])

    def test_首个候选帧按B返回背景帧(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"
            auto_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            auto_mask[10:18, 10:18] = 255

            result, _previews, texts, _edit = self._run(
                [13, ord("b"), 13, 13, 13, 13],
                auto_mask=auto_mask,
                corrected_dir=corrected_dir,
            )

            base_texts = [
                text for text in texts if text.startswith("BASE FRAME")
            ]
            self.assertEqual(len(base_texts), 2)
            self.assertEqual(
                cv2.countNonZero(result.selected_masks[1]),
                cv2.countNonZero(_make_masks(3)[1]),
            )
            self.assertEqual(
                cv2.countNonZero(result.selected_masks[2]),
                cv2.countNonZero(_make_masks(3)[2]),
            )

    def test_base_frame_10按第10张抽帧复核并记录(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"
            auto_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            auto_mask[10:18, 10:18] = 255

            result, _previews, texts, _edit = self._run(
                [ord("s")] + [ord("s")] * 11 + [13],
                base_index=9,
                image_count=12,
                auto_mask=auto_mask,
                corrected_dir=corrected_dir,
            )

            self.assertIn("BASE FRAME", texts[0])
            self.assertIn("frame_0010", texts[0])
            self.assertIsNone(result.background_mask)
            self.assertEqual(_state(corrected_dir)["base_frame"], 10)

    def test_第二次运行复用人工修正缓存(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"
            manual_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            manual_mask[35:45, 35:45] = 255
            auto_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            auto_mask[10:18, 10:18] = 255

            self._run(
                [ord("m"), 13, 13, 13, 13],
                auto_mask=auto_mask,
                edit_mask=manual_mask,
                corrected_dir=corrected_dir,
            )

            different_auto = np.zeros((SIZE, SIZE), dtype=np.uint8)
            different_auto[60:70, 60:70] = 255
            result, _previews, texts, edit_manual = self._run(
                [13, 13, 13, 13],
                auto_mask=different_auto,
                corrected_dir=corrected_dir,
            )

            self.assertIsNone(edit_manual)
            self.assertIn("Mask: MANUAL", texts[0])
            np.testing.assert_array_equal(
                result.background_mask,
                manual_mask,
            )

    def test_force忽略人工缓存并恢复自动结果(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"
            manual_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            manual_mask[35:45, 35:45] = 255
            auto_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            auto_mask[10:18, 10:18] = 255
            _save_background_selection(
                corrected_dir,
                0,
                BACKGROUND_SOURCE_MANUAL,
                manual_mask,
            )

            result, _previews, texts, _edit = self._run(
                [13, 13, 13, 13],
                auto_mask=auto_mask,
                force=True,
                corrected_dir=corrected_dir,
            )

            self.assertIn("Mask: AUTO", texts[0])
            np.testing.assert_array_equal(
                result.background_mask,
                auto_mask,
            )
            self.assertFalse(
                (corrected_dir / BACKGROUND_CACHE_MASK_NAME).exists()
            )
            self.assertFalse(
                (corrected_dir / BACKGROUND_CACHE_STATE_NAME).exists()
            )

    def test_未启用背景复核时直接进入候选帧(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            corrected_dir = Path(temporary_dir) / "masks_corrected"

            result, _previews, texts, _edit = self._run(
                [13, 13, 13],
                background_review=False,
                corrected_dir=corrected_dir,
            )

            self.assertIsNone(result.background_mask)
            self.assertFalse(
                any(text.startswith("BASE FRAME") for text in texts)
            )
            self.assertTrue(texts[0].startswith("frame_0002"))


if __name__ == "__main__":
    unittest.main()
