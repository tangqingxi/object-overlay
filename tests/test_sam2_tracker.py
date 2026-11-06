import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from object_overlay.sam2_tracker import Sam2Tracker
from object_overlay.video import write_image
import cv2


SIZE = 80


class FakePredictor:
    """只模拟官方 API 的状态语义，不做真实推理。"""

    def __init__(self, num_frames: int) -> None:
        self.num_frames = num_frames
        self.init_calls = 0
        self.add_mask_calls: list[int] = []
        self.clear_calls: list[int] = []
        self.propagate_calls: list[tuple[int, bool]] = []

    def init_state(self, video_path: str, **kwargs: object) -> dict:
        del video_path, kwargs
        self.init_calls += 1
        return {
            "num_frames": self.num_frames,
            "anchors": {},
        }

    def add_new_points_or_box(
        self,
        state: dict,
        frame_idx: int,
        obj_id: int,
        box: np.ndarray | None = None,
        **_kwargs: object,
    ) -> tuple[int, list[int], np.ndarray]:
        del obj_id

        if box is None:
            mask = np.zeros((SIZE, SIZE), dtype=np.float32)
        else:
            x1, y1, x2, y2 = (int(value) for value in box)
            mask = np.zeros((SIZE, SIZE), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0

        state["anchors"][frame_idx] = mask > 0
        return frame_idx, [1], mask[None, None]

    def add_new_mask(
        self,
        state: dict,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
    ) -> tuple[int, list[int], np.ndarray]:
        del obj_id
        self.add_mask_calls.append(frame_idx)
        state["anchors"][frame_idx] = np.asarray(mask) > 0
        logits = state["anchors"][frame_idx].astype(np.float32)
        return frame_idx, [1], logits[None, None]

    def clear_all_prompts_in_frame(
        self,
        state: dict,
        frame_idx: int,
        obj_id: int,
    ) -> None:
        del obj_id
        self.clear_calls.append(frame_idx)
        state["anchors"].pop(frame_idx, None)

    def propagate_in_video(
        self,
        state: dict,
        start_frame_idx: int,
        reverse: bool = False,
        **_kwargs: object,
    ):
        self.propagate_calls.append((start_frame_idx, reverse))
        anchors = state["anchors"]
        order = (
            range(start_frame_idx, -1, -1)
            if reverse
            else range(start_frame_idx, self.num_frames)
        )

        for frame_idx in order:
            if frame_idx in anchors:
                mask = anchors[frame_idx]
            elif reverse:
                candidates = [
                    key for key in anchors if key >= frame_idx
                ]
                mask = (
                    anchors[min(candidates)]
                    if candidates
                    else np.zeros((SIZE, SIZE), dtype=bool)
                )
            else:
                candidates = [
                    key for key in anchors if key <= frame_idx
                ]
                mask = (
                    anchors[max(candidates)]
                    if candidates
                    else np.zeros((SIZE, SIZE), dtype=bool)
                )

            logits = mask.astype(np.float32)
            yield frame_idx, [1], logits[None, None]


def _make_case(root: Path, count: int = 6) -> tuple[list[Path], list[np.ndarray]]:
    images = [
        np.full((SIZE, SIZE, 3), 40, dtype=np.uint8)
        for _ in range(count)
    ]
    frame_files = [root / f"frame_{index:04d}.png" for index in range(1, count + 1)]

    for path, image in zip(frame_files, images):
        if not path.exists():
            write_image(path, image)

    return frame_files, images


class Sam2TrackerTest(unittest.TestCase):
    def _tracker(
        self,
        root: Path,
        predictor: FakePredictor,
        *,
        count: int = 6,
        base_index: int = 0,
        force: bool = False,
        target_track_fps: float = 2.0,
    ) -> tuple[Sam2Tracker, FakePredictor]:
        frame_files, images = _make_case(root, count)
        video_path = root / "clip.mp4"

        if not video_path.exists():
            video_path.write_bytes(b"placeholder")
        predictor.num_frames = count * max(
            1,
            int(round(target_track_fps / 2.0)),
        )
        tracker = Sam2Tracker(
            video_path,
            frame_files,
            images,
            root,
            base_index,
            0.5,
            target_track_fps=target_track_fps,
            predictor=predictor,
            initial_box=(10, 10, 20, 20),
            force=force,
        )
        return tracker, predictor

    def test_初始化后向前向反传播并建立缓存(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            predictor = FakePredictor(1)
            tracker, predictor = self._tracker(root, predictor)

            with patch.object(
                Sam2Tracker,
                "_review_initial_mask",
                return_value=("accept", np.ones((SIZE, SIZE), np.uint8) * 255),
            ):
                tracker.prepare()

            self.assertTrue(predictor.init_calls == 1)
            self.assertGreaterEqual(len(predictor.propagate_calls), 1)
            self.assertTrue(tracker.anchors_path.exists())
            self.assertTrue(tracker.manifest_path.exists())
            self.assertTrue((tracker.auto_dir / "frame_0001.png").exists())
            self.assertGreater(
                int(np.count_nonzero(tracker.masks[1])),
                0,
            )

    def test_按M修正后下游Mask更新且锚点保持(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tracker, predictor = self._tracker(root, FakePredictor(1))
            initial = np.zeros((SIZE, SIZE), dtype=np.uint8)
            initial[10:20, 10:20] = 255

            with patch.object(
                Sam2Tracker,
                "_review_initial_mask",
                return_value=("accept", initial),
            ):
                tracker.prepare()

            corrected = np.zeros((SIZE, SIZE), dtype=np.uint8)
            corrected[50:60, 50:60] = 255
            tracker.correct(3, corrected)

            np.testing.assert_array_equal(tracker.masks[3], corrected)
            np.testing.assert_array_equal(tracker.masks[4], corrected)
            np.testing.assert_array_equal(tracker.masks[2], initial)
            self.assertTrue(
                any(start == 3 for start, _ in predictor.propagate_calls)
            )

    def test_按R恢复人工锚点并重传播(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tracker, predictor = self._tracker(root, FakePredictor(1))
            initial = np.zeros((SIZE, SIZE), dtype=np.uint8)
            initial[10:20, 10:20] = 255

            with patch.object(
                Sam2Tracker,
                "_review_initial_mask",
                return_value=("accept", initial),
            ):
                tracker.prepare()

            corrected = np.zeros((SIZE, SIZE), dtype=np.uint8)
            corrected[50:60, 50:60] = 255
            tracker.correct(3, corrected)
            tracker.restore(3)

            self.assertNotIn(3, tracker.anchor_masks)
            np.testing.assert_array_equal(tracker.masks[3], initial)
            self.assertTrue(any(frame == 3 for frame in predictor.clear_calls))

    def test_恢复早期锚点不覆盖后期锚点(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tracker, _predictor = self._tracker(root, FakePredictor(1))
            initial = np.zeros((SIZE, SIZE), dtype=np.uint8)
            initial[10:20, 10:20] = 255

            with patch.object(
                Sam2Tracker,
                "_review_initial_mask",
                return_value=("accept", initial),
            ):
                tracker.prepare()

            early = np.zeros((SIZE, SIZE), dtype=np.uint8)
            early[30:40, 30:40] = 255
            late = np.zeros((SIZE, SIZE), dtype=np.uint8)
            late[50:60, 50:60] = 255
            tracker.correct(2, early)
            tracker.correct(5, late)
            tracker.restore(2)

            self.assertNotIn(2, tracker.anchor_masks)
            np.testing.assert_array_equal(tracker.masks[3], initial)
            np.testing.assert_array_equal(tracker.masks[5], late)

    def test_第二次运行复用缓存且不重复加载模型(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tracker, _ = self._tracker(root, FakePredictor(1))
            initial = np.zeros((SIZE, SIZE), dtype=np.uint8)
            initial[10:20, 10:20] = 255

            with patch.object(
                Sam2Tracker,
                "_review_initial_mask",
                return_value=("accept", initial),
            ):
                tracker.prepare()

            corrected = np.zeros((SIZE, SIZE), dtype=np.uint8)
            corrected[40:50, 40:50] = 255
            tracker.correct(2, corrected)

            second_predictor = FakePredictor(1)
            second, _ = self._tracker(root, second_predictor, force=False)

            with patch.object(
                Sam2Tracker,
                "_review_initial_mask",
                side_effect=AssertionError("不应重新初始化"),
            ):
                second.prepare()

            self.assertTrue(second._cache_loaded)
            self.assertEqual(second_predictor.init_calls, 0)
            np.testing.assert_array_equal(second.masks[2], corrected)
            self.assertIn(2, second.anchor_masks)

    def test_base_frame非首帧时向前后两个方向传播(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tracker, predictor = self._tracker(
                root,
                FakePredictor(1),
                base_index=3,
            )
            initial = np.zeros((SIZE, SIZE), dtype=np.uint8)
            initial[10:20, 10:20] = 255

            with patch.object(
                Sam2Tracker,
                "_review_initial_mask",
                return_value=("accept", initial),
            ):
                tracker.prepare()

            for frame_index in range(6):
                self.assertGreater(
                    int(np.count_nonzero(tracker.masks[frame_index])),
                    0,
                )

            # 初始化帧在 3：反向传播到 0，正向传播到 5。
            directions = {
                reverse for _start, reverse in predictor.propagate_calls
            }
            self.assertEqual(directions, {False, True})

            earlier = np.zeros((SIZE, SIZE), dtype=np.uint8)
            earlier[40:50, 40:50] = 255
            tracker.correct(1, earlier)
            np.testing.assert_array_equal(tracker.masks[1], earlier)
            np.testing.assert_array_equal(tracker.masks[2], earlier)

    def test_force保留人工锚点但重新传播(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tracker, _ = self._tracker(root, FakePredictor(1))
            initial = np.zeros((SIZE, SIZE), dtype=np.uint8)
            initial[10:20, 10:20] = 255

            with patch.object(
                Sam2Tracker,
                "_review_initial_mask",
                return_value=("accept", initial),
            ):
                tracker.prepare()

            corrected = np.zeros((SIZE, SIZE), dtype=np.uint8)
            corrected[40:50, 40:50] = 255
            tracker.correct(2, corrected)

            predictor = FakePredictor(1)
            forced, _ = self._tracker(root, predictor, force=True)
            forced.prepare()

            self.assertGreater(predictor.init_calls, 0)
            self.assertIn(2, forced.anchor_masks)
            np.testing.assert_array_equal(forced.masks[2], corrected)


if __name__ == "__main__":
    unittest.main()
