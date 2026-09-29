import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from object_overlay.compose import compose_ghost_overlay
from object_overlay.trajectory import (
    draw_trajectory,
    format_hex_color,
    mask_center,
    parse_hex_color,
    smooth_path,
)
from object_overlay.video import read_image, write_image


SAMPLES = 16


class MaskCenterTest(unittest.TestCase):
    def test_空掩膜返回None(self) -> None:
        self.assertIsNone(mask_center(np.zeros((10, 10), dtype=np.uint8)))

    def test_返回外接矩形中心(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2, 4] = 255
        mask[6, 8] = 255

        self.assertEqual(mask_center(mask), (6.0, 4.0))


class HexColorTest(unittest.TestCase):
    def test_六位十六进制解析成BGR(self) -> None:
        self.assertEqual(parse_hex_color("#65548F"), (143, 84, 101))

    def test_可以省略井号(self) -> None:
        self.assertEqual(parse_hex_color("65548F"), (143, 84, 101))

    def test_支持三位简写(self) -> None:
        self.assertEqual(parse_hex_color("#ABC"), (204, 187, 170))

    def test_大小写不敏感(self) -> None:
        self.assertEqual(parse_hex_color("#65548f"), (143, 84, 101))
        self.assertEqual(parse_hex_color("#aBc"), (204, 187, 170))

    def test_非法输入抛出异常(self) -> None:
        for text in ("#12345", "#GGGGGG", "", "#"):
            with self.assertRaises(ValueError):
                parse_hex_color(text)

    def test_回显为十六进制(self) -> None:
        self.assertEqual(format_hex_color((143, 84, 101)), "#65548F")
        self.assertEqual(format_hex_color((0, 0, 0)), "#000000")
        self.assertEqual(format_hex_color((255, 255, 255)), "#FFFFFF")


class TrajectoryColorTest(unittest.TestCase):
    def test_轨迹线使用指定颜色(self) -> None:
        samples = np.array([[10.0, 50.0], [90.0, 50.0]])
        taus = np.array([0.0, 1.0])

        for color in ((143, 84, 101), (255, 0, 0)):
            canvas = np.full((100, 100, 3), 200, dtype=np.uint8)
            draw_trajectory(canvas, samples, taus, 1.0, color)

            # 不透明度为 1 时，线条覆盖处应当就是所给颜色本身。
            np.testing.assert_array_equal(canvas[50, 85], color)

    def test_可以关闭描边(self) -> None:
        # 画布要够宽：线宽随画面宽度缩放，画布太小的话线条只有两三像素，
        # 描边和彩线的抗锯齿羽化会糊在一起，分不出两种设置的区别。
        samples = np.array([[10.0, 50.0], [3800.0, 50.0]])
        taus = np.array([0.0, 1.0])
        color = (0, 165, 255)

        with_outline = np.full((100, 3840, 3), 200, dtype=np.uint8)
        draw_trajectory(with_outline, samples, taus, 1.0, color, outline=True)

        without_outline = np.full((100, 3840, 3), 200, dtype=np.uint8)
        draw_trajectory(
            without_outline,
            samples,
            taus,
            1.0,
            color,
            outline=False,
        )

        # 描边比彩线宽一倍，第 45 行只落在描边里。具体哪几行属于描边，
        # 受抗锯齿羽化和顶点取整影响，是量出来的而不是算出来的。
        self.assertLess(int(with_outline[45, 3600, 0]), 60)
        self.assertEqual(int(without_outline[45, 3600, 0]), 200)

        # 彩线本身两种设置下完全一致。
        np.testing.assert_array_equal(
            with_outline[50, 3600],
            without_outline[50, 3600],
        )


class SmoothPathTest(unittest.TestCase):
    def test_曲线严格穿过每个控制点(self) -> None:
        points = [(0.0, 0.0), (40.0, 90.0), (120.0, 30.0), (200.0, 140.0)]
        samples, _ = smooth_path(points, samples_per_segment=SAMPLES)

        # 每段的第一个采样点就是该段起点，最后一个控制点单独补在末尾。
        for index in range(len(points) - 1):
            np.testing.assert_allclose(
                samples[index * SAMPLES],
                points[index],
                atol=1e-9,
            )

        np.testing.assert_allclose(samples[-1], points[-1], atol=1e-9)

    def test_间距极不均匀时结果仍然有限(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (500.0, 300.0), (1000.0, 0.0)]
        samples, taus = smooth_path(points, samples_per_segment=SAMPLES)

        self.assertTrue(np.all(np.isfinite(samples)))
        self.assertEqual(len(samples), len(taus))
        self.assertAlmostEqual(float(taus[0]), 0.0)
        self.assertAlmostEqual(float(taus[-1]), 1.0)

    def test_重复控制点不会除零(self) -> None:
        points = [(10.0, 10.0), (10.0, 10.0), (80.0, 40.0)]
        samples, _ = smooth_path(points, samples_per_segment=SAMPLES)

        self.assertTrue(np.all(np.isfinite(samples)))

    def test_时间参数单调递增并覆盖完整区间(self) -> None:
        points = [(0.0, 0.0), (50.0, 10.0), (60.0, 90.0), (300.0, 120.0)]
        taus = [0.0, 0.2, 0.25, 1.0]
        samples, sample_taus = smooth_path(
            points,
            taus,
            samples_per_segment=SAMPLES,
        )

        self.assertEqual(len(samples), len(sample_taus))
        self.assertTrue(np.all(np.diff(sample_taus) >= 0))
        self.assertAlmostEqual(float(sample_taus[0]), 0.0)
        self.assertAlmostEqual(float(sample_taus[-1]), 1.0)

    def test_只有两个控制点时退化为直线(self) -> None:
        points = [(0.0, 0.0), (100.0, 0.0)]
        samples, taus = smooth_path(points, samples_per_segment=4)

        self.assertEqual(len(samples), len(taus))
        np.testing.assert_allclose(samples[0], points[0], atol=1e-9)
        np.testing.assert_allclose(samples[-1], points[1], atol=1e-9)
        np.testing.assert_allclose(samples[:, 1], 0.0, atol=1e-9)

    def test_时间参数数量不匹配时抛出异常(self) -> None:
        with self.assertRaises(ValueError):
            smooth_path([(0.0, 0.0), (10.0, 10.0)], [0.0, 0.5, 1.0])


class GhostOverlayTest(unittest.TestCase):
    def _make_case(
        self,
        root: Path,
    ) -> tuple[list[Path], list[np.ndarray], list[np.ndarray]]:
        """构造三帧素材：固定背景 + 两个不同位置的目标。"""
        height = width = 80
        frames = [
            np.full((height, width, 3), 40, dtype=np.uint8)
            for _ in range(3)
        ]
        frames[1][10:16, 10:16] = 200
        frames[2][60:66, 60:66] = 200

        frame_paths = [
            root / f"frame_{index:04d}.png" for index in range(1, 4)
        ]

        for path, frame in zip(frame_paths, frames):
            write_image(path, frame)

        masks = [
            np.zeros((height, width), dtype=np.uint8) for _ in range(3)
        ]
        masks[1][10:16, 10:16] = 255
        masks[2][60:66, 60:66] = 255

        return frame_paths, masks, frames

    def test_最新时刻保持原始像素而最早时刻被混合(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            frame_paths, masks, frames = self._make_case(root)
            early, late = frames[1], frames[2]

            ghost = read_image(
                compose_ghost_overlay(
                    frame_paths,
                    masks,
                    base_index=0,
                    output_dir=root,                    min_alpha=0.2,
                ),
                cv2.IMREAD_UNCHANGED,
            )

            self.assertIsNotNone(ghost)
            assert ghost is not None
            self.assertEqual(ghost.shape, late.shape)
            self.assertEqual(ghost.dtype, late.dtype)

            # 最新时刻 alpha=1，即使轨迹线从下方穿过也被完全覆盖。
            np.testing.assert_array_equal(ghost[62, 62], late[62, 62])
            self.assertFalse(np.array_equal(ghost[12, 12], early[12, 12]))

    def test_不透明度为1时所有时刻都保持原始像素(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            frame_paths, masks, frames = self._make_case(root)
            early, late = frames[1], frames[2]

            ghost = read_image(
                compose_ghost_overlay(
                    frame_paths,
                    masks,
                    base_index=0,
                    output_dir=root,                    min_alpha=1.0,
                ),
                cv2.IMREAD_UNCHANGED,
            )

            assert ghost is not None
            np.testing.assert_array_equal(ghost[12, 12], early[12, 12])
            np.testing.assert_array_equal(ghost[62, 62], late[62, 62])

    def test_轨迹线绘制在目标之外的区域(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            frame_paths, masks, frames = self._make_case(root)
            base = frames[0]

            ghost = read_image(
                compose_ghost_overlay(
                    frame_paths,
                    masks,
                    base_index=0,
                    output_dir=root,                    min_alpha=0.2,
                ),
                cv2.IMREAD_UNCHANGED,
            )

            assert ghost is not None
            difference = np.abs(
                ghost.astype(np.int16) - base.astype(np.int16)
            ).max(axis=2)
            covered = (masks[1] > 0) | (masks[2] > 0)

            self.assertGreater(int(difference[~covered].max()), 0)

    def test_只有单个有效目标时不绘制轨迹线(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            frame_paths, masks, frames = self._make_case(root)
            base, early = frames[0], frames[1]
            masks[2][:] = 0

            ghost = read_image(
                compose_ghost_overlay(
                    frame_paths,
                    masks,
                    base_index=0,
                    output_dir=root,                    min_alpha=0.2,
                ),
                cv2.IMREAD_UNCHANGED,
            )

            assert ghost is not None

            # 唯一时刻的时间参数为 1.0，因此完全不透明。
            np.testing.assert_array_equal(ghost[12, 12], early[12, 12])

            difference = np.abs(
                ghost.astype(np.int16) - base.astype(np.int16)
            ).max(axis=2)
            self.assertEqual(int(difference[~(masks[1] > 0)].max()), 0)

    def test_背景帧里的目标会被擦除并作为第一个时刻参与渐隐(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            size = 100
            frames = [
                np.full((size, size), 30, dtype=np.uint8) for _ in range(5)
            ]
            frames[0][10:40, 10:40] = 200
            frames[4][60:90, 60:90] = 200

            frame_paths = [
                root / f"frame_{index:04d}.png" for index in range(1, 6)
            ]

            for path, frame in zip(frame_paths, frames):
                write_image(path, frame)

            background_mask = np.zeros((size, size), dtype=np.uint8)
            background_mask[10:40, 10:40] = 255

            masks = [
                np.zeros((size, size), dtype=np.uint8) for _ in range(5)
            ]
            masks[4][60:90, 60:90] = 255

            def render(background: np.ndarray | None) -> np.ndarray:
                return read_image(
                    compose_ghost_overlay(
                        frame_paths,
                        masks,
                        base_index=0,
                        output_dir=root,                        min_alpha=0.2,
                        background_mask=background,
                    ),
                    cv2.IMREAD_UNCHANGED,
                )

            without_erase = render(None)
            with_erase = render(background_mask)

            assert without_erase is not None
            assert with_erase is not None

            # 不擦除时，背景帧的目标以 alpha 混合它自己，结果仍是完整不透明。
            self.assertEqual(int(without_erase[14, 36]), 200)

            # 擦除后它先变成干净背景，再作为最早的时刻以最淡的不透明度合成。
            self.assertLess(int(with_erase[14, 36]), 120)

            # 最新时刻始终完全不透明。
            self.assertEqual(int(with_erase[64, 86]), 200)

            # 背景帧的目标进入了轨迹：最早与最新的时刻之间画出了线。
            self.assertNotEqual(int(with_erase[50, 50]), 30)

    def test_关闭轨迹线后目标之外的区域与背景完全一致(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            frame_paths, masks, frames = self._make_case(root)
            base, early = frames[0], frames[1]

            ghost = read_image(
                compose_ghost_overlay(
                    frame_paths,
                    masks,
                    base_index=0,
                    output_dir=root,
                    min_alpha=0.2,
                    show_trajectory=False,
                ),
                cv2.IMREAD_UNCHANGED,
            )

            assert ghost is not None

            # 没有画线：目标之外的像素与背景帧逐像素相同。
            difference = np.abs(
                ghost.astype(np.int16) - base.astype(np.int16)
            ).max(axis=2)
            covered = (masks[1] > 0) | (masks[2] > 0)
            self.assertEqual(int(difference[~covered].max()), 0)

            # 但渐隐本身仍在：最早的时刻不是完整不透明。
            self.assertFalse(np.array_equal(ghost[12, 12], early[12, 12]))

    def test_灰度画布同样可以绘制轨迹(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            height = width = 60
            frames = [
                np.full((height, width), 40, dtype=np.uint8) for _ in range(3)
            ]
            frames[1][8:14, 8:14] = 200
            frames[2][44:50, 44:50] = 200

            frame_paths = [
                root / f"frame_{index:04d}.png" for index in range(1, 4)
            ]

            for path, frame in zip(frame_paths, frames):
                write_image(path, frame)

            masks = [
                np.zeros((height, width), dtype=np.uint8) for _ in range(3)
            ]
            masks[1][8:14, 8:14] = 255
            masks[2][44:50, 44:50] = 255

            ghost = read_image(
                compose_ghost_overlay(
                    frame_paths,
                    masks,
                    base_index=0,
                    output_dir=root,                    min_alpha=0.3,
                ),
                cv2.IMREAD_UNCHANGED,
            )

            assert ghost is not None
            self.assertEqual(ghost.ndim, 2)
            np.testing.assert_array_equal(ghost[46, 46], frames[2][46, 46])

    def test_十六位深结果保持位深(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            height = width = 20
            frames: list[np.ndarray] = []
            masks = [
                np.zeros((height, width), dtype=np.uint8) for _ in range(3)
            ]

            for index in range(3):
                frame = np.full(
                    (height, width, 3),
                    1000,
                    dtype=np.uint16,
                )

                if index > 0:
                    row = 4 * index
                    frame[row : row + 3, row : row + 3] = 40000
                    masks[index][row : row + 3, row : row + 3] = 255

                frames.append(frame)

            frame_paths = [
                root / f"frame_{index:04d}.png" for index in range(1, 4)
            ]

            for path, frame in zip(frame_paths, frames):
                write_image(path, frame)

            ghost = read_image(
                compose_ghost_overlay(
                    frame_paths,
                    masks,
                    base_index=0,
                    output_dir=root,                    min_alpha=0.25,
                ),
                cv2.IMREAD_UNCHANGED,
            )

            assert ghost is not None
            self.assertEqual(ghost.dtype, np.uint16)
            np.testing.assert_array_equal(ghost[8, 8], frames[2][8, 8])


if __name__ == "__main__":
    unittest.main()
