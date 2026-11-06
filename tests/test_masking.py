import io
import unittest
from unittest.mock import patch

import numpy as np

from object_overlay.masking import _select_target_roi


class SelectTargetRoiTest(unittest.TestCase):
    def test_框选窗口内显示操作说明(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        texts: list[str] = []

        def capture_text(
            _image: np.ndarray,
            text: str,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            texts.append(text)

        with (
            patch("cv2.selectROI", return_value=(10, 10, 20, 20)),
            patch("cv2.destroyWindow"),
            patch("cv2.putText", side_effect=capture_text),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            roi = _select_target_roi(image, 1)

        self.assertEqual(roi, (10, 10, 20, 20))
        self.assertTrue(
            any("Drag a box around the target" in text for text in texts)
        )
        self.assertTrue(any("confirm" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
