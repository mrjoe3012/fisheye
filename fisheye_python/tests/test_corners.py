"""
This file contains the tests for the corner detection functions.
"""
from fisheye.corners import find_corners_opencv
import pytest
import cv2
import os
import numpy as np

class TestCornerDetectionFunctions:
    black_img_gray = np.zeros(
        (800, 600),
        dtype=np.uint8
    )
    black_img_colour = np.zeros(
        (1920, 1080, 3),
        dtype=np.uint8
    )
    img_gray = cv2.imread(
        os.path.join(
            os.path.dirname(__file__),
            'data',
            'board.png'
        ),
        cv2.IMREAD_GRAYSCALE
    )

    def test_find_corners_black_img(self) -> None:
        corners = find_corners_opencv(
            self.black_img_gray, 9, 6
        )
        assert corners is None

    def test_find_corners_real_img(self) -> None:
        corners = find_corners_opencv(
            self.img_gray, 9, 6
        )
        assert type(corners) == np.ndarray

    def test_colour_image(self) -> None:
        with pytest.raises(Exception):
            find_corners_opencv(
                self.black_img_colour, 9, 6
            )

    def test_invalid_pattern_size(self) -> None:
        with pytest.raises(Exception):
            find_corners_opencv(
                self.black_img_gray, -1, -1
            )
        with pytest.raises(Exception):
            find_corners_opencv(
                self.black_img_gray, 0, 3
            )
