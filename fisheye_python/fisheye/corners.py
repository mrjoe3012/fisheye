"""
Functions for detecting corners in a calibration pattern. Also
provides a basic program for detecting corners and displaying them.
"""
from typing import Optional
import numpy as np
import cv2
import logging
import argparse

logger = logging.getLogger(__file__)

def find_corners_opencv(grayscale_img: np.ndarray, pattern_rows: int,
                 pattern_cols: int) -> Optional[np.ndarray]:
    """
    Finds corners in the provided image using opencv.
    :param grayscale_img: A grayscale image to detect
    corners within.
    :param pattern_rows: The number of rows in the calibration
    pattern.
    :param pattern_cols: The number of columns in the calibration
    pattern.
    :returns: The detected corners or None if the pattern
    could not be found.
    """
    logger.debug("Finding corners (opencv)")
    assert len(grayscale_img.shape) == 2
    found, corners = cv2.findChessboardCorners(
        grayscale_img,
        (pattern_rows, pattern_cols),
        flags=(cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
               + cv2.CALIB_CB_NORMALIZE_IMAGE)
    )
    if not found:
        logger.debug("Corners were not found.")
        return None
    logger.debug("Corners found. Refining corners...")
    window_size = (11, 11)
    zero_zone = (-1, -1)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    refined_corners = cv2.cornerSubPix(grayscale_img, corners, window_size, zero_zone, term)
    logger.debug("Finished refining corners.")
    return refined_corners

class ProgramArguments(argparse.Namespace):
    img_paths: list[str]
    pattern_rows: int
    pattern_cols: int

def main(arguments: Optional[ProgramArguments]) -> None:
    logging.basicConfig(level='DEBUG')
    if arguments is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('pattern_rows', type=int,
                            help='Number of (inner) rows in the pattern.')
        parser.add_argument('pattern_cols', type=int,
                            help='Number of (inner) columns in the pattern.')
        parser.add_argument('img_paths', nargs='+', type=str)
        args = parser.parse_args(
            namespace=ProgramArguments()
        )
        main(args)
        return
    pattern_size = (arguments.pattern_rows, arguments.pattern_cols)
    imgs = arguments.img_paths
    logger.info(f"{imgs=}")
    logger.info(f"{pattern_size=}")
    found = 0
    for path in imgs:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = find_corners_opencv(img_gray, pattern_size[0], pattern_size[1])
        disp = img
        if corners is not None:
            disp = cv2.drawChessboardCorners(disp, pattern_size, corners, True)
            found += 1
        disp = cv2.resize(disp, (800, int(800*(disp.shape[0]/disp.shape[1]))))
        cv2.imshow(f"{path=}", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    logger.info(f"Found corners for {found} / {len(imgs)} images.")

if __name__ == '__main__':
    main(None)
