"""
Numpy implementation of the calibration procedure described
in "A Toolbox for Easily Calibrating Omnidirectional Cameras," by Scaramuzza et al.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
logger.addHandler(handler)

def check_pattern_observations_shape(pattern_observations: np.ndarray) -> bool:
    """
    :param pattern_observations: Observations of the calibration pattern in the
    image space. The first dimension should be the number of patterns, the second
    dimension is the number of rows * cols in the pattern and the last dimension
    is 2 for the x-y coordinates of the pixels.
    :returns: True if the shape is correct.
    """
    return len(pattern_observations.shape) == 3 and pattern_observations.shape[-1] == 2

def check_pattern_world_coords_shape(pattern_world_coords: np.ndarray) -> bool:
    """
    :param pattern_world_coords: The coordinates of the calibration pattern
    in 3D. Should be Nx3 where n is rows * cols and 3 is the 3d coordinates
    (z is always zero).
    :returns: True if the shape is correct.
    """
    return len(pattern_world_coords.shape) == 2 and pattern_world_coords.shape[-1] == 3

def check_extrinsics_shape(extrinsics: np.ndarray) -> bool:
    """
    :param extrinsics: The array of extrinsics matrices. Each of these
    should be a 4x4 homogenous 3d transformation. So the shape must be
    Nx4x4 where N is the number of transformations.
    :returns: True if the shape is correct.
    """
    return len(extrinsics.shape) == 3 and extrinsics.shape[1:] == (4, 4)

def check_intrinsics_shape(intrinsics: np.ndarray) -> bool:
    """
    :param intrinics: The array of polynomial coefficients
    in descending order of power.
    :returns: True if the shape is exactly (5,)
    """
    return intrinsics.shape == (5,)

def generate_pattern_world_coords(pattern_rows: int, pattern_cols: int,
                                  pattern_square_size: float) -> np.ndarray:
    """
    Generate the Nx3 array of world coordinates for a pattern,
    assuming the top-left corner is at (0,0,0)  and the pattern
    lies on a place with z=0.
    :param pattern_rows: Number of corners per column.
    :param pattern_cols: Number of corners per row.
    :param pattern_square_size: The size, in metres, of each square in the calibration
    pattern (usually around 20-40mm).
    :returns: N,3 array of coordinates in world space where
    N = pattern_rows * pattern_cols. The coordinates are returned from
    left to right and top to bottom in row-major order.
    """
    assert pattern_rows > 0 and pattern_cols > 0 and pattern_square_size > 0
    # generate a grid of coordinates representing the pattern's 3d coordinates
    world_coords = np.stack(
        np.meshgrid(
            np.arange(pattern_cols) * pattern_square_size,
            np.arange(pattern_rows) * pattern_square_size,
        ),
        axis=-1
    ).reshape(-1, 2)
    world_coords = np.concatenate(
        [
            world_coords,
            np.ones((world_coords.shape[0], 1))
        ],
        axis=-1
    )
    return world_coords

def partial_extrinsics(
        pattern_observations: np.ndarray,
        pattern_world_coords: np.ndarray) -> np.ndarray:
    """
    Given observations of the calibration pattern, returns a (partial)
    estimate of the homogenous transformation representing the camera's
    position and orientation relative to the checkerboard.
    :param Observations of the calibration pattern (num_obs, N, 2), should
    be centred around the initial distortion centre (middle of the image).
    :param pattern_world_coords: Array of shape (R*C, 3) where R is the number
    of rows in the pattern and C is the number of columns in the pattern. The points
    should be in row-major order.
    :returns: The num_obsx4x4 homogenous transformation matrix with the element corresponding
    to the translation in the z-dimension set to np.nan for each one.
    """
    assert check_pattern_observations_shape(pattern_observations)
    assert check_pattern_world_coords_shape(pattern_world_coords)
    logger.debug(f"Computing partial extrinsics for {pattern_observations.shape[0]} observations.")
    # generate the (empty) result
    result = np.tile(
        np.concatenate(
            [
                np.full((3, 4), np.nan, dtype=pattern_observations.dtype),
                np.array([0, 0, 0, 1]).reshape(1, 4)
            ],
            axis=0
        ),
        (pattern_observations.shape[0], 1, 1)
    )
    # form the matrix containing the linear systems
    M = np.concatenate(
        [
            - pattern_observations[..., [1]] * pattern_world_coords[..., :2],
            pattern_observations[..., [0]] * pattern_world_coords[..., :2],
            pattern_observations[..., ::-1] * [-1, 1]
        ],
        axis=-1
    )
    # batch compute the solutions
    Vh = np.linalg.svd(M)[2]
    H = Vh.swapaxes(1 ,2)[..., -1]
    residual = np.sum((M @ H[..., None]) ** 2, axis=(1, 2))
    logger.debug(f"Computed initial solution with {residual=}")
    result[:, :2, :3] = H.reshape(-1, 3, 2).swapaxes(1, 2)
    # find the scaled r_13 by solving the 4th degree polynomial
    A1 = np.sum(H[..., :2] ** 2, axis=-1) - np.sum(H[..., 2:4] ** 2, axis=-1)
    A2 = (H[..., None, :2] @ H[..., 2:4, None]).squeeze((-1, -2))
    A3 = A2 ** 2
    for i in range(pattern_observations.shape[0]):
        roots = np.roots(
            [
                1, 0, A1[i], 0, -A3[i]
            ]
        )
        real_roots = np.real(roots[np.isclose(np.imag(roots), 0)])
        if real_roots.shape[0] == 0:
            logger.debug(f"No real roots found for observation {i+1}.") 
            continue
        positive_roots = real_roots[real_roots > 0]
        if positive_roots.shape[0] == 0:
            logger.debug(f"No positive roots found for observation {i+1}.")
            continue
        if positive_roots.shape[0] > 1:
            logger.debug(f"Several positive real roots were found for observations {i+1}.")
            logger.debug(f"{roots=}")
        result[i, 2, 0] = positive_roots[0]
    result[:, :2, 3] = H[:, -2:]
    result[:, 2, 1] = - A2 / result[:, 2, 0]
    result[:, :3, :] /= np.linalg.norm(result[:, :3, [0]], axis=1)[:, None, :]
    result[:, :3, 2] = np.cross(result[:, :3, 0], result[:, :3, 1])
    return result

def linear_intrinsics_and_z_translation(pattern_observations: np.ndarray,
                                        pattern_world_coords: np.ndarray,
                                        extrinsics: np.ndarray) -> np.ndarray:
    """
    Solves a system of linear equations to simultaneously find the
    z-translation and polynomial coefficients.
    :param pattern_observations: The pattern coordinates in image
    space centred around the initial centre of distortion (middle
    of the image). Size should be (N, M, 2) where N is the number
    of observations of the pattern and M is the total number of
    corners in the calibration pattern stored in row-major order.
    :param extrinsics: The (N, 4, 4) extrinsics transformation matrix
    computed by partial_extrinsics() which has all z-translation
    componens set to np.nan. This will be modified in-place such
    that z-components are set to the linear estimate.
    :returns: An array of 5 polynomial coefficients in descending
    order of power. The second-to-lowest power is always 0.
    """
    assert check_pattern_observations_shape(pattern_observations)
    assert check_pattern_world_coords_shape(pattern_world_coords)
    assert check_extrinsics_shape(extrinsics)
    logger.debug("Computing linear intrinsics and z-translation...")
    # compute rho, the radial distance
    rho = np.linalg.norm(pattern_observations, axis=-1)
    # setup A, B, C & D - Terms in the linear system of equations
    A = np.sum(
        pattern_world_coords[:, :2] * extrinsics[:, None, :2, 1],
        axis=-1
    ) + extrinsics[:, 1, [-1]]
    B = pattern_observations[..., 1] * np.sum(
        pattern_world_coords[:, :2] * extrinsics[:, None, :2, 2],
        axis=-1
    )
    C = np.sum(
        pattern_world_coords[:, :2] * extrinsics[:, None, :2, 0],
        axis=-1
    ) + extrinsics[:, 0, [-1]]
    D = pattern_observations[..., 0] * np.sum(
        pattern_world_coords[:, :2]  * extrinsics[:, None, :2, 2],
        axis=-1
    ) # potential mistake in paper, paper says to multiple by pattern_world_coords[:, [0]], but probs should be :2
    # put the linear system in matrix form M * H = b
    poly_rho = np.stack(
        [
            np.ones_like(rho), rho ** 2,
            rho ** 3, rho ** 4
        ],
        axis=-1
    )
    M = np.concatenate(
        [
            np.stack(
                [
                    A[..., None] * poly_rho,
                    C[..., None] * poly_rho
                ],
                axis=-1
            ).swapaxes(-1, -2).reshape(-1, 4),
            np.stack(
                [
                    np.stack(
                        [np.diag(-a[:, 1]) for a in pattern_observations.swapaxes(0, 1)],
                        axis=0
                    ),
                    np.stack(
                        [np.diag(-a[:, 0]) for a in pattern_observations.swapaxes(0, 1)],
                        axis=0
                    ),
                ],
                axis=-1
            ).transpose(1, 0, 3, 2).reshape(-1, pattern_observations.shape[0]),
        ],
        axis=-1
    )
    b = np.stack(
        [
            B, D
        ],
        axis=-1
    ).flatten()
    H = np.linalg.lstsq(M, b)[0]
    residual = np.sum(
        (M @ H[..., None]) ** 2,
        axis=(-1, -2)
    )
    logger.debug(f"Computed a solution for linear intrinsics and z-translation with {residual=}")
    intrinsics = np.array([
        H[0], 0, *H[1:4]
    ])[::-1]
    extrinsics[:, 2, -1] = H[4:]
    logger.debug("Finished computing linear intrinsics and z-translation.")
    return intrinsics

def linear_refinement_extrinsics(pattern_observations: np.ndarray,
                      pattern_world_coords: np.ndarray, extrinsics: np.ndarray,
                      intrinsics: np.ndarray) -> np.ndarray:
    """
    Solves all linear equations simultanesouly
    using the estimated intrinsic parameters to
    refine the extrinsic parameters.
    :param pattern_observations: The pattern coordinates in image
    space centred around the initial centre of distortion (middle
    of the image). Size should be (N, M, 2) where N is the number
    of observations of the pattern and M is the total number of
    corners in the calibration pattern stored in row-major order.
    :param extrinsics: The (N, 4, 4) extrinsics transformation matrix
    computed by partial_extrinsics() which has all z-translation
    componens set to np.nan. This will be modified in-place such
    that z-components are set to the linear estimate.
    :param intrinsics: An array of 5 polynomial coefficients in descending
    order of power. The second-to-lowest power is always 0.
    :returns: The refined extrinsics with the same shape as the input
    extrinsics.
    """
    logger.debug("Performing a linear refinement of the extrinsic parameters.")
    assert check_extrinsics_shape(extrinsics)
    assert check_pattern_observations_shape(pattern_observations)
    assert check_pattern_world_coords_shape(pattern_world_coords)
    assert check_intrinsics_shape(intrinsics)
    # create the empty result array
    result = np.zeros_like(extrinsics)
    result[:, -1, -1] = 1
    # compute the radial distance for the observations
    rho = np.linalg.norm(pattern_observations, axis=-1)
    # store the shape of first two dims (n_obs, n_corners)
    base_shape = rho.shape
    # evaluate the model
    f_rho = np.polyval(intrinsics, rho)[..., None]
    # set up the system of linear homogenous equations M * H = 0
    # where H = [r_11, r_12, r_21, r_22, r_31, r_32 t_1, t_2, t_3]^T
    M = np.stack(
        [
            np.concatenate(
                [
                    np.zeros((*base_shape, 2)),
                    -pattern_world_coords[..., :2] * f_rho,
                    pattern_world_coords[..., :2] * pattern_observations[..., [1]],
                    np.zeros((*base_shape, 1)), - f_rho,
                    pattern_observations[..., [1]]
                ],
                axis=-1
            ),
            np.concatenate(
                [
                    pattern_world_coords[..., :2] * f_rho, np.zeros((*base_shape, 2)),
                    -pattern_world_coords[..., :2] * pattern_observations[..., [0]],
                    f_rho, np.zeros((*base_shape, 1)), -pattern_observations[..., [0]]
                ],
                axis=-1
            ),
            np.concatenate(
                [
                    - pattern_observations[..., [1]] * pattern_world_coords[..., :2],
                    pattern_observations[..., [0]] * pattern_world_coords[..., :2],
                    np.zeros((*base_shape, 2)),
                    pattern_observations[..., ::-1] * [-1, 1], np.zeros((*base_shape, 1))
                ],
                axis=-1
            ),
        ],
        axis=-1
    ).swapaxes(-1, -2).reshape(len(pattern_observations), -1, 9)
    # TODO: below is a target for refactor => residual = solve_and_scale(M, extrinsics)
    Vh = np.linalg.svd(M)[-1]
    H = Vh[:, -1, :]
    # for numerical stability
    residual = np.sum((M @ H[..., None]) ** 2, axis=(1, 2))
    logger.debug(f"Computed solution for extrinsics refinement with {residual=}")
    # ignore the solution for r_31 and r_32 and recompute it using
    # the cross product for stability
    result[..., :2, :2] = H[:, :4].reshape(-1, 2, 2).swapaxes(-1, -2)
    result[..., :3, -1] = H[:, -3:]
    # find the value of r_13 such that r_1 and r_2 are orthogonal
    # and of the same magnitude
    A1 = np.sum(H[..., :2] ** 2, axis=-1) - np.sum(H[..., 2:4] ** 2, axis=-1)
    A2 = (H[..., None, :2] @ H[..., 2:4, None]).squeeze((-1, -2))
    A3 = A2 ** 2
    for i in range(pattern_observations.shape[0]):
        roots = np.roots(
            [
                1, 0, A1[i], 0, -A3[i]
            ]
        )
        real_roots = np.real(roots[np.isclose(np.imag(roots), 0)])
        if real_roots.shape[0] == 0:
            logger.debug(f"No real roots found for observation {i+1}.") 
            continue
        positive_roots = real_roots[real_roots > 0]
        if positive_roots.shape[0] == 0:
            logger.debug(f"No positive roots found for observation {i+1}.")
            continue
        if positive_roots.shape[0] > 1:
            logger.debug(f"Several positive real roots were found for observations {i+1}.")
            logger.debug(f"{roots=}")
        result[i, 2, 0] = np.max(positive_roots)
    result[:, 2, 1] = - A2 / result[:, 2, 0]
    result[:, :3, :-1] /= np.linalg.norm(result[:, :3, [0]], axis=1)[:, None, :]
    result[:, :3, 2] = np.cross(result[:, :3, 0], result[:, :3, 1])
    return result

def linear_refinement_intrinsics(pattern_observations: np.ndarray,
                                 pattern_world_coords: np.ndarray,
                                 extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Uses the refined extrinsics from linear_refinement_extrinsics
    to solve a linear system of equations and thus improve the
    current estimate for the intrinsic parameters.
    :param pattern_observations: The pattern coordinates in image
    space centred around the initial centre of distortion (middle
    of the image). Size should be (N, M, 2) where N is the number
    of observations of the pattern and M is the total number of
    corners in the calibration pattern stored in row-major order.
    :param extrinsics: The (N, 4, 4) extrinsics transformation matrix
    computed by partial_extrinsics() which has all z-translation
    componens set to np.nan. This will be modified in-place such
    that z-components are set to the linear estimate.
    :param intrinsics: An array of 5 polynomial coefficients in descending
    order of power. The second-to-lowest power is always 0.
    :returns: The refined intrinsics with the same shape as the input
    intrinsics.
    """
    logger.debug("Performing linear refinement of intrinsics parameters...")
    assert check_pattern_observations_shape(pattern_observations)
    assert check_pattern_world_coords_shape(pattern_world_coords)
    assert check_extrinsics_shape(extrinsics)
    assert check_intrinsics_shape(intrinsics)
    # create the empty result array
    result = np.zeros_like(intrinsics)
    # compute rho, the radial distance
    rho = np.linalg.norm(
        pattern_observations,
        axis=-1,
    )
    # variable names for the constants used in the matrix
    A = np.sum(
        pattern_world_coords[:, :2] * extrinsics[:, None, :2, 1],
        axis=-1
    ) + extrinsics[:, 1, [-1]]
    B = pattern_observations[..., 1] * (np.sum(
        pattern_world_coords[:, :2] * extrinsics[:, None, :2, 2],
        axis=-1
    ) + extrinsics[:, 2, [-1]])
    C = np.sum(
        pattern_world_coords[:, :2] * extrinsics[:, None, :2, 0],
        axis=-1
    ) + extrinsics[:, 0, [-1]]
    D = pattern_observations[..., 0] * (np.sum(
        pattern_world_coords[:, :2]  * extrinsics[:, None, :2, 2],
        axis=-1
    ) + extrinsics[:, 2, [-1]])
    # setup the linear system of equations Tx = Y
    T = np.stack(
        [
            np.stack(
                [
                    A, rho**2 * A,
                    rho ** 3 * A, rho ** 4 * A
                ],
                axis=-1
            ),
            np.stack(
                [
                    C, rho**2 * C,
                    rho ** 3 * C, rho ** 4 * C
                ],
                axis=-1
            )
        ],
        axis=-2
    ).reshape(-1, 4)
    Y = np.stack(
        [
            B, D
        ],
        axis=-1
    ).flatten()
    x = np.linalg.lstsq(T, Y)[0]
    residual = np.sum(
        (T @ x[..., None]) ** 2,
        axis=(-1, -2)
    )
    logger.debug(f"Computed solution for intrinsic refinement with {residual=}")
    result[[4, 2, 1, 0]] = x
    logger.debug(f"Computed linear refinement of intrinsic parameters: {result=}")
    return result

# def nonlinear_refinement():
#     pass

# def optimise_centre():
#     pass

# def calibrate():
#     extrinsics = partial_extrinsics()
#     intrinsics = linear_intrinsics_and_z_translation(extrinsics)
#     linear_refinement(intrinsics, extrinsics)
#     nonlinear_refinement(intrinsics, extrinsics)

if __name__ == '__main__':
    from fisheye.corners import find_corners_opencv
    from glob import glob
    import cv2
    paths = glob('../fisheye-imgs/*.tif')
    corners = []
    show = False
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        c = find_corners_opencv(img, 9, 6)
        if c is not None:
            corners.append(c)
        if show:
            cv2.namedWindow('Corners')
            cv2.moveWindow('Corners', 30, 30)
            disp = img.copy()
            disp = cv2.drawChessboardCorners(disp, (6, 9), c, c is not None)
            disp = cv2.resize(disp, (1280, 720))
            cv2.imshow('Corners', disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    corners = np.stack(corners, axis=0).squeeze()
    corners -= np.array([img.shape[1], img.shape[0]], dtype=np.float64) / 2
    pattern_world_coords = generate_pattern_world_coords(
        9, 6, 0.034
    )
    extr = partial_extrinsics(
        corners, pattern_world_coords
    )
    print(extr)
    intrinsics = linear_intrinsics_and_z_translation(
        corners, pattern_world_coords, extr
    )
    print("intrinsics")
    print(intrinsics)
    print("extrinsics")
    print(extr)
    
