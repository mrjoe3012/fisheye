import numpy as np
from test_scaramuzza import TestScaramuzza as T
from fisheye.scaramuzza import logger
from fisheye.scaramuzza import partial_extrinsics, linear_intrinsics_and_z_translation, linear_refinement_extrinsics, linear_refinement_intrinsics, project, nonlinear_refinement
from fisheye.corners import find_corners_opencv
from glob import glob
from pyocamcalib.modelling.camera import Camera
import cv2
import matplotlib.pyplot as plt
logger.setLevel('DEBUG')

cam = Camera()
cam.taylor_coefficient = T.true_intrinsic_parameters[::-1].copy()
cam.distortion_center = T.true_distortion_centre.copy()

T.pattern_cols = 6
T.pattern_rows = 9
T.pattern_tile_size = 0.024
pattern_world_coords = np.concatenate(
    [
        np.stack(
            np.meshgrid(
                np.arange(T.pattern_cols) * T.pattern_tile_size,
                np.arange(T.pattern_rows) * T.pattern_tile_size
            ),
            axis=-1
        ),
        np.zeros((T.pattern_rows, T.pattern_cols, 1))
    ],
    axis=-1
).reshape(-1, 3)
# pattern_cam_coords = np.stack(
#     [
#         cam.world2cam(pattern_world_coords, extrinsics=T.true_world_2_cam_transformation_1),
#         cam.world2cam(pattern_world_coords, extrinsics=T.true_world_2_cam_transformation_2)
#     ],
#     axis=0
# )

# imgs = glob('/home/joe/source/fisheye-imgs/*.tif')
# pattern_cam_coords = []
# for img in imgs:
#     img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#     if img_gray is None:
#         raise RuntimeError()
#     corners = find_corners_opencv(img_gray, T.pattern_rows, T.pattern_cols)
#     if corners is None:
#         continue
#     pattern_cam_coords.append(corners)
# pattern_cam_coords = np.stack(pattern_cam_coords, axis=0).reshape(len(pattern_cam_coords), -1, 2)

with open('pattern_cam_coords.p', 'rb') as f:
    import pickle
    pattern_cam_coords = pickle.load(f)
extr_init = partial_extrinsics(pattern_cam_coords - T.true_distortion_centre, pattern_world_coords)
intrinsics_init = linear_intrinsics_and_z_translation(pattern_cam_coords - T.true_distortion_centre, pattern_world_coords, extr_init)
extr = linear_refinement_extrinsics(pattern_cam_coords - T.true_distortion_centre, pattern_world_coords, extr_init, intrinsics_init)
intrinsics = linear_refinement_intrinsics(pattern_cam_coords - T.true_distortion_centre, pattern_world_coords, extr, intrinsics_init)
extr, intrinsics = nonlinear_refinement(pattern_cam_coords - T.true_distortion_centre, pattern_world_coords, extr, intrinsics, 2)
# intrinsics = intrinsics_init
# extr = extr_init
# plot the linear polynomial versus the true one
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
x = np.linspace(0, 1500, 250)
y1 = np.polyval(intrinsics, x)
y2 = np.polyval(cam.taylor_coefficient[::-1], x)
ax.plot(x, y1, label='Linear Estimate')
ax.plot(x, y2, label='True Values')
ax.set_title("Linear Estimate of Intrinsics")
ax.legend()
ax.grid()
plt.show()
# plot extrinsics
fig = plt.figure(figsize=(12, 9))
ax = fig.add_axes(111, projection='3d')
ax.set_aspect('equal')
# ax.set_xlim((-1, -1))
# ax.set_ylim((-1, -1))
# ax.set_zlim((-1, -1))
ax.set_title('Extrinsics')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
for i in range(len(extr)):
    coords = (extr[i, :3, :3] @ pattern_world_coords[..., None]).squeeze(-1) + extr[i, :3, -1]
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
for color, vec in zip(('red', 'green', 'blue'), np.eye(3, dtype=np.float64)):
    ax.quiver(0, 0, 0, *vec, normalize=True, length=0.1, color=color)
plt.show()
# show the reprojection error
cam.taylor_coefficient = intrinsics[::-1].copy()
for i in range(len(pattern_cam_coords)):
    disp = np.zeros(
        (int(T.true_distortion_centre[1]*2), int(T.true_distortion_centre[0]*2), 3),
        dtype=np.uint8
    )
    disp = cv2.drawChessboardCorners(
        disp, (T.pattern_cols, T.pattern_rows), pattern_cam_coords[i].reshape(-1, 2, 1).astype(np.float32),
        True
    )
    disp = cv2.circle(disp, (int(pattern_cam_coords[i, 0, 0]), int(pattern_cam_coords[i, 0, 1])), 15, (0, 255, 0), 3)
    # projected = cam.world2cam(pattern_world_coords, extr[i][:3, :])
    projected = project(
        (extr[i, None, :3, :3] @ pattern_world_coords[..., None]).squeeze(-1) + extr[i, :3, -1], intrinsics
    ) + T.true_distortion_centre
    for corner in projected.reshape(-1, 2):
        corner[np.isnan(corner)] = -10000
        disp = cv2.circle(disp, (int(corner[0]), int(corner[1])), 10, (0, 0, 255), 3)
    disp = cv2.resize(disp, (1280, 720))
    cv2.namedWindow(f"Corners {i=}")
    cv2.moveWindow(f"Corners {i=}", 30, 30)
    cv2.imshow(f"Corners {i=}", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

