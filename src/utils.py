import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.file_io import read_mask

def project_points(points_world, camera_colib):
    c2w = np.asarray(camera_colib['transform_matrix'], dtype=np.float64)
    w2c = np.linalg.inv(c2w)

    N = points_world.shape[0]

    points_h = np.concatenate(
        [points_world, np.ones((N, 1))], axis=1
    )

    cam_h = (w2c @ points_h.T).T
    Xc, Yc, Zc = cam_h[:, 0], cam_h[:, 1], cam_h[:, 2]

    Yc = -Yc
    Zc = -Zc

    valid = Zc > 0
    Zc_safe = np.where(valid, Zc, 1.0)

    x = Xc / Zc_safe
    y = Yc / Zc_safe

    k1, k2, k3, k4 = camera_colib['k1'], camera_colib['k2'], camera_colib['k3'], camera_colib['k4']
    r2 = x * x + y * y
    dist = (
        1
        + k1 * r2
        + k2 * r2**2
        + k3 * r2**3
        + k4 * r2**4
    )

    x *= dist
    y *= dist

    u = camera_colib['fl_x'] * x + camera_colib['cx']
    v = camera_colib['fl_y'] * y + camera_colib['cy']

    pixels = np.stack([u, v], axis=1)
    return pixels, valid

def filter_static_gaussian_ids(xyz: np.ndarray,
                               mask: np.ndarray,
                               camera_colib: Dict[str, Any]):
    
    H = int(camera_colib['h'])
    W = int(camera_colib['w'])
    projected_xy, valid = project_points(xyz, camera_colib)
    projected_xy = projected_xy.round().astype(np.int32)
    out_of_range = np.logical_or(
        np.logical_or(projected_xy[:, 0] < 0, projected_xy[:, 0] >= W),
        np.logical_or(projected_xy[:, 1] < 0, projected_xy[:, 1] >= H)
    )
    out_of_range = np.logical_or(out_of_range, np.logical_not(valid))
    projected_xy[out_of_range] = 0

    masked = mask[projected_xy[:, 1], projected_xy[:, 0]]
    selected = np.logical_and(masked, np.logical_not(out_of_range))
    return np.where(selected)[0]

def filter_frame(xyz: np.ndarray,
                 frame_id: int,
                 camera_colibs: Dict[str, Any],
                 mask_dir: Path):
    
    N = xyz.shape[0]
    counts = np.zeros(N, np.uint32)
    
    for camera_name, camera_colib in camera_colibs.items():
        H = int(camera_colib['h'])
        W = int(camera_colib['w'])
        resize = (W, H)
        mask = read_mask(mask_dir, frame_id, camera_name, resize)
        filtered = filter_static_gaussian_ids(xyz, mask, camera_colib)
        counts[filtered] += 1

    return np.where(counts > 5)[0]

def inverse_ids(n: int, ids: np.ndarray):
    mask = np.ones(n, dtype=bool)
    mask[ids] = False
    return np.where(mask)[0]
