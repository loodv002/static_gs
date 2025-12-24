from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

from src.file_io import read_camera_colibs, read_ply, write_ply
from src.utils import filter_frame, inverse_ids, get_n_frames

os.chdir(Path(__file__).parent)

PLY_INPUT_DIR = Path('./0448_ply')
MASK_INPUT_DIR = Path('./0448_masks_static')
TRANSFORMS_PATH = Path('./transforms.json')

PLY_OUTPUT_DIR = Path('./output')
if not PLY_OUTPUT_DIR.exists(): PLY_OUTPUT_DIR.mkdir(parents=True)

N_FRAMES = get_n_frames(PLY_INPUT_DIR)

camera_colibs = read_camera_colibs(TRANSFORMS_PATH)

# Extract first frame
xyz_0, nxyz_0, color_0, opacity_0, scale_0, rotation_0 = read_ply(PLY_INPUT_DIR, 0)
gaussian_ids = filter_frame(xyz_0, 0, camera_colibs, MASK_INPUT_DIR)
write_ply(PLY_OUTPUT_DIR, 0, xyz_0, nxyz_0, color_0, opacity_0, scale_0, rotation_0)
xyz_0 = xyz_0[gaussian_ids]
nxyz_0 = nxyz_0[gaussian_ids]
color_0 = color_0[gaussian_ids]
opacity_0 = opacity_0[gaussian_ids]
scale_0 = scale_0[gaussian_ids]
rotation_0 = rotation_0[gaussian_ids]

for frame_id in tqdm(range(1, N_FRAMES)):
    xyz, nxyz, color, opacity, scale, rotation = read_ply(PLY_INPUT_DIR, frame_id)

    gaussian_ids = filter_frame(xyz, frame_id, camera_colibs, MASK_INPUT_DIR)
    inversed_ids = inverse_ids(xyz.shape[0], gaussian_ids)

    xyz = np.concatenate([xyz[inversed_ids], xyz_0], axis=0)
    nxyz = np.concatenate([nxyz[inversed_ids], nxyz_0], axis=0)
    color = np.concatenate([color[inversed_ids], color_0], axis=0)
    opacity = np.concatenate([opacity[inversed_ids], opacity_0], axis=0)
    scale = np.concatenate([scale[inversed_ids], scale_0], axis=0)
    rotation = np.concatenate([rotation[inversed_ids], rotation_0], axis=0)

    write_ply(PLY_OUTPUT_DIR, frame_id, xyz, nxyz, color, opacity, scale, rotation)