from pathlib import Path
import json
import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from typing import Tuple

def read_camera_colibs(transforms_path: Path):
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)

    camera_colibs = {
        Path(camera['file_path']).stem: camera
        for camera in transforms['frames']
    }

    return camera_colibs

def read_mask(mask_dir: Path, frame_id: int, camera_name: str, resize: Tuple[int, int]):
    mask_path = mask_dir / camera_name / f'{frame_id:05d}.png'
    mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, resize)
    mask = mask >= 128
    return mask

def read_ply(ply_dir: Path, frame_id: int):
    ply_path = ply_dir / f'time_{frame_id:05d}.ply'
    ply = PlyData.read(ply_path.as_posix())
    v = ply['vertex'].data

    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
    nxyz = np.stack([v['nx'], v['ny'], v['nz']], axis=1)
    color = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)
    opacity = v['opacity']
    scale = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1)
    rotation = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1)

    return xyz, nxyz, color, opacity, scale, rotation

def write_ply(ply_dir: Path, 
              frame_id: int,
              xyz: np.ndarray,
              nxyz: np.ndarray,
              color: np.ndarray,
              opacity: np.ndarray,
              scale: np.ndarray,
              rotation: np.ndarray):
    
    output_path = ply_dir / f'time_{frame_id:05d}.ply'
    
    N = xyz.shape[0]
    vertex_data = np.zeros(N, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ])

    # Fill in some example values
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]

    vertex_data['nx'] = nxyz[:, 0]
    vertex_data['ny'] = nxyz[:, 1]
    vertex_data['nz'] = nxyz[:, 2]

    vertex_data['f_dc_0'] = color[:, 0]
    vertex_data['f_dc_1'] = color[:, 1]
    vertex_data['f_dc_2'] = color[:, 2]

    vertex_data['opacity'] = opacity

    vertex_data['scale_0'] = scale[:, 0]
    vertex_data['scale_1'] = scale[:, 1]
    vertex_data['scale_2'] = scale[:, 2]

    vertex_data['rot_0'] = rotation[:, 0]
    vertex_data['rot_1'] = rotation[:, 1]
    vertex_data['rot_2'] = rotation[:, 2]
    vertex_data['rot_3'] = rotation[:, 3]

    vertex_element = PlyElement.describe(vertex_data, 'vertex')

    PlyData([vertex_element], text=False).write(output_path.as_posix())