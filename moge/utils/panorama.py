import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
from typing import *
import itertools
import json
import warnings

import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm, trange
from scipy.sparse import csr_array, hstack, vstack
from scipy.ndimage import convolve
from scipy.sparse.linalg import lsmr

try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d


def get_panorama_cameras():
    vertices, _ = utils3d.np.create_icosahedron_mesh()
    intrinsics = utils3d.np.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.np.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)


def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions


def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)


def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, resolution: int):
    height, width = image.shape[:2]
    uv = utils3d.np.uv_map((resolution, resolution))
    splitted_images = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(utils3d.np.unproject_cv(uv, np.ones_like(uv[..., 0]), extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.np.uv_to_pixel(spherical_uv, (height, width)).astype(np.float32)

        splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR)    
        splitted_images.append(splitted_image)
    return splitted_images


def poisson_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> Tuple[csr_array, ndarray]:
    grid_index = np.arange(height * width).reshape(height, width)
    grid_index = np.pad(grid_index, ((0, 0), (1, 1)), mode='wrap' if wrap_x else 'edge')
    grid_index = np.pad(grid_index, ((1, 1), (0, 0)), mode='wrap' if wrap_y else 'edge')
    
    data = np.array([[-4, 1, 1, 1, 1]], dtype=np.float32).repeat(height * width, axis=0).reshape(-1)
    indices = np.stack([
        grid_index[1:-1, 1:-1],
        grid_index[:-2, 1:-1],         # up
        grid_index[2:, 1:-1],          # down
        grid_index[1:-1, :-2],         # left
        grid_index[1:-1, 2:]           # right
    ], axis=-1).reshape(-1)                                                                 
    indptr = np.arange(0, height * width * 5 + 1, 5) 
    A = csr_array((data, indices, indptr), shape=(height * width, height * width))
    
    return A


def grad_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> Tuple[csr_array, np.ndarray]:
    grid_index = np.arange(width * height).reshape(height, width)
    if wrap_x:
        grid_index = np.pad(grid_index, ((0, 0), (0, 1)), mode='wrap')
    if wrap_y:
        grid_index = np.pad(grid_index, ((0, 1), (0, 0)), mode='wrap')

    data = np.concatenate([
        np.concatenate([
            np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),       # x[i,j-1]           
        ], axis=1).reshape(-1),
        np.concatenate([
            np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),       # x[i-1,j]           
        ], axis=1).reshape(-1),
    ])
    indices = np.concatenate([
        np.concatenate([
            grid_index[:, :-1].reshape(-1, 1),
            grid_index[:, 1:].reshape(-1, 1),
        ], axis=1).reshape(-1),
        np.concatenate([
            grid_index[:-1, :].reshape(-1, 1),
            grid_index[1:, :].reshape(-1, 1),
        ], axis=1).reshape(-1),
    ])
    indptr = np.arange(0, grid_index.shape[0] * (grid_index.shape[1] - 1) * 2 + (grid_index.shape[0] - 1) * grid_index.shape[1] * 2 + 1, 2)
    A = csr_array((data, indices, indptr), shape=(grid_index.shape[0] * (grid_index.shape[1] - 1) + (grid_index.shape[0] - 1) * grid_index.shape[1], height * width))

    return A


def merge_panorama_depth(width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics: List[np.ndarray]):
    if max(width, height) > 256:
        panorama_depth_init, _ = merge_panorama_depth(width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics)
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    else:
        panorama_depth_init = None

    uv = utils3d.np.uv_map(height, width)
    spherical_directions = spherical_uv_to_directions(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    panorama_log_distance_values = []  # Fix #5: raw (0th-order) values per view, for anchoring
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = utils3d.np.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
        
        projected_pixels = utils3d.np.uv_to_pixel(np.clip(projected_uv, 0, 1), distance_maps[i].shape).astype(np.float32)
        
        # Fix #3: the previous version remapped the value channel with
        # INTER_LINEAR and the mask channel with INTER_NEAREST at the same
        # coordinates. Those disagree right at valid/invalid boundaries:
        # NEAREST can round to "valid" while LINEAR still blends in a large
        # fraction of the invalid source pixel (measured: up to 31% error on
        # a pixel trusted as valid). Fix via alpha-premultiplication: remap
        # (value * mask) and mask separately, both with INTER_LINEAR, then
        # divide -- invalid source data contributes exactly zero to the
        # numerator, so it cannot leak into the result, and the mask decision
        # (weight > 0.5) uses the same continuous interpolation as the value.
        log_splitted_distance = np.log(distance_maps[i])
        valid_weight = cv2.remap(pred_masks[i].astype(np.float32), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        weighted_log_distance = cv2.remap(log_splitted_distance * pred_masks[i].astype(np.float32), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        safe_log_distance = weighted_log_distance / np.clip(valid_weight, 1e-6, None)
        panorama_log_distance_map = np.where(projection_valid_mask, safe_log_distance, 0)
        panorama_pred_mask = projection_valid_mask & (valid_weight > 0.5)

        # calculate gradient map
        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode='wrap')
        grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

        padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
        mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]
        
        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # calculate laplacian map
        padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        mask = convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(mask)
        
        panorama_pred_masks.append(panorama_pred_mask)  
        panorama_log_distance_values.append(panorama_log_distance_map)  # Fix #5
        
    panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
    panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

    panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(panorama_grad_mask_x, axis=0).clip(1e-3)
    panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(panorama_grad_mask_y, axis=0).clip(1e-3)

    panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(panorama_laplacian_masks, axis=0).clip(1e-3)

    grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
    grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

    # --- Fix #5: 0th-order anchor to remove the global scale null-space ---
    # A as built below has only 1st/2nd-derivative rows, so any constant C
    # added to x leaves every row satisfied (A(x+C) = Ax): the absolute
    # scale is otherwise fixed only implicitly by the coarse x0 guess, and
    # measurably drifts (~-49% observed under 5% per-view noise in testing).
    # Anchor each valid pixel lightly to the masked average of the raw
    # per-view log-distance observations already captured above.
    panorama_pred_masks_stack = np.stack(panorama_pred_masks, axis=0)              # [V, H, W]
    panorama_log_distance_values_stack = np.stack(panorama_log_distance_values, 0)  # [V, H, W]
    anchor_value = (
        np.sum(panorama_log_distance_values_stack * panorama_pred_masks_stack, axis=0)
        / np.sum(panorama_pred_masks_stack, axis=0).clip(1e-3)
    )
    anchor_mask = np.any(panorama_pred_masks_stack, axis=0).reshape(-1)
    anchor_weight = 0.01  # small relative to the unit-weighted grad/Laplacian rows
    n_anchor = int(anchor_mask.sum())
    anchor_rows = csr_array(
        (np.full(n_anchor, anchor_weight, dtype=np.float32),
         (np.arange(n_anchor), np.flatnonzero(anchor_mask))),
        shape=(n_anchor, height * width),
    )

    # Solve overdetermined system
    A = vstack([
        grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
        poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
        anchor_rows,
    ])
    b = np.concatenate([
        panorama_log_distance_grad_x.reshape(-1)[grad_x_mask], 
        panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
        panorama_laplacian_map.reshape(-1)[laplacian_mask],
        anchor_weight * anchor_value.reshape(-1)[anchor_mask],
    ])
    x, *_ = lsmr(
        A, b, 
        atol=1e-5, btol=1e-5,
        x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None, 
        show=False,
    )
    
    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    return panorama_depth, panorama_mask
         
