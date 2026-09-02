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
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = utils3d.np.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
        
        projected_pixels = utils3d.np.uv_to_pixel(np.clip(projected_uv, 0, 1), distance_maps[i].shape).astype(np.float32)
        
        log_splitted_distance = np.log(distance_maps[i])
        panorama_log_distance_map = np.where(projection_valid_mask, cv2.remap(log_splitted_distance, projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_pred_mask = projection_valid_mask & (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)

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

    # Solve overdetermined system
    A = vstack([
        grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
        poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
    ])
    b = np.concatenate([
        panorama_log_distance_grad_x.reshape(-1)[grad_x_mask], 
        panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
        panorama_laplacian_map.reshape(-1)[laplacian_mask]
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


def merge_panorama_depth_raycast(
    width: int, 
    height: int, 
    distance_maps: List[np.ndarray], 
    pred_masks: List[np.ndarray], 
    extrinsics: List[np.ndarray], 
    intrinsics: List[np.ndarray],
    normal_maps: Optional[List[np.ndarray]] = None,
    power: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Direct Spherical Cosine-Weighted Raycasting Fusion for 360 Panorama.
    Preserves exact metric scale and avoids polar distortion or integration drift.
    """
    uv = utils3d.np.uv_map(height, width)
    spherical_directions = spherical_uv_to_directions(uv)

    accum_dist = np.zeros((height, width), dtype=np.float32)
    accum_weight = np.zeros((height, width), dtype=np.float32)
    has_normals = normal_maps is not None and any(n is not None for n in normal_maps)
    accum_normal = np.zeros((height, width, 3), dtype=np.float32) if has_normals else None
    panorama_mask = np.zeros((height, width), dtype=bool)

    for i in range(len(distance_maps)):
        R = extrinsics[i][:3, :3]
        optical_axis = extrinsics[i][2, :3]

        projected_uv, projected_depth = utils3d.np.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        valid = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)

        projected_pixels = utils3d.np.uv_to_pixel(np.clip(projected_uv, 0, 1), distance_maps[i].shape).astype(np.float32)
        sampled_dist = cv2.remap(distance_maps[i], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR)
        sampled_mask = (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST) > 0) & valid

        cos_theta = np.clip(np.sum(spherical_directions * optical_axis, axis=-1), 0, 1)
        weight = np.where(sampled_mask, np.power(cos_theta, power), 0.0).astype(np.float32)

        accum_dist += sampled_dist * weight
        accum_weight += weight
        panorama_mask |= sampled_mask

        if has_normals and normal_maps[i] is not None:
            sampled_norm_cam = cv2.remap(normal_maps[i], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR)
            sampled_norm_world = sampled_norm_cam @ R
            accum_normal += sampled_norm_world * weight[..., None]

    nonzero = accum_weight > 1e-6
    panorama_depth = np.where(nonzero, accum_dist / np.maximum(accum_weight, 1e-6), 0.0).astype(np.float32)

    if accum_normal is not None:
        norm_len = np.linalg.norm(accum_normal, axis=-1, keepdims=True)
        panorama_normal = np.where(norm_len > 1e-6, accum_normal / np.maximum(norm_len, 1e-6), 0.0).astype(np.float32)
    else:
        panorama_normal = None

    return panorama_depth, panorama_mask, panorama_normal


def build_panorama_mesh_multiview(
    splitted_images: List[np.ndarray],
    splitted_points: List[np.ndarray],
    splitted_normals: Optional[List[np.ndarray]],
    splitted_masks: List[np.ndarray],
    splitted_extrinsics: List[np.ndarray],
    threshold: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Direct Multi-View 3D Mesh Reconstruction for 360 Panorama.
    Builds clean perspective meshes from each icosahedral view, eliminates flying edges,
    and rotates all vertices and normals into global world space.
    """
    vertices_list = []
    faces_list = []
    colors_list = []
    normals_list = []
    total_vertices = 0

    has_normals = splitted_normals is not None and any(n is not None for n in splitted_normals)

    for i in range(len(splitted_images)):
        points = splitted_points[i]
        mask = splitted_masks[i]
        image = splitted_images[i]
        R = splitted_extrinsics[i][:3, :3]

        H, W = points.shape[:2]
        depth = points[..., 2]
        mask_cleaned = mask & (depth > 0)

        view_has_normal = has_normals and splitted_normals[i] is not None
        if threshold < float('inf'):
            edge_mask = utils3d.np.depth_map_edge(depth, rtol=threshold)
            if view_has_normal:
                edge_mask = edge_mask & utils3d.np.normal_map_edge(splitted_normals[i], tol=5)
            mask_cleaned = mask_cleaned & ~edge_mask

        uv = utils3d.np.uv_map((H, W))
        if view_has_normal:
            faces_i, vert_i, colors_i, _, norm_i = utils3d.np.build_mesh_from_map(
                points,
                image.astype(np.float32) / 255.0,
                uv,
                splitted_normals[i],
                mask=mask_cleaned,
                tri=True
            )
        else:
            faces_i, vert_i, colors_i, _ = utils3d.np.build_mesh_from_map(
                points,
                image.astype(np.float32) / 255.0,
                uv,
                mask=mask_cleaned,
                tri=True
            )
            norm_i = None

        if len(vert_i) == 0 or len(faces_i) == 0:
            continue

        # Rotate camera coordinates to world coordinates: P_world = P_cam @ R
        vert_world_i = vert_i @ R
        vertices_list.append(vert_world_i)
        faces_list.append(faces_i + total_vertices)
        colors_list.append(colors_i)

        if norm_i is not None:
            norm_world_i = norm_i @ R
            normals_list.append(norm_world_i)

        total_vertices += len(vert_world_i)

    if total_vertices == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3)), None

    all_vertices = np.concatenate(vertices_list, axis=0)
    all_faces = np.concatenate(faces_list, axis=0)
    all_colors = np.concatenate(colors_list, axis=0)
    all_normals = np.concatenate(normals_list, axis=0) if len(normals_list) > 0 else None

    # OpenGL coordinate conventions:
    # world coordinate system: x right, y up, z backward.
    all_vertices = all_vertices * [1, -1, -1]
    if all_normals is not None:
        all_normals = all_normals * [1, -1, -1]

    return all_vertices, all_faces, all_colors, all_normals
         
