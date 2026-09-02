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


def fit_ground_plane(
    splitted_points: List[np.ndarray],
    splitted_normals: Optional[List[np.ndarray]],
    splitted_masks: List[np.ndarray],
    splitted_extrinsics: List[np.ndarray],
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Fits the ground floor plane (n . P + d = 0) using upward-facing surface normals
    in world coordinates (Z is UP, floor is at negative Z).
    Returns (n_floor, d) where n_floor is approximately [0, 0, 1] and d > 0.
    """
    if splitted_normals is None or not any(n is not None for n in splitted_normals):
        return None

    floor_pts_list = []
    for i in range(len(splitted_points)):
        if splitted_normals[i] is None:
            continue
        R = splitted_extrinsics[i][:3, :3]
        pts = splitted_points[i]
        norm = splitted_normals[i]
        mask = splitted_masks[i] & (pts[..., 2] > 0)

        # Rotate to world coordinates: P_world = P_cam @ R
        pts_w = pts @ R
        norm_w = norm @ R

        # Candidate floor: below camera (Z < -0.3) and normal pointing UP (norm_w[..., 2] > 0.80)
        is_floor_cand = mask & (pts_w[..., 2] < -0.3) & (norm_w[..., 2] > 0.80)
        if np.any(is_floor_cand):
            floor_pts_list.append(pts_w[is_floor_cand])

    if len(floor_pts_list) == 0:
        return None

    all_cand_pts = np.concatenate(floor_pts_list, axis=0)
    if len(all_cand_pts) < 100:
        return None

    # Subsample if too many points for speed
    if len(all_cand_pts) > 20000:
        all_cand_pts = all_cand_pts[::max(1, len(all_cand_pts) // 20000)]

    # Robust median height
    z_vals = all_cand_pts[:, 2]
    z_median = float(np.median(z_vals))
    inlier_mask = np.abs(z_vals - z_median) < 0.20
    inlier_pts = all_cand_pts[inlier_mask]

    if len(inlier_pts) < 50:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32), float(-z_median)

    # Fit plane n . P + d = 0 using SVD on inliers
    centroid = np.mean(inlier_pts, axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid)
    normal = vh[2]

    # Ensure normal points UP (positive Z)
    if normal[2] < 0:
        normal = -normal

    # If normal is too tilted (> 20 deg from vertical), force horizontal [0, 0, 1]
    if normal[2] < 0.94:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        d = float(-centroid[2])
    else:
        normal = (normal / np.linalg.norm(normal)).astype(np.float32)
        d = float(-np.dot(normal, centroid))

    return normal, d


def merge_panorama_depth_raycast(
    width: int, 
    height: int, 
    distance_maps: List[np.ndarray], 
    pred_masks: List[np.ndarray], 
    extrinsics: List[np.ndarray], 
    intrinsics: List[np.ndarray],
    normal_maps: Optional[List[np.ndarray]] = None,
    ground_plane: Optional[Tuple[np.ndarray, float]] = None,
    power: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Direct Spherical Cosine-Weighted Raycasting Fusion for 360 Panorama.
    Preserves exact metric scale, eliminates polar distortion, and optionally snaps floor rays.
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

    # Snap floor rays if ground plane is provided
    if ground_plane is not None and panorama_normal is not None:
        n_floor, d_floor = ground_plane
        denom = np.sum(spherical_directions * n_floor, axis=-1)
        is_floor_ray = (spherical_directions[..., 2] < -0.15) & (denom < -0.05) & (panorama_normal[..., 2] > 0.70)
        r_floor = -d_floor / np.minimum(denom, -0.05)
        depth_diff = np.abs(panorama_depth - r_floor)
        snap_mask = is_floor_ray & (depth_diff < 0.35)
        panorama_depth[snap_mask] = r_floor[snap_mask]
        panorama_normal[snap_mask] = n_floor

    return panorama_depth, panorama_mask, panorama_normal


def build_panorama_mesh_multiview(
    splitted_images: List[np.ndarray],
    splitted_points: List[np.ndarray],
    splitted_normals: Optional[List[np.ndarray]],
    splitted_masks: List[np.ndarray],
    splitted_extrinsics: List[np.ndarray],
    threshold: float = 0.03,
    planar_floor: bool = True,
    ground_plane: Optional[Tuple[np.ndarray, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Tuple[np.ndarray, float]]]:
    """
    Direct Multi-View 3D Mesh Reconstruction for 360 Panorama.
    Builds clean perspective meshes from each icosahedral view, eliminates grazing-angle splatter,
    snaps floor points to a flat ground plane, and rotates all geometry into global world space.
    """
    has_normals = splitted_normals is not None and any(n is not None for n in splitted_normals)

    # Fit ground plane if requested and not provided
    if planar_floor and ground_plane is None and has_normals:
        ground_plane = fit_ground_plane(splitted_points, splitted_normals, splitted_masks, splitted_extrinsics)

    vertices_list = []
    faces_list = []
    colors_list = []
    normals_list = []
    total_vertices = 0

    for i in range(len(splitted_images)):
        points = splitted_points[i].copy()
        mask = splitted_masks[i].copy()
        image = splitted_images[i]
        R = splitted_extrinsics[i][:3, :3]
        H, W = points.shape[:2]

        view_has_normal = has_normals and splitted_normals[i] is not None
        normal = splitted_normals[i].copy() if view_has_normal else None

        valid_snap = np.zeros((H, W), dtype=bool)

        # Ground plane snapping
        if ground_plane is not None and view_has_normal:
            n_floor, d_floor = ground_plane
            pts_w = points @ R
            norm_w = normal @ R

            dist_to_plane = np.abs(np.sum(pts_w * n_floor, axis=-1) + d_floor)
            is_floor = (pts_w[..., 2] < -0.3) & (norm_w[..., 2] > 0.75) & (dist_to_plane < 0.25)

            ray_w = pts_w / np.maximum(np.linalg.norm(pts_w, axis=-1, keepdims=True), 1e-6)
            denom = np.sum(ray_w * n_floor, axis=-1)
            valid_snap = is_floor & (denom < -0.05)

            if np.any(valid_snap):
                r_exact = -d_floor / np.minimum(denom, -0.05)
                pts_w[valid_snap] = ray_w[valid_snap] * r_exact[valid_snap, None]
                points[valid_snap] = pts_w[valid_snap] @ R.T
                normal[valid_snap] = n_floor @ R.T

        depth = points[..., 2]
        mask_cleaned = mask & (depth > 0)

        # Edge discontinuity pruning
        if threshold < float('inf'):
            edge_mask = utils3d.np.depth_map_edge(depth, rtol=threshold)
            if view_has_normal:
                edge_mask = edge_mask & utils3d.np.normal_map_edge(normal, tol=5)
            # Do not cut edges on flat floor
            edge_mask = edge_mask & ~valid_snap
            mask_cleaned = mask_cleaned & ~edge_mask

        # Grazing angle filtering: remove points where viewing ray is nearly parallel to surface
        if view_has_normal:
            v_cam = points / np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), 1e-6)
            cos_inc = -np.sum(v_cam * normal, axis=-1)
            grazing_mask = (cos_inc < 0.20) & (~valid_snap)
            mask_cleaned = mask_cleaned & ~grazing_mask

        uv = utils3d.np.uv_map((H, W))
        if view_has_normal:
            faces_i, vert_i, colors_i, uvs_i, norm_i = utils3d.np.build_mesh_from_map(
                points,
                image.astype(np.float32) / 255.0,
                uv,
                normal,
                mask=mask_cleaned,
                tri=True
            )
        else:
            faces_i, vert_i, colors_i, uvs_i = utils3d.np.build_mesh_from_map(
                points,
                image.astype(np.float32) / 255.0,
                uv,
                mask=mask_cleaned,
                tri=True
            )
            norm_i = None

        if len(vert_i) == 0 or len(faces_i) == 0:
            continue

        # Prune stretched / flying triangles with oversized edge-to-distance ratio
        v0 = vert_i[faces_i[:, 0]]
        v1 = vert_i[faces_i[:, 1]]
        v2 = vert_i[faces_i[:, 2]]
        d01 = np.linalg.norm(v0 - v1, axis=-1)
        d12 = np.linalg.norm(v1 - v2, axis=-1)
        d20 = np.linalg.norm(v2 - v0, axis=-1)
        max_edge = np.maximum(np.maximum(d01, d12), d20)
        min_dist = np.minimum(np.minimum(np.linalg.norm(v0, axis=-1), np.linalg.norm(v1, axis=-1)), np.linalg.norm(v2, axis=-1))

        valid_tri = max_edge < (0.25 * min_dist + 0.15)
        faces_i = faces_i[valid_tri]
        if len(faces_i) == 0:
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
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3)), None, ground_plane

    all_vertices = np.concatenate(vertices_list, axis=0)
    all_faces = np.concatenate(faces_list, axis=0)
    all_colors = np.concatenate(colors_list, axis=0)
    all_normals = np.concatenate(normals_list, axis=0) if len(normals_list) > 0 else None
    if all_normals is not None and len(all_normals) != len(all_vertices):
        all_normals = None

    # OpenGL coordinate conventions:
    # world coordinate system: x right, y up, z backward.
    all_vertices = all_vertices * [1, -1, -1]
    if all_normals is not None:
        all_normals = all_normals * [1, -1, -1]

    return all_vertices, all_faces, all_colors, all_normals, ground_plane


def align_view_scales(
    splitted_points: List[np.ndarray],
    splitted_depth: List[np.ndarray],
    splitted_distance: List[np.ndarray],
    splitted_normals: Optional[List[np.ndarray]],
    splitted_masks: List[np.ndarray],
    splitted_extrinsics: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Normalizes monocular metric scale across all 12 perspective views.
    Anchors views seeing the floor to a global consensus ground plane height,
    then propagates scale consistency to the remaining views via overlapping ray consistency.
    """
    num_views = len(splitted_points)
    scales = np.ones(num_views, dtype=np.float32)

    # 1. Floor-based anchor for views seeing the floor
    floor_heights = {}
    for i in range(num_views):
        if splitted_normals is None or splitted_normals[i] is None:
            continue
        R = splitted_extrinsics[i][:3, :3]
        pts_w = splitted_points[i] @ R
        norm_w = splitted_normals[i] @ R
        mask = splitted_masks[i] & (splitted_points[i][..., 2] > 0)
        is_floor = mask & (pts_w[..., 2] < -0.3) & (norm_w[..., 2] > 0.80)
        if np.count_nonzero(is_floor) > 100:
            floor_z = pts_w[is_floor, 2]
            z_med = float(np.median(floor_z))
            floor_heights[i] = -z_med

    if len(floor_heights) > 0:
        h_consensus = float(np.median(list(floor_heights.values())))
        for i, h_i in floor_heights.items():
            if h_i > 0.1:
                scales[i] = h_consensus / h_i

    # 2. Propagate scale to non-floor views via overlap ray consistency
    for i in range(num_views):
        if i in floor_heights:
            continue
        ratios = []
        opt_i = splitted_extrinsics[i][2, :3]
        for j in floor_heights.keys():
            opt_j = splitted_extrinsics[j][2, :3]
            if np.dot(opt_i, opt_j) > 0.35:
                d_i = splitted_distance[i]
                d_j = splitted_distance[j] * scales[j]
                m_i = splitted_masks[i]
                m_j = splitted_masks[j]
                if np.any(m_i) and np.any(m_j):
                    med_i = float(np.median(d_i[m_i]))
                    med_j = float(np.median(d_j[m_j]))
                    if med_i > 0.1 and med_j > 0.1:
                        ratios.append(med_j / med_i)
        if len(ratios) > 0:
            scales[i] = float(np.median(ratios))

    scaled_points = [splitted_points[i] * scales[i] for i in range(num_views)]
    scaled_depth = [splitted_depth[i] * scales[i] for i in range(num_views)]
    scaled_dist = [splitted_distance[i] * scales[i] for i in range(num_views)]

    return scaled_points, scaled_depth, scaled_dist


def build_panorama_mesh_equirectangular(
    image: np.ndarray,
    panorama_depth: np.ndarray,
    panorama_mask: np.ndarray,
    panorama_normal: Optional[np.ndarray] = None,
    threshold: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Builds a continuous, seamless 360-degree room mesh from an equirectangular depth map.
    Handles 360-degree horizontal wrapping and filters depth edge jumps.
    Returns (vertices, faces, vertex_colors, vertex_uvs, vertex_normals).
    """
    H, W = panorama_depth.shape[:2]
    uv = utils3d.np.uv_map((H, W))
    spherical_dirs = spherical_uv_to_directions(uv)
    points = panorama_depth[..., None] * spherical_dirs

    # Clean mask: must have positive depth and be valid
    valid = panorama_mask & (panorama_depth > 0.01)

    # Filter depth edges horizontally and vertically
    padded_depth = np.pad(panorama_depth, ((0, 0), (0, 1)), mode='wrap')
    p_min = np.maximum(np.minimum(padded_depth[:, :-1], padded_depth[:, 1:]), 1e-3)
    edge_x = (np.abs(padded_depth[:, :-1] - padded_depth[:, 1:]) / p_min) > threshold

    p_y_min = np.maximum(np.minimum(panorama_depth[:-1, :], panorama_depth[1:, :]), 1e-3)
    edge_y_core = (np.abs(panorama_depth[:-1, :] - panorama_depth[1:, :]) / p_y_min) > threshold
    edge_y = np.pad(edge_y_core, ((0, 1), (0, 0)), mode='edge')

    # Duplicate column 0 as column W to wrap horizontally with continuous UV coordinates:
    pts_wrapped = np.concatenate([points, points[:, :1]], axis=1) # (H, W + 1, 3)
    u_coords = np.linspace(0.0, 1.0, W + 1, dtype=np.float32)
    v_coords = np.linspace(0.0, 1.0, H, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)
    uvs_wrapped = np.stack([uu, vv], axis=-1) # (H, W + 1, 2)

    img_float = image.astype(np.float32) / 255.0
    colors_wrapped = np.concatenate([img_float, img_float[:, :1]], axis=1) # (H, W + 1, 3)

    if panorama_normal is not None:
        norm_wrapped = np.concatenate([panorama_normal, panorama_normal[:, :1]], axis=1)
    else:
        norm_wrapped = None

    valid_wrapped = np.concatenate([valid, valid[:, :1]], axis=1)

    # Flatten arrays
    W_ext = W + 1
    total_verts = H * W_ext
    vert_indices = np.arange(total_verts).reshape(H, W_ext)

    # Quad corners:
    tl = vert_indices[:-1, :-1]
    tr = vert_indices[:-1, 1:]
    bl = vert_indices[1:, :-1]
    br = vert_indices[1:, 1:]

    v_tl = valid_wrapped[:-1, :-1]
    v_tr = valid_wrapped[:-1, 1:]
    v_bl = valid_wrapped[1:, :-1]
    v_br = valid_wrapped[1:, 1:]

    e_x_top = edge_x[:-1, :]
    e_x_bot = edge_x[1:, :]
    e_y_left = edge_y[:-1, :]
    e_y_right = np.roll(edge_y[:-1, :], -1, axis=1)

    # Triangle 1: (TL, BL, TR)
    tri1_valid = v_tl & v_bl & v_tr & ~e_x_top & ~e_y_left
    # Triangle 2: (TR, BL, BR)
    tri2_valid = v_tr & v_bl & v_br & ~e_x_bot & ~e_y_right

    f1 = np.stack([tl[tri1_valid], bl[tri1_valid], tr[tri1_valid]], axis=-1)
    f2 = np.stack([tr[tri2_valid], bl[tri2_valid], br[tri2_valid]], axis=-1)
    faces = np.concatenate([f1, f2], axis=0)

    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3)), np.zeros((0, 2)), None

    # Compact unreferenced vertices
    used_indices, inverse_indices = np.unique(faces, return_inverse=True)
    faces = inverse_indices.reshape(faces.shape).astype(np.int32)
    vertices = vertices[used_indices]
    vertex_colors = vertex_colors[used_indices]
    vertex_uvs = vertex_uvs[used_indices]
    if vertex_normals is not None:
        vertex_normals = vertex_normals[used_indices]

    # Follow OpenGL conventions: x right, y up, z backward
    # Texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top
    vertices = vertices * [1, -1, -1]
    vertex_uvs = vertex_uvs * [1, -1] + [0, 1]
    if vertex_normals is not None:
        vertex_normals = vertex_normals * [1, -1, -1]

    return vertices, faces, vertex_colors, vertex_uvs, vertex_normals
         
