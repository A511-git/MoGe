import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import itertools
import json
import warnings

import click

         
@click.command(help='Inference script for panorama images')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True, help='Input image or folder path. "jpg" and "png" are supported.')
@click.option('--output', '-o', 'output_path', type=click.Path(), default='./output', help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default=None, help='Pretrained model name or path. Optional for v1/v2 and required for v3.')
@click.option('--version', 'model_version', type=click.Choice(['v1', 'v2', 'v3']), default='v3', help='Model version. Defaults to "v3"')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Use fp16 precision for faster inference.')
@click.option('--resize', 'resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level of inference. The higher, the better but slower. Defaults to 9. Note that it is irrelevant to the output resolution.')
@click.option('--num_tokens', type=int, default=None, help='Number of tokens used for inference. Overrides resolution_level if provided.')
@click.option('--refine_steps', type=click.IntRange(min=0), default=3, help='Number of sparse refinement steps for v3. Defaults to 3.')
@click.option('--split_resolution', type=int, default=512, help='Resolution for each splitted perspective view. Defaults to 512.')
@click.option('--threshold', type=float, default=0.03, help='Threshold for removing edges. Defaults to 0.03. Smaller value removes more edges. "inf" means no thresholding.')
@click.option('--batch_size', type=int, default=4, help='Batch size for inference. Defaults to 4.')
@click.option('--merge_method', type=click.Choice(['raycast', 'poisson']), default='raycast', help='Method for merging panorama 2D maps: "raycast" (default, fast, metric-accurate) or "poisson" (legacy gradient integration).')
@click.option('--planar_floor/--no_planar_floor', 'planar_floor', default=True, help='Fit a planar ground floor and snap floor points to prevent ground splatter. Defaults to True.')
@click.option('--splitted', 'save_splitted', is_flag=True, help='Whether to save the splitted perspective views and all associated data (RGB, depth, distance, points, mask, normal, cameras). Defaults to False.')
@click.option('--use_cache', is_flag=True, help='Use cached splitted views from <output>/splitted to skip neural inference and continue directly with 2D merging and 3D mesh building.')
@click.option('--maps', 'save_maps_', is_flag=True, help='Whether to save the output maps and fov(image, depth, mask, points, normal, fov).')
@click.option('--glb', 'save_glb_', is_flag=True, help='Whether to save the output as a .glb file. The color will be saved as a texture.')
@click.option('--ply', 'save_ply_', is_flag=True, help='Whether to save the output as a .ply file. The color will be saved as vertex colors.')
@click.option('--show', 'show', is_flag=True, help='Whether show the output in a window. Note that this requires pyglet<2 installed as required by trimesh.')
def main(
    input_path: str,
    output_path: str,
    pretrained_model_name_or_path: str,
    model_version: str,
    device_name: str,
    use_fp16: bool,
    resize_to: int,
    resolution_level: int,
    num_tokens: int,
    refine_steps: int,
    split_resolution: int,
    threshold: float,
    batch_size: int,
    merge_method: str,
    planar_floor: bool,
    save_splitted: bool,
    use_cache: bool,
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    show: bool,
):  
    # Lazy import
    import cv2
    import numpy as np
    from numpy import ndarray
    import torch
    from PIL import Image
    from tqdm import tqdm, trange
    import trimesh
    import trimesh.visual
    from scipy.sparse import csr_array, hstack, vstack
    from scipy.ndimage import convolve
    from scipy.sparse.linalg import lsmr

    try:
        import utils3d_moge as utils3d
    except ImportError:
        import utils3d
    from moge.model import import_model_class_by_version
    from moge.utils.io import save_glb, save_ply
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.panorama import (
        spherical_uv_to_directions, 
        get_panorama_cameras, 
        split_panorama_image, 
        merge_panorama_depth,
        merge_panorama_depth_raycast,
        build_panorama_mesh_multiview,
        fit_ground_plane
    )

    
    device = torch.device(device_name)

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    # Write outputs
    if not any([save_maps_, save_glb_, save_ply_]):
        warnings.warn('No output format specified. Defaults to saving all. Please use "--maps", "--glb", or "--ply" to specify the output.')
        save_maps_ = save_glb_ = save_ply_ = True

    if pretrained_model_name_or_path is None:
        default_pretrained_models = {
            'v1': 'Ruicheng/moge-vitl',
            'v2': 'Ruicheng/moge-2-vitl-normal',
            'v3': 'Ruicheng/moge-3-vitl',
        }
        pretrained_model_name_or_path = default_pretrained_models[model_version]

    model = None
    def load_model():
        nonlocal model
        if model is None:
            model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()
            if use_fp16 and model_version != 'v3':
                model.half()
        return model

    for image_path in (pbar := tqdm(image_paths, desc='Total images', disable=len(image_paths) <= 1)):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if resize_to is not None:
            height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        
        save_path = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
        save_path.mkdir(exist_ok=True, parents=True)

        splitted_extrinsics, splitted_intriniscs = get_panorama_cameras()
        splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, split_resolution)

        splitted_save_path = save_path / 'splitted'
        has_cache = use_cache and splitted_save_path.exists() and (splitted_save_path / '00.jpg').exists() and (splitted_save_path / '00_points.exr').exists()

        if has_cache:
            print('Loading cached splitted views from disk...') if pbar.disable else pbar.set_postfix_str('Loading cache')
            splitted_images = []
            splitted_distance_maps, splitted_masks = [], []
            splitted_depth_maps, splitted_points_maps, splitted_normal_maps = [], [], []

            for i in range(len(splitted_extrinsics)):
                img_file = splitted_save_path / f'{i:02d}.jpg'
                mask_file = splitted_save_path / f'{i:02d}_mask.png'
                depth_file = splitted_save_path / f'{i:02d}_depth.exr'
                dist_file = splitted_save_path / f'{i:02d}_distance.exr'
                pts_file = splitted_save_path / f'{i:02d}_points.exr'
                norm_exr = splitted_save_path / f'{i:02d}_normal.exr'

                img_bgr = cv2.imread(str(img_file))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                splitted_images.append(img_rgb)

                mask = (cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 127)
                splitted_masks.append(mask)

                depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                splitted_depth_maps.append(depth)

                if dist_file.exists():
                    dist = cv2.imread(str(dist_file), cv2.IMREAD_UNCHANGED)
                else:
                    dist = depth
                splitted_distance_maps.append(dist)

                pts_raw = cv2.imread(str(pts_file), cv2.IMREAD_UNCHANGED)
                if pts_raw is not None and pts_raw.ndim == 3 and pts_raw.shape[-1] >= 3:
                    pts = cv2.cvtColor(pts_raw[..., :3], cv2.COLOR_BGR2RGB)
                else:
                    pts = pts_raw
                splitted_points_maps.append(pts)

                if norm_exr.exists():
                    norm_raw = cv2.imread(str(norm_exr), cv2.IMREAD_UNCHANGED)
                    norm = cv2.cvtColor(norm_raw[..., :3], cv2.COLOR_BGR2RGB)
                    splitted_normal_maps.append(norm)
                elif pts is not None:
                    norm, _ = utils3d.np.point_map_to_normal_map(pts, mask=mask)
                    splitted_normal_maps.append(norm)
        else:
            # Infer each view 
            print('Inferring...') if pbar.disable else pbar.set_postfix_str(f'Inferring')
            model_instance = load_model()

            splitted_distance_maps, splitted_masks = [], []
            splitted_depth_maps, splitted_points_maps, splitted_normal_maps = [], [], []

            for i in trange(0, len(splitted_images), batch_size, desc='Inferring splitted views', disable=len(splitted_images) <= batch_size, leave=False):
                image_tensor = torch.tensor(np.stack(splitted_images[i:i + batch_size]) / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                fov_x, fov_y = np.rad2deg(utils3d.np.intrinsics_to_fov(np.array(splitted_intriniscs[i:i + batch_size])))
                fov_x = torch.tensor(fov_x, dtype=torch.float32, device=device)

                infer_kwargs = {
                    'fov_x': fov_x,
                    'resolution_level': resolution_level,
                    'use_fp16': use_fp16,
                    'apply_mask': False,
                }
                if num_tokens is not None:
                    infer_kwargs['num_tokens'] = num_tokens
                if model_version == 'v3':
                    infer_kwargs['refine_steps'] = refine_steps

                output = model_instance.infer(image_tensor, **infer_kwargs)
                distance_map, mask = output['points'].norm(dim=-1).cpu().numpy(), output['mask'].cpu().numpy()
                splitted_distance_maps.extend(list(distance_map))
                splitted_masks.extend(list(mask))
                splitted_depth_maps.extend(list(output['depth'].cpu().numpy()))
                splitted_points_maps.extend(list(output['points'].cpu().numpy()))
                if 'normal' in output and output['normal'] is not None:
                    splitted_normal_maps.extend(list(output['normal'].cpu().numpy()))

            # Save splitted
            if save_splitted:
                splitted_save_path.mkdir(exist_ok=True, parents=True)
                cameras_meta = []
                for i in range(len(splitted_images)):
                    # RGB image
                    cv2.imwrite(str(splitted_save_path / f'{i:02d}.jpg'), cv2.cvtColor(splitted_images[i], cv2.COLOR_RGB2BGR))
                    # Validity mask
                    cv2.imwrite(str(splitted_save_path / f'{i:02d}_mask.png'), (splitted_masks[i] * 255).astype(np.uint8))
                    # Depth (raw float32 and visualization)
                    cv2.imwrite(str(splitted_save_path / f'{i:02d}_depth.exr'), splitted_depth_maps[i], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    cv2.imwrite(str(splitted_save_path / f'{i:02d}_depth_vis.png'), cv2.cvtColor(colorize_depth(splitted_depth_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))
                    # Distance (raw float32 and visualization)
                    cv2.imwrite(str(splitted_save_path / f'{i:02d}_distance.exr'), splitted_distance_maps[i], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    cv2.imwrite(str(splitted_save_path / f'{i:02d}_distance_vis.png'), cv2.cvtColor(colorize_depth(splitted_distance_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))
                    # Point map (swapped to BGR for OpenCV EXR writer)
                    cv2.imwrite(str(splitted_save_path / f'{i:02d}_points.exr'), cv2.cvtColor(splitted_points_maps[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    # Normal (if available)
                    if len(splitted_normal_maps) > i and splitted_normal_maps[i] is not None:
                        cv2.imwrite(str(splitted_save_path / f'{i:02d}_normal.png'), cv2.cvtColor(colorize_normal(splitted_normal_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(splitted_save_path / f'{i:02d}_normal.exr'), cv2.cvtColor(splitted_normal_maps[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    # Camera parameters
                    fov_xi, fov_yi = np.rad2deg(utils3d.np.intrinsics_to_fov(splitted_intriniscs[i]))
                    cam_info = {
                        'index': i,
                        'image': f'{i:02d}.jpg',
                        'fov_x': round(float(fov_xi), 2),
                        'fov_y': round(float(fov_yi), 2),
                        'intrinsics': splitted_intriniscs[i].tolist(),
                        'extrinsics': splitted_extrinsics[i].tolist(),
                    }
                    cameras_meta.append(cam_info)
                    with open(splitted_save_path / f'{i:02d}_camera.json', 'w') as f:
                        json.dump(cam_info, f, indent=2)

                with open(splitted_save_path / 'cameras.json', 'w') as f:
                    json.dump({'views': cameras_meta}, f, indent=2)

        # Fit ground plane if planar_floor is enabled
        ground_plane = None
        if planar_floor and len(splitted_normal_maps) > 0:
            ground_plane = fit_ground_plane(
                splitted_points_maps,
                splitted_normal_maps,
                splitted_masks,
                splitted_extrinsics
            )

        # Build 3D mesh directly in world space
        vertices, faces, vertex_colors, vertex_normals = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3)), None
        if save_glb_ or save_ply_ or show:
            print('Building 3D multi-view mesh...') if pbar.disable else pbar.set_postfix_str(f'Building 3D mesh')
            vertices, faces, vertex_colors, vertex_normals, _ = build_panorama_mesh_multiview(
                splitted_images,
                splitted_points_maps,
                splitted_normal_maps if len(splitted_normal_maps) > 0 else None,
                splitted_masks,
                splitted_extrinsics,
                threshold=threshold,
                planar_floor=planar_floor,
                ground_plane=ground_plane
            )

        # Merge 2D maps
        print('Merging 2D maps...') if pbar.disable else pbar.set_postfix_str(f'Merging 2D maps')
        if merge_method == 'raycast':
            panorama_depth, panorama_mask, panorama_normal = merge_panorama_depth_raycast(
                width, height,
                splitted_distance_maps, splitted_masks,
                splitted_extrinsics, splitted_intriniscs,
                normal_maps=splitted_normal_maps if len(splitted_normal_maps) > 0 else None,
                ground_plane=ground_plane
            )
            points = panorama_depth[:, :, None] * spherical_uv_to_directions(utils3d.np.uv_map(height, width))
            if panorama_normal is None:
                panorama_normal, _ = utils3d.np.point_map_to_normal_map(points, panorama_mask)
        else:
            merging_width, merging_height = min(1920, width), min(960, height)
            panorama_depth, panorama_mask = merge_panorama_depth(merging_width, merging_height, splitted_distance_maps, splitted_masks, splitted_extrinsics, splitted_intriniscs)
            panorama_depth = panorama_depth.astype(np.float32)
            panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
            panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0
            points = panorama_depth[:, :, None] * spherical_uv_to_directions(utils3d.np.uv_map(height, width))
            panorama_normal, _ = utils3d.np.point_map_to_normal_map(points, panorama_mask)

        # Write outputs
        print('Writing outputs...') if pbar.disable else pbar.set_postfix_str(f'Writing outputs')
        if save_maps_:
            cv2.imwrite(str(save_path / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(panorama_depth, mask=panorama_mask), cv2.COLOR_RGB2BGR))
            if panorama_normal is not None:
                cv2.imwrite(str(save_path / 'normal_vis.png'), cv2.cvtColor(colorize_normal(panorama_normal, mask=panorama_mask), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth.exr'), panorama_depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(save_path / 'points.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(save_path / 'mask.png'), (panorama_mask * 255).astype(np.uint8))

        if (save_glb_ or save_ply_ or show) and len(vertices) > 0:
            if save_glb_:
                save_glb(save_path / 'mesh.glb', vertices, faces, vertex_colors=vertex_colors, vertex_normals=vertex_normals)

            if save_ply_:
                save_ply(save_path / 'mesh.ply', vertices, faces, vertex_colors=vertex_colors, vertex_normals=vertex_normals)

            if show:
                trimesh.Trimesh(
                    vertices=vertices,
                    vertex_colors=vertex_colors,
                    vertex_normals=vertex_normals,
                    faces=faces, 
                    process=False
                ).show()
        elif (save_glb_ or save_ply_ or show) and len(vertices) == 0:
            warnings.warn('Reconstructed mesh has 0 vertices; skipping mesh export.')


if __name__ == '__main__':
    main()