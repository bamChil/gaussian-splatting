#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 0.5, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image, output_T = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image, output_T = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> render with output_T <<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
    # compute colors using T
    def tensor_to_heatmap(T):
        # Step 1: Replace -1 with 0
        T = torch.where(T == -1, torch.zeros_like(T), T)
        
        # Step 2: Initialize the color tensor (N, 3)
        device = T.device
        colors = torch.zeros((T.shape[0], 3), device=device)
        
        # Case 1: T >= 0.75 (Red to Orange)
        mask_high = (T >= 0.75)
        t_high = (T[mask_high] - 0.75) * 4  # Normalize to [0,1] for 0.75-1.0
        colors[mask_high, 0] = 1            # R: 1 (constant)
        colors[mask_high, 1] = 0.5 * t_high # G: 0 -> 0.5
        colors[mask_high, 2] = 0            # B: 0
        
        # Case 2: 0.5 <= T < 0.75 (Orange to Yellow)
        mask_mid_high = (T >= 0.5) & (T < 0.75)
        t_mid_high = (T[mask_mid_high] - 0.5) * 4  # Normalize to [0,1] for 0.5-0.75
        colors[mask_mid_high, 0] = 1                # R: 1 (constant)
        colors[mask_mid_high, 1] = 0.5 + 0.5 * t_mid_high  # G: 0.5 -> 1
        colors[mask_mid_high, 2] = 0.5 * t_mid_high        # B: 0 -> 0.25
        
        # Case 3: 0.25 <= T < 0.5 (Yellow to Teal)
        mask_mid_low = (T >= 0.25) & (T < 0.5)
        t_mid_low = (T[mask_mid_low] - 0.25) * 4  # Normalize to [0,1] for 0.25-0.5
        colors[mask_mid_low, 0] = 1 - t_mid_low          # R: 1 -> 0
        colors[mask_mid_low, 1] = 1 - 0.5 * t_mid_low    # G: 1 -> 0.5
        colors[mask_mid_low, 2] = 0.25 + 0.25 * t_mid_low # B: 0.25 -> 0.5
        
        # Case 4: T < 0.25 (Teal to Dark Blue)
        mask_low = (T < 0.25)
        t_low = T[mask_low] * 4  # Normalize to [0,1] for 0-0.25
        colors[mask_low, 0] = 0                  # R: 0
        colors[mask_low, 1] = 0.5 - 0.5 * t_low  # G: 0.5 -> 0
        colors[mask_low, 2] = 0.5                # B: 0.5 (constant)
        
        return colors
    
    colors_T = tensor_to_heatmap(output_T)
    if separate_sh:
        rendered_T, radii, depth_image, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_T,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_T, radii, depth_image, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_T,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> render with output_T <<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
         
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        "output_T": output_T,
        "rendered_T": rendered_T
        }
    
    return out
