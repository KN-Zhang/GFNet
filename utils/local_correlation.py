import torch
import torch.nn.functional as F

def local_correlation(
    featuremap_size,
    feature0,
    feature1,
    local_radius,
    num_grid,
    padding_mode="zeros",
    flow = None,
    im_A_coords = None,
    sample_mode = "bilinear",
    grid_based_correlation=False,
    num_level=1,
):
    r = local_radius
    K = (2*r+1)**2 * num_level
    B, c, h, w = featuremap_size
    corr = torch.empty((B,K,num_grid,num_grid), device = feature0.device, dtype=feature0.dtype)
    if flow is None:
        # If flow is None, assume feature0 and feature1 are aligned
        coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=feature0.device),
                    torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=feature0.device),
                ))
        coords = torch.stack((coords[1], coords[0]), dim=-1)[
            None
        ].expand(B, h, w, 2)
    else:
        coords = flow.permute(0,2,3,1) # If using flow, sample around flow target.
    if grid_based_correlation:
        local_window = torch.meshgrid(
                    (
                        torch.linspace(-2*local_radius/num_grid, 2*local_radius/num_grid, 2*r+1, device=feature0.device),
                        torch.linspace(-2*local_radius/num_grid, 2*local_radius/num_grid, 2*r+1, device=feature0.device),
                    ),
                    indexing = 'ij'
                    )
    else:
        local_window = torch.meshgrid(
                    (
                        torch.linspace(-2*local_radius/h, 2*local_radius/h, 2*r+1, device=feature0.device),
                        torch.linspace(-2*local_radius/w, 2*local_radius/w, 2*r+1, device=feature0.device),
                    ),
                    indexing = 'ij'
                    )        
    local_window = torch.stack((local_window[1], local_window[0]), dim=-1)[
            None
        ].expand(1, 2*r+1, 2*r+1, 2).reshape(1, (2*r+1)**2, 2)
    if num_level == 1:
        for _ in range(B):
            with torch.no_grad():
                local_window_coords = (coords[_,:,:,None]+local_window[:,None,None]).reshape(1,num_grid, num_grid*(2*r+1)**2,2)
                window_feature = F.grid_sample(
                    feature1[_:_+1], local_window_coords, padding_mode=padding_mode, align_corners=False, mode = sample_mode, #
                )
                window_feature = window_feature.reshape(c,num_grid,num_grid,(2*r+1)**2)
            corr[_] = (feature0[_,...,None]/(c**.5)*window_feature).sum(dim=0).permute(2,0,1).contiguous() ## cosine similarity
    else:
        for level in range(num_level):
            for _ in range(B):
                with torch.no_grad():
                    local_window_coords = (coords[_,:,:,None]+local_window[:,None,None]).reshape(1,num_grid, num_grid*(2*r+1)**2,2)
                    window_feature = F.grid_sample(
                        feature1[_:_+1], local_window_coords, padding_mode=padding_mode, align_corners=False, mode = sample_mode, #
                    )
                    window_feature = window_feature.reshape(c,num_grid,num_grid,(2*r+1)**2)
                corr[_, (2*r+1)**2*level: (2*r+1)**2*(level+1)] = (feature0[_,...,None]/(c**.5)*window_feature).sum(dim=0).permute(2,0,1).contiguous() ## cosine similarity                    
            feature1 = F.avg_pool2d(feature1, kernel_size=2, stride=2)
    return corr
