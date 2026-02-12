import torch
from torch import Tensor

def trilinear_splatting(
    coord: Tensor,           # (N, 3), differentiable
    feat: Tensor,           # (N, C), differentiable
    batch: Tensor,          # (N,), int (will be detached)
    grid_size: float,
    eps: float = 1e-8,
    normalize_feat: bool = True,
) -> tuple[
    Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor,
    Tensor, Tensor,
]:
    """
    Differentiable trilinear splatting from points to a voxel grid.

    Each point contributes its feature to the 8 corners of its enclosing voxel,
    weighted by trilinear interpolation weights. Features are aggregated per voxel
    via **weighted average**. Continuous coordinates of the resulting voxels are
    computed as the weighted average of the original points' coordinates.

    Args:
        coord: (N, 3) – continuous point coordinates (differentiable)
        feat:  (N, C) – point features (differentiable)
        batch: (N,)   – batch indices (integer, will be detached)
        grid_size: voxel size
        eps: small constant for numerical stability

    Returns:
        voxel_feat:      (M, C) – aggregated features (differentiable)
        voxel_coord:     (M, 3) – continuous coordinates of voxels (differentiable, weighted average)
        voxel_cont_grid: (M, 3) – normalized grid coordinates of voxels (differentiable)
        voxel_indices:   (M, 3) – integer voxel indices (detached)
        voxel_batch:     (M,)   – batch indices (detached, sorted)
        inverse:         (N*8,) – mapping from each corner contribution to voxel index (unsorted order)
        total_weight:    (M,)   – sum of weights per voxel (used for averaging)
        corner_weight:   (N*8,) – trilinear weight for each corner contribution
    """
    device = coord.device
    dtype = coord.dtype
    N, C = feat.shape

    # ------------------------------------------------------------------
    # 1. Continuous grid coordinates (differentiable)
    # ------------------------------------------------------------------
    # Global minimum across the entire batch – detached to keep origin fixed
    coord_min = coord.min(dim=0, keepdim=True)[0].detach()
    grid_cont = (coord - coord_min) / grid_size          # (N,3), differentiable
    grid_int = torch.floor(grid_cont).int().detach()     # (N,3), detached
    frac = grid_cont - grid_int                          # (N,3), differentiable

    # ------------------------------------------------------------------
    # 2. Eight corner offsets and trilinear weights
    # ------------------------------------------------------------------
    offsets = torch.tensor(
        [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
         [0,0,1],[1,0,1],[0,1,1],[1,1,1]],
        device=device, dtype=torch.int32
    )  # (8,3)

    # Expand everything to 8 copies per point
    feat_exp = feat.unsqueeze(1).expand(N, 8, C).reshape(-1, C)        # (N*8, C)
    batch_exp = batch.unsqueeze(1).expand(N, 8).reshape(-1)            # (N*8,)
    grid_int_exp = grid_int.unsqueeze(1).expand(N, 8, 3).reshape(-1, 3)
    frac_exp = frac.unsqueeze(1).expand(N, 8, 3).reshape(-1, 3)
    offset_exp = offsets.unsqueeze(0).expand(N, 8, 3).reshape(-1, 3)

    corner_int = grid_int_exp + offset_exp                             # (N*8,3)
    # Trilinear weight = product of (1 - |fraction - offset|)
    w = torch.prod(1 - torch.abs(frac_exp - offset_exp.float()), dim=1)  # (N*8,), differentiable

    # ------------------------------------------------------------------
    # 3. Unique voxel identification (detached)
    #    Use explicit [batch, x, y, z] keys – safe and overflow‑free.
    # ------------------------------------------------------------------
    key = torch.cat(
        [batch_exp.to(torch.int64).unsqueeze(1),
         corner_int.to(torch.int64)],
        dim=1
    ).detach()                                                         # (N*8, 4)

    uniq_key, inverse, _ = torch.unique(
        key, sorted=False, return_inverse=True, return_counts=True, dim=0
    )
    M = len(uniq_key)

    voxel_batch = uniq_key[:, 0].to(torch.long)                        # (M,)
    voxel_indices = uniq_key[:, 1:].to(torch.int32)                   # (M,3)

    # ------------------------------------------------------------------
    # 4. Feature aggregation (differentiable)
    # ------------------------------------------------------------------
    total_weight = torch.zeros(M, device=device, dtype=dtype)
    total_weight.index_add_(0, inverse, w)                             # (M,)

    voxel_feat = torch.zeros(M, C, device=device, dtype=dtype)
    voxel_feat.index_add_(0, inverse, w.unsqueeze(-1) * feat_exp)     # (M,C)
    if normalize_feat:
        voxel_feat = voxel_feat / (total_weight.unsqueeze(-1) + eps)

    # ------------------------------------------------------------------
    # 5. Continuous coordinates of voxels (weighted average)
    # ------------------------------------------------------------------
    # Original continuous grid coordinates, expanded
    grid_cont_exp = grid_cont.unsqueeze(1).expand(N, 8, 3).reshape(-1, 3)  # (N*8,3)
    voxel_cont_grid = torch.zeros(M, 3, device=device, dtype=dtype)
    voxel_cont_grid.index_add_(0, inverse, w.unsqueeze(-1) * grid_cont_exp)
    voxel_cont_grid = voxel_cont_grid / (total_weight.unsqueeze(-1) + eps)  # (M,3), differentiable

    # Convert back to original coordinate space
    voxel_coord = voxel_cont_grid * grid_size + coord_min              # (M,3), differentiable

    # ------------------------------------------------------------------
    # 6. Sort voxels by batch (and then by grid_coord)
    # ------------------------------------------------------------------
    # Pack [batch, x, y, z] into a sortable int64 key.
    # This is safe as long as each coordinate fits in 16 bits.
    sort_key = voxel_batch.long() * (2**48) + (
        voxel_indices[:, 0].long() * (2**32)
        + voxel_indices[:, 1].long() * (2**16)
        + voxel_indices[:, 2].long()
    )
    sort_perm = torch.argsort(sort_key)
    inv_sort_perm = torch.zeros_like(sort_perm).scatter_(
        0, sort_perm, torch.arange(M, device=sort_perm.device)
    )

    # Reorder all voxel‑level tensors
    voxel_batch = voxel_batch[sort_perm]
    voxel_indices = voxel_indices[sort_perm]
    voxel_feat = voxel_feat[sort_perm]
    voxel_coord = voxel_coord[sort_perm]
    voxel_cont_grid = voxel_cont_grid[sort_perm]
    total_weight = total_weight[sort_perm]

    # Remap inverse mapping to sorted order
    inverse_sorted = inv_sort_perm[inverse]                            # (N*8,)

    return (
        voxel_feat,
        voxel_coord,
        voxel_cont_grid,
        voxel_indices,
        voxel_batch,
        inverse_sorted,
        total_weight,
        w,
    )