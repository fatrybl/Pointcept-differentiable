"""Custom implementation of Point() structure with differentiable operations."""

import torch
from typing import Optional, Sequence, Union
from pointcept.models.utils.structure import Point
from pointcept.models.utils import batch2offset
from pointcept.models.utils.serialization import encode
from pointcept.models.point_transformer_v3.differentiable.utils.trilinear_splatting import (
    trilinear_splatting,
)

class DiffPoint(Point):
    """
    Differentiable extension of Point class.

    Adds support for continuous coordinates and differentiable voxelization / serialization.
    All newly added tensors that are meant to carry gradients are stored as separate attributes.
    Original attributes required by the base model (grid_coord, serialized_code, ...) are
    set to **detached** versions derived from the differentiable pipeline, ensuring full
    compatibility with non‑differentiable operations (spconv, indexing, etc.).

    New attributes (all optional):
        continuous_grid_coord: (M, 3) – normalized continuous grid coordinates of each voxel,
            differentiable w.r.t. input points.
        voxel_indices: (M, 3) – integer voxel indices (detached).
        voxel_feat: (M, C) – aggregated features per voxel, differentiable.
        voxel_batch: (M,) – batch indices of voxels (detached).
        continuous_code: (k, M) – continuous approximation of space‑filling curve codes.
        hard_code: (k, M) – integer codes from encode() on detached grid_coord.
        hard_order: (k, M) – argsort of hard_code.
        hard_inverse: (k, M) – inverse of hard_order.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.continuous_grid_coord: Optional[torch.Tensor] = None
        self.voxel_indices: Optional[torch.Tensor] = None
        self.voxel_feat: Optional[torch.Tensor] = None
        self.voxel_batch: Optional[torch.Tensor] = None
        self.continuous_code: Optional[torch.Tensor] = None
        self.hard_code: Optional[torch.Tensor] = None
        self.hard_order: Optional[torch.Tensor] = None
        self.hard_inverse: Optional[torch.Tensor] = None

    def differentiable_voxelize(self, grid_size: Optional[float] = None) -> None:
        """Perform differentiable voxelization via trilinear splatting.

        Replaces the hard voxelization step with a continuous, differentiable
        aggregation of points into a sparse voxel grid. Gradients flow from
        the aggregated voxel features back to the original point coordinates
        through the trilinear weights.

        After this method:
            - self.feat is replaced by self.voxel_feat (differentiable)
            - self.grid_coord is replaced by self.voxel_indices (detached)
            - self.batch is replaced by self.voxel_batch (detached)
            - self.offset is recomputed accordingly
            - self.continuous_grid_coord stores the normalized continuous
              coordinates of each voxel (differentiable)

        Args:
            grid_size: size of voxel grid. If None, tries to read self.grid_size.
        """
        if grid_size is None:
            grid_size = self.get('grid_size')
        assert grid_size is not None, 'grid_size must be provided or set in Point'

        num_points = self.coord.shape[0]
        orig_batch_size = int(self.batch.max()) + 1 if self.batch.numel() > 0 else 0

        (
            voxel_feat,
            voxel_coord,
            voxel_cont_grid,
            voxel_indices,
            batch_voxel,
            inverse,
            total_weight,
            corner_weight,
        ) = trilinear_splatting(
            coord=self.coord,
            feat=self.feat,
            batch=self.batch,
            grid_size=float(grid_size),
            eps=1e-8,
            normalize_feat=False,
        )

        assert torch.all(batch_voxel >= 0) and torch.all(batch_voxel < orig_batch_size), \
            "Decoded batch indices out of range"

        num_voxels = voxel_indices.shape[0]

        voxel_origin_coord = None
        if "origin_coord" in self.keys():
            origin_coord_exp = self.origin_coord.unsqueeze(1).expand(num_points, 8, 3).reshape(-1, 3)
            voxel_origin_coord = torch.zeros(
                num_voxels, 3, device=self.origin_coord.device, dtype=self.origin_coord.dtype
            )
            voxel_origin_coord.index_add_(0, inverse, corner_weight.unsqueeze(-1) * origin_coord_exp)
            voxel_origin_coord = voxel_origin_coord / (total_weight.unsqueeze(-1) + 1e-8)

        voxel_color = None
        if "color" in self.keys():
            color_dim = self.color.shape[1]
            color_exp = self.color.unsqueeze(1).expand(num_points, 8, color_dim).reshape(-1, color_dim)
            voxel_color = torch.zeros(
                num_voxels, color_dim, device=self.color.device, dtype=self.color.dtype
            )
            voxel_color.index_add_(0, inverse, corner_weight.unsqueeze(-1) * color_exp)
            voxel_color = voxel_color / (total_weight.unsqueeze(-1) + 1e-8)

        self._point_to_voxel_map = inverse

        # ------------------------------------------------------------------
        # 7. Store results and update base attributes
        # ------------------------------------------------------------------
        self.continuous_grid_coord = voxel_cont_grid
        self.voxel_indices = voxel_indices
        self.voxel_feat = voxel_feat
        self.voxel_batch = batch_voxel 

        # Replace base attributes with voxel‑level data
        self.coord = voxel_coord
        self.feat = voxel_feat
        self.grid_coord = voxel_indices
        self.batch = batch_voxel
        self.offset = batch2offset(self.batch)
        if voxel_origin_coord is not None:
            self.origin_coord = voxel_origin_coord
        if voxel_color is not None:
            self.color = voxel_color
        if self.batch.numel() > 0:
            self['batch_size'] = int(self.batch.max()) + 1

        assert self.coord.shape[0] == self.feat.shape[0] == self.grid_coord.shape[0] == self.batch.shape[0], \
            "DiffPoint voxelization produced inconsistent tensor lengths"

    @staticmethod
    def _continuous_morton_code(coords: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Compute a continuous, differentiable approximation of Morton (Z‑order) code.

        Args:
            coords: (N, 3) tensor of coordinates in the range [0, 2^depth).
            depth: number of bits for serialization.

        Returns:
            (N,) tensor of continuous codes, differentiable w.r.t. coords.
        """
        norm_coords = coords / (2 ** depth)                 # [0,1)
        code = torch.zeros(len(coords), device=coords.device)
        for i in range(depth):
            scale = 2 ** i
            x = (norm_coords[:, 0] * scale).fmod(1.0)
            y = (norm_coords[:, 1] * scale).fmod(1.0)
            z = (norm_coords[:, 2] * scale).fmod(1.0)
            code += (
                x * 4 ** (depth - i - 1)
                + y * 2 ** (depth - i - 1)
                + z
            )
        return code

    def differentiable_serialize(
        self,
        order: Union[str, Sequence[str]] = 'z',
        depth: Optional[int] = None,
        shuffle_orders: bool = False,
    ) -> None:
        """
        Differentiable serialization.

        Assumes differentiable_voxelize() has been called beforehand.
        Computes both continuous (differentiable) and hard (detached) serialization codes.
        Sets the original base class serialization attributes using the hard versions
        so that the rest of the model remains compatible.

        Args:
            order: ordering strategy ('z', 'z‑trans', or a list/tuple of such).
            depth: depth of serialization cube. If None, inferred from self.grid_coord.
            shuffle_orders: if True, randomly permute the order of multiple orderings.
        """
        # Ensure voxelization is done
        if self.continuous_grid_coord is None:
            self.differentiable_voxelize()

        if depth is None:
            depth = int(self.grid_coord.max() + 1).bit_length()
        self['serialized_depth'] = depth

        if isinstance(order, str):
            orders = [order]
        else:
            orders = list(order)

        # Prepare containers
        num_orders = len(orders)
        M = len(self.grid_coord)

        cont_codes = torch.zeros(num_orders, M, device=self.feat.device)
        hard_codes = torch.zeros(num_orders, M, device=self.feat.device, dtype=torch.int64)
        hard_orders = torch.zeros(num_orders, M, device=self.feat.device, dtype=torch.long)
        hard_inverses = torch.zeros(num_orders, M, device=self.feat.device, dtype=torch.long)

        # Continuous code: uses continuous_grid_coord (differentiable)
        # Hard code: uses encode() on detached grid_coord and batch
        assert self.continuous_grid_coord is not None
        for i, ord_str in enumerate(orders):
            # Continuous (differentiable)
            cont_codes[i] = self._continuous_morton_code(self.continuous_grid_coord, depth)

            # Hard (detached) – original encode expects int64 grid_coord
            hard_code = encode(
                self.grid_coord.to(torch.int64),
                self.batch,
                depth,
                order=ord_str
            )                                                       # (M,), int64
            hard_codes[i] = hard_code

            # Sorting
            hard_order = torch.argsort(hard_code)
            hard_inverse = torch.zeros_like(hard_order).scatter_(
                dim=0,
                index=hard_order,
                src=torch.arange(M, device=hard_order.device)
            )
            hard_orders[i] = hard_order
            hard_inverses[i] = hard_inverse

        if shuffle_orders:
            perm = torch.randperm(num_orders)
            cont_codes = cont_codes[perm]
            hard_codes = hard_codes[perm]
            hard_orders = hard_orders[perm]
            hard_inverses = hard_inverses[perm]

        # Store continuous codes (for later differentiable ordering)
        self.continuous_code = cont_codes

        # Store hard tensors (detached) for base class compatibility
        self.hard_code = hard_codes
        self.hard_order = hard_orders
        self.hard_inverse = hard_inverses

        # Set the original serialization attributes (expected by the model)
        self['serialized_code'] = hard_codes
        self['serialized_order'] = hard_orders
        self['serialized_inverse'] = hard_inverses
        self['order'] = orders

    def serialization(
        self,
        order: Union[str, Sequence[str]] = 'z',
        depth: Optional[int] = None,
        shuffle_orders: bool = False
    ) -> None:
        """
        Override of base serialization.

        Replaces the non‑differentiable hard voxelization + bit‑interleaving
        with the differentiable pipeline. After this call, the Point instance
        contains all attributes required by the Sonata model, but gradients
        can flow from downstream losses back to the original point coordinates.

        See differentiable_serialize() for details.
        """
        self.differentiable_serialize(order, depth, shuffle_orders)