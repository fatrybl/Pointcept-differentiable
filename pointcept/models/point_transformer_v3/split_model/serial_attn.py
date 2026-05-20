import math

import torch
import torch.nn as nn
import flash_attn

from pointcept.models.modules import PointModule
from pointcept.models.utils.misc import offset2bincount

from .rpe import RPE


class Point3DRoPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 3 == 0, (
            f"Head dimension must be divisible by 3 for 3D RoPE, got {head_dim}"
        )
        self.chunk_dim = head_dim // 3
        inv_freq = 1.0 / (base ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _get_cos_sin(self, xyz):
        freqs = self.inv_freq.unsqueeze(0)
        chunks = []
        for i in range(3):
            emb = xyz[:, i : i + 1] * freqs
            chunks.append(torch.cat([emb, emb], dim=-1))
        emb_3d = torch.cat(chunks, dim=-1)
        return emb_3d.cos().unsqueeze(1), emb_3d.sin().unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q, k, xyz):
        cos, sin = self._get_cos_sin(xyz)
        q_chunks = torch.split(q, self.chunk_dim, dim=-1)
        k_chunks = torch.split(k, self.chunk_dim, dim=-1)
        cos_chunks = torch.split(cos, self.chunk_dim, dim=-1)
        sin_chunks = torch.split(sin, self.chunk_dim, dim=-1)
        q_out, k_out = [], []
        for i in range(3):
            q_out.append(q_chunks[i] * cos_chunks[i] + self._rotate_half(q_chunks[i]) * sin_chunks[i])
            k_out.append(k_chunks[i] * cos_chunks[i] + self._rotate_half(k_chunks[i]) * sin_chunks[i])
        return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
            self.flash_dtype = None
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

        self.rope_base = rope_base
        if rope_base:
            self.rope = Point3DRoPE(head_dim=channels // num_heads, base=rope_base)
            self.shift_coords = shift_coords
            self.jitter_coords = jitter_coords
            self.rescale_coords = rescale_coords

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if self.rope_base:
            rope_coord = point.coord[order].clone()
            if self.training:
                dd = {"device": rope_coord.device, "dtype": rope_coord.dtype}
                if self.shift_coords is not None and self.shift_coords > 0:
                    rope_coord = rope_coord + torch.empty(3, **dd).uniform_(-self.shift_coords, self.shift_coords)
                if self.jitter_coords is not None and self.jitter_coords > 1.0:
                    jitter = math.log(self.jitter_coords)
                    rope_coord = rope_coord * torch.empty(3, **dd).uniform_(-jitter, jitter).exp()
                if self.rescale_coords is not None and self.rescale_coords > 1.0:
                    rescale = math.log(self.rescale_coords)
                    rope_coord = rope_coord * torch.empty(1, **dd).uniform_(-rescale, rescale).exp()

            qkv = qkv.reshape(-1, 3, H, C // H)
            q, k, v = qkv.unbind(dim=1)
            q, k = self.rope(q, k, rope_coord)

            if not self.enable_flash:
                q = q.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
                k = k.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
                v = v.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
                if self.upcast_attention:
                    q, k = q.float(), k.float()
                attn = (q * self.scale) @ k.transpose(-2, -1)
                if self.enable_rpe:
                    attn = attn + self.rpe(self.get_rel_pos(point, order))
                if self.upcast_softmax:
                    attn = attn.float()
                attn = self.softmax(attn)
                attn = self.attn_drop(attn).to(v.dtype)
                feat = (attn @ v).transpose(1, 2).reshape(-1, C)
            else:
                if self.flash_dtype is None:
                    self.flash_dtype = (
                        torch.bfloat16
                        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                        else torch.float16
                    )
                feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                    torch.stack([q, k, v], dim=1).to(self.flash_dtype),
                    cu_seqlens,
                    max_seqlen=self.patch_size,
                    dropout_p=self.attn_drop if self.training else 0,
                    softmax_scale=self.scale,
                ).reshape(-1, C)
                feat = feat.to(qkv.dtype)
        elif not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            if self.upcast_attention:
                q, k = q.float(), k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            if self.flash_dtype is None:
                self.flash_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(self.flash_dtype).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point