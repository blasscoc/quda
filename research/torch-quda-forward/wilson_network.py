"""Differentiable, matrix-free Wilson even/odd operator, using only PyTorch.

Gauge: [4, 2, V/2, 3, 3, 2], QDP direction/parity/checkerboard ordering.
Spinor: [batch, V/2, 4, 3, 2], final axis real/imaginary.
Coordinates are (x,y,z,t), x fastest. UKQCD gamma basis. No clover term.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


def complex_matrix_as_real(matrix):
    """[..., rows, cols, real/imag] -> [..., 2*rows, 2*cols]."""
    re, im = matrix.unbind(-1)
    rows = torch.stack((torch.stack((re, -im), -1),
                        torch.stack((im, re), -1)), -2)
    # rows currently [..., row, col, row_component, col_component].
    return rows.transpose(-3, -2).flatten(-4, -3).flatten(-2, -1)


def gamma_projectors(dtype=torch.float64):
    """I +/- gamma_mu, from QUDA's UKQCD basis (include/gamma.cuh)."""
    cols = [[3, 2, 1, 0], [3, 2, 1, 0], [2, 3, 0, 1], [0, 1, 2, 3]]
    vals = [[1j, 1j, -1j, -1j], [1, -1, -1, 1],
            [1j, -1j, -1j, 1j], [1, 1, -1, -1]]
    gamma = torch.zeros(4, 4, 4, dtype=torch.complex128)
    for mu in range(4):
        gamma[mu, torch.arange(4), cols[mu]] = torch.tensor(vals[mu], dtype=gamma.dtype)
    eye = torch.eye(4, dtype=gamma.dtype)
    return torch.stack([complex_matrix_as_real(torch.view_as_real(eye - gamma)),
                        complex_matrix_as_real(torch.view_as_real(eye + gamma))]).to(dtype)


def checkerboard_geometry(dims, temporal_boundary=1):
    """Neighbour gather indices and boundary signs, output-parity indexed."""
    if len(dims) != 4 or any(d <= 0 or d % 2 for d in dims):
        raise ValueError('All four dimensions must be positive and even.')
    if temporal_boundary not in (-1, 1):
        raise ValueError('temporal_boundary must be +1 or -1')
    lx, ly, lz, lt = dims
    cb = torch.arange(math.prod(dims) // 2, dtype=torch.int64)
    xh = cb % (lx // 2)
    y = cb // (lx // 2) % ly
    z = cb // (lx // 2 * ly) % lz
    t = cb // (lx // 2 * ly * lz)
    neighbours, boundaries = [], []
    for parity in (0, 1):
        x = 2 * xh + ((y + z + t + parity) % 2)
        coord = [x, y, z, t]
        p_neighbours, p_boundaries = [], []
        for mu in range(4):
            n_mu, b_mu = [], []
            for delta in (1, -1):
                c = list(coord)
                c[mu] = (coord[mu] + delta) % dims[mu]
                lex = c[0] + lx * (c[1] + ly * (c[2] + lz * c[3]))
                n_mu.append(lex // 2)
                wrap = (coord[mu] == (dims[mu] - 1 if delta == 1 else 0))
                sign = torch.ones_like(cb, dtype=torch.float64)
                if mu == 3:
                    sign[wrap] = temporal_boundary
                b_mu.append(sign)
            p_neighbours.append(torch.stack(n_mu))
            p_boundaries.append(torch.stack(b_mu))
        neighbours.append(torch.stack(p_neighbours))
        boundaries.append(torch.stack(p_boundaries))
    return torch.stack(neighbours), torch.stack(boundaries)


def load_qdp_gauge(path, dims, *, dtype=torch.float64, device='cpu'):
    """Read this project's raw FP64 QDP checkpoint, without reinterpreting axes."""
    count = 4 * math.prod(dims) * 18
    if Path(path).stat().st_size != count * 8:
        raise ValueError('Checkpoint size does not match supplied full lattice dimensions')
    a = np.fromfile(path, dtype='<f8', count=count).reshape(4, 2, -1, 3, 3, 2)
    return torch.from_numpy(a).to(device=device, dtype=dtype)


def from_quda_native(flat, vector_reals=4):
    """QUDA native vectorized storage -> network layout. Build-dependent order!

    The locally tested FP64 bridge uses four reals (two complex values) per
    site per vector. FLOAT2 builds instead require vector_reals=2.
    """
    if vector_reals not in (2, 4, 8):
        raise ValueError('Supported native vector sizes: 2, 4, 8 reals')
    nc = vector_reals // 2
    return flat.reshape(12//nc, -1, nc, 2).permute(1, 0, 2, 3).reshape(1, -1, 4, 3, 2).contiguous()


def to_quda_native(spinor, vector_reals=4):
    if spinor.shape[0] != 1:
        raise ValueError('QUDA replay bridge accepts one RHS at a time')
    if vector_reals not in (2, 4, 8):
        raise ValueError('Supported native vector sizes: 2, 4, 8 reals')
    nc = vector_reals // 2
    return spinor[0].reshape(-1, 12//nc, nc, 2).permute(1, 0, 2, 3).contiguous().flatten()


class WilsonNormalNetwork(nn.Module):
    """A fixed linear network: forward(x) = M_even^dagger M_even x.

    M_even = I - kappa^2 D_eo D_oe. Each D consists of neighbour gathers,
    gauge/color matrix multiplication, spin projection and summation.
    Gauge weights are buffers, not learned parameters. Autograd differentiates
    the actual PyTorch operations; no QUDA calls or hand-written backward.
    """
    def __init__(self, gauge, dims, kappa=0.156, temporal_boundary=1, checkpoint_stages=False):
        super().__init__()
        dims = tuple(int(d) for d in dims)
        expected = (4, 2, math.prod(dims) // 2, 3, 3, 2)
        if tuple(gauge.shape) != expected or not gauge.is_floating_point():
            raise ValueError(f'Expected a real floating gauge tensor of shape {expected}')
        if not math.isfinite(kappa) or kappa <= 0:
            raise ValueError('kappa must be finite and positive')
        neighbours, signs = checkerboard_geometry(dims, temporal_boundary)
        self.dims, self.kappa = dims, float(kappa)
        self.temporal_boundary = temporal_boundary
        self.checkpoint_stages = bool(checkpoint_stages)
        self.register_buffer('gauge', gauge)
        self.register_buffer('neighbours', neighbours.to(gauge.device), persistent=False)
        self.register_buffer('signs', signs.to(device=gauge.device, dtype=gauge.dtype), persistent=False)
        self.register_buffer('projectors', gamma_projectors(gauge.dtype).to(gauge.device), persistent=False)

    def _check(self, x):
        if x.ndim != 5 or tuple(x.shape[1:]) != (math.prod(self.dims)//2, 4, 3, 2):
            raise ValueError('Expected spinor [batch, V/2, 4, 3, 2]')
        if x.device != self.gauge.device or x.dtype != self.gauge.dtype:
            raise ValueError('Spinor and network must share device and real dtype')

    @staticmethod
    def _color(u, x, dagger=False):
        w = complex_matrix_as_real(u)
        # Row-vector convention: x @ U^T for Ux, x @ U for U^dagger x
        # in the real representation (transpose is the complex adjoint).
        if not dagger:
            w = w.transpose(-1, -2)
        return torch.matmul(x.flatten(-2), w).reshape_as(x)

    @staticmethod
    def _spin(p, x):
        # [..., spin, color, RI] -> [..., color, spin*RI]
        a = x.transpose(-3, -2).flatten(-2)
        return torch.matmul(a, p.T).reshape(*x.shape[:-3], 3, 4, 2).transpose(-3, -2)

    def dslash(self, x, output_parity, dagger=False):
        """Input lives on the opposite parity. No kappa factor in D."""
        self._check(x)
        if output_parity not in (0, 1):
            raise ValueError('output_parity must be 0 or 1')
        out = torch.zeros_like(x)
        for mu in range(4):
            plus, minus = self.neighbours[output_parity, mu]
            fwd = x.index_select(1, plus)
            back = x.index_select(1, minus)
            uf = self.gauge[mu, output_parity]
            ub = self.gauge[mu, 1-output_parity].index_select(0, minus)
            fwd = self._spin(self.projectors[int(dagger), mu], self._color(uf, fwd))
            back = self._spin(self.projectors[1-int(dagger), mu], self._color(ub, back, dagger=True))
            sf, sb = self.signs[output_parity, mu]
            out = out + fwd * sf[None, :, None, None, None] + back * sb[None, :, None, None, None]
        return out

    def schur(self, x, dagger=False):
        def apply(z, parity):
            if self.checkpoint_stages and torch.is_grad_enabled() and z.requires_grad:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(lambda y: self.dslash(y, parity, dagger), z, use_reentrant=False)
            return self.dslash(z, parity, dagger)
        odd = apply(x, 1)
        even = apply(odd, 0)
        return x - self.kappa**2 * even

    def forward(self, x):
        return self.schur(self.schur(x), dagger=True)

    def energy(self, x, rhs, normalize=False):
        """0.5 ||M x||^2 - <b,x>; gradient is A x - b (or normalized)."""
        mx = self.schur(x)
        # Keep reductions out of FP16/BF16 to avoid volume-dependent overflow.
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        loss = 0.5 * mx.to(acc).square().sum() - (rhs.to(acc)*x.to(acc)).sum()
        return loss / rhs.to(acc).square().sum().clamp_min(torch.finfo(acc).tiny) if normalize else loss

    def save(self, path):
        torch.save({'dims': self.dims, 'kappa': self.kappa,
                    'temporal_boundary': self.temporal_boundary,
                    'checkpoint_stages': self.checkpoint_stages,
                    'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device, weights_only=True)
        return cls(data['state_dict']['gauge'], data['dims'], data['kappa'],
                   data['temporal_boundary'], data.get('checkpoint_stages', False))
