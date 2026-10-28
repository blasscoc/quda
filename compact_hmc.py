"""Experimental standalone two-flavour Wilson HMC; no QUDA dependency.

Persistent numerical fields use the selected codec. FP32/complex64 arithmetic
is confined to site tiles, plus scalar reductions and native MMA accumulators.
Quantized solves and forces are approximate; this is NOT a validated sampler.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from packed_fp4 import PackedWilsonFP4, pack_rows, unpack_rows
from reduced_forward import ReducedWilson, DTYPES, encode_rows, decode_rows
from wilson_network import WilsonNormalNetwork, gamma_projectors, complex_matrix_as_real

PRECISIONS = ('bf16', 'fp16', 'tf32', 'fp8', 'fp4')


class Field:
    """Rows of six reals; FP4 stores three bytes + one FP8 scale per row."""
    def __init__(self, shape, precision, pair=None):
        self.shape, self.precision = tuple(shape), precision
        assert shape[-1] == 6
        if pair is not None:
            self.q, self.s = pair
            return
        self.q = torch.empty((*shape[:-1], 3 if precision == 'fp4' else 6),
                             device='cuda', dtype=torch.uint8 if precision == 'fp4' else DTYPES[precision])
        self.s = torch.empty(shape[:-1] if precision in ('fp4', 'fp8') else (0,),
                             device='cuda', dtype=torch.float8_e4m3fn if precision == 'fp4' else torch.float32)

    def get(self, index=...):
        q = self.q[index].contiguous()
        s = self.s[index].contiguous() if self.s.numel() else self.s
        return unpack_rows(q, s) if self.precision == 'fp4' else decode_rows(q, s, self.precision)

    def put(self, index, value):
        q, s = pack_rows(value) if self.precision == 'fp4' else encode_rows(value, self.precision)
        self.q[index] = q
        if self.s.numel():
            self.s[index] = s

    def clone(self):
        return Field(self.shape, self.precision, (self.q.clone(), self.s.clone()))

    def bytes(self):
        return self.q.numel()*self.q.element_size() + self.s.numel()*self.s.element_size()

    def pair(self):
        return self.q, self.s


def generators():
    lam = torch.zeros(8, 3, 3, dtype=torch.complex64, device='cuda')
    for a, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        lam[2*a, i, j] = lam[2*a, j, i] = 1
        lam[2*a+1, i, j], lam[2*a+1, j, i] = -1j, 1j
    lam[6] = torch.diag(torch.tensor([1., -1., 0.], device='cuda')).to(torch.complex64)
    lam[7] = torch.diag(torch.tensor([1., 1., -2.], device='cuda')).to(torch.complex64)/math.sqrt(3)
    return 1j*lam/math.sqrt(2)  # -Tr(Ta Tb) = delta_ab


class HMC:
    def __init__(self, dims, precision, kappa=.156, beta=5.6, tile=8192, boundary=1):
        if len(dims) != 4 or any(n <= 0 or n % 2 for n in dims):
            raise ValueError('Four positive even lattice dimensions required')
        if tile <= 0 or not 0 < kappa or not math.isfinite(kappa) or not math.isfinite(beta):
            raise ValueError('Invalid tile, kappa or beta')
        self.dims, self.v, self.precision = tuple(dims), math.prod(dims)//2, precision
        self.kappa, self.beta, self.tile, self.boundary = kappa, beta, tile, boundary
        self.gauge = Field((4, 2, self.v, 3, 6), precision)
        self.T = generators()
        self.projectors = gamma_projectors(torch.float32).cuda()
        # Allocate native operator WITHOUT constructing a full floating gauge.
        cls = PackedWilsonFP4 if precision == 'fp4' else ReducedWilson
        self.op = cls.__new__(cls)
        torch.nn.Module.__init__(self.op)
        for k, val in dict(dims=self.dims, v=self.v, kappa=kappa, precision=precision,
                           group_sites=4, unroll=8, num_warps=1).items():
            setattr(self.op, k, val)
        self.op.register_buffer('indices', torch.empty(2, 4, 2, self.v, device='cuda', dtype=torch.int32))
        self.op.register_buffer('signs', torch.ones(2, 4, 2, self.v, device='cuda', dtype=torch.int8))
        weights = Field((2, 8, self.v, 3, 6), precision)
        self.op.register_buffer('weights', weights.q)
        self.op.register_buffer('scales', weights.s)
        self.weights = weights
        for p in (0, 1):
            for sl, ids in self.tiles():
                for mu in range(4):
                    for b, delta in enumerate((1, -1)):
                        j, sign = self.shift(p, ids, mu, delta)
                        self.op.indices[p, mu, b, sl] = j.int()
                        self.op.signs[p, mu, b, sl] = sign.to(torch.int8)
        self.breakdowns = torch.zeros((), device='cuda', dtype=torch.int64)

    def tiles(self):
        for start in range(0, self.v, self.tile):
            end = min(start+self.tile, self.v)
            yield slice(start, end), torch.arange(start, end, device='cuda')

    def shift(self, p, ids, mu, delta):
        lx, ly, lz, lt = self.dims
        y = ids//(lx//2) % ly
        z = ids//(lx//2*ly) % lz
        t = ids//(lx//2*ly*lz)
        coords = [2*(ids % (lx//2)) + (y+z+t+p) % 2, y, z, t]
        wrap = (coords[mu]+delta < 0) | (coords[mu]+delta >= self.dims[mu])
        coords[mu] = (coords[mu]+delta) % self.dims[mu]
        idx = (coords[0]+lx*(coords[1]+ly*(coords[2]+lz*coords[3])))//2
        sign = torch.where(wrap, self.boundary, 1) if mu == 3 else torch.ones_like(ids)
        return idx, sign

    def u(self, mu, p, ids):
        return torch.view_as_complex(self.gauge.get((mu, p, ids)).reshape(-1, 3, 3, 2).contiguous())

    def cold(self):
        eye = torch.view_as_real(torch.eye(3, dtype=torch.complex64, device='cuda')).reshape(3, 6)
        for sl, ids in self.tiles():
            for mu in range(4):
                for p in (0, 1):
                    self.gauge.put((mu, p, sl), eye.expand(len(ids), 3, 6))
        self.refresh()

    def refresh(self):
        for p in (0, 1):
            for sl, ids in self.tiles():
                for mu in range(4):
                    for back in (0, 1):
                        j = self.op.indices[p, mu, 1, sl].long() if back else ids
                        u = self.u(mu, 1-p if back else p, j)
                        if back:
                            u = u.mH
                        w = complex_matrix_as_real(torch.view_as_real(u.resolve_conj()))[:, ::2, :]
                        self.weights.put((p, 2*mu+back, sl), w)

    def vector(self):
        return Field((1, self.v, 4, 6), self.precision)

    def random_vector(self):
        x = self.vector()
        for sl, ids in self.tiles():
            x.put((slice(None), sl), torch.randn(1, len(ids), 4, 6, device='cuda')/math.sqrt(2))
        return x

    def apply(self, x):
        return Field(x.shape, self.precision, self.op(*x.pair()))

    def dslash(self, x, parity, dagger=False, skip=None):
        return Field(x.shape, self.precision, self.op.dslash(*x.pair(), parity, dagger,
                     skip=None if skip is None else skip.pair()))

    def schur(self, x, dagger=False):
        odd = self.dslash(x, 1, dagger)
        return self.dslash(odd, 0, dagger, skip=x)

    def dot(self, x, y):
        total = torch.zeros((), device='cuda')
        for sl, _ in self.tiles():
            ix = (slice(None), sl)
            total += (x.get(ix)*y.get(ix)).sum()
        return total

    def axpy(self, out, a, x, b, y):
        for sl, _ in self.tiles():
            ix = (slice(None), sl)
            out.put(ix, a*x.get(ix)+b*y.get(ix))

    def solve(self, rhs, iterations):
        x = self.vector()
        x.q.zero_()
        if x.s.numel():
            x.s.fill_(1)
        r, p = rhs.clone(), rhs.clone()
        rr = self.dot(r, r)
        # Fixed number of applications, no convergence checks or host transfers.
        for _ in range(iterations):
            ap = self.apply(p)
            pap = self.dot(p, ap)
            good = torch.isfinite(pap) & (pap > 1.e-30) & torch.isfinite(rr)
            self.breakdowns += (~good).long()
            alpha = torch.where(good, rr/pap.clamp_min(1.e-30), 0.)
            self.axpy(x, 1., x, alpha, p)
            self.axpy(r, 1., r, -alpha, ap)
            new = self.dot(r, r)
            ratio = torch.where(rr > 1.e-30, new/rr.clamp_min(1.e-30), 0.)
            self.axpy(p, 1., r, ratio, p)
            rr = new
            del ap
        return x

    def momentum(self):
        # Eight independent Lie algebra coordinates, padded to two codec rows.
        p = Field((4, 2, self.v, 2, 6), self.precision)
        for sl, ids in self.tiles():
            for mu in range(4):
                for parity in (0, 1):
                    a = torch.zeros(len(ids), 12, device='cuda')
                    a[:, :8] = torch.randn(len(ids), 8, device='cuda')
                    p.put((mu, parity, sl), a.reshape(-1, 2, 6))
        return p

    def kinetic(self, mom):
        out = torch.zeros((), device='cuda')
        for sl, _ in self.tiles():
            for mu in range(4):
                for p in (0, 1):
                    out += .5*mom.get((mu, p, sl)).flatten(1)[:, :8].square().sum()
        return out

    def gauge_action(self):
        out = torch.zeros((), device='cuda')
        for sl, ids in self.tiles():
            for p in (0, 1):
                for mu in range(4):
                    jm, _ = self.shift(p, ids, mu, 1)
                    for nu in range(mu+1, 4):
                        jn, _ = self.shift(p, ids, nu, 1)
                        loop = self.u(mu, p, ids)@self.u(nu, 1-p, jm)@self.u(mu, 1-p, jn).mH@self.u(nu, p, ids).mH
                        out += self.beta*(3.-loop.diagonal(dim1=-2, dim2=-1).real.sum(-1)).sum()/3
        return out

    def energy(self, phi, mom, iterations):
        x = self.solve(phi, iterations)
        return self.gauge_action()+self.kinetic(mom)+self.dot(phi, x)

    def force_tile(self, mu, p, ids, left, right):
        """Derivative along exp(theta_a T_a)U. Autograd retains one tile only.

        Fermion surrogate: dS_f = -x^dag d(M^dag M)x with x held fixed.
        Quantization is not differentiated; approximate smooth force at decoded U.
        """
        u = self.u(mu, p, ids)
        staple = torch.zeros_like(u)
        jm, sign = self.shift(p, ids, mu, 1)
        for nu in range(4):
            if nu == mu:
                continue
            jn, _ = self.shift(p, ids, nu, 1)
            jbn, _ = self.shift(p, ids, nu, -1)
            jmbn, _ = self.shift(1-p, jm, nu, -1)
            staple += self.u(nu, 1-p, jm)@self.u(mu, 1-p, jn).mH@self.u(nu, p, ids).mH
            staple += self.u(nu, p, jmbn).mH@self.u(mu, 1-p, jbn).mH@self.u(nu, 1-p, jbn)
        with torch.enable_grad():
            theta = torch.zeros(len(ids), 8, device='cuda', requires_grad=True)
            du = torch.einsum('na,aij->nij', theta.to(torch.complex64), self.T)@u
            ue = u+du
            action = -self.beta/3*(ue@staple).diagonal(dim1=-2, dim2=-1).real.sum()
            ur = torch.view_as_real(ue)
            lp = left[p].get((0, ids)).reshape(-1, 4, 3, 2)
            lq = left[1-p].get((0, jm)).reshape(-1, 4, 3, 2)
            rp = right[p].get((0, ids)).reshape(-1, 4, 3, 2)
            rq = right[1-p].get((0, jm)).reshape(-1, 4, 3, 2)
            fw = WilsonNormalNetwork._spin(self.projectors[0, mu], WilsonNormalNetwork._color(ur, rq))
            bw = WilsonNormalNetwork._spin(self.projectors[1, mu], WilsonNormalNetwork._color(ur, rp, dagger=True))
            action = action+2*self.kappa**2*((lp*fw+lq*bw)*sign[:, None, None, None]).sum()
            return torch.autograd.grad(action, theta)[0].detach()

    def kick(self, mom, phi, dt, iterations):
        x = self.solve(phi, iterations)
        z = self.dslash(x, 1)
        y = self.dslash(z, 0, skip=x)  # M x
        w = self.dslash(y, 1, dagger=True)
        for sl, ids in self.tiles():
            for mu in range(4):
                for p in (0, 1):
                    grad = self.force_tile(mu, p, ids, (y, w), (x, z))
                    coeff = mom.get((mu, p, sl)).flatten(1)
                    coeff[:, :8] -= dt*grad
                    mom.put((mu, p, sl), coeff.reshape(-1, 2, 6))

    def drift(self, mom, dt):
        for sl, ids in self.tiles():
            for mu in range(4):
                for p in (0, 1):
                    a = mom.get((mu, p, sl)).flatten(1)[:, :8]
                    gen = torch.einsum('na,aij->nij', a.to(torch.complex64), self.T)
                    u = torch.linalg.matrix_exp(dt*gen)@self.u(mu, p, ids)
                    self.gauge.put((mu, p, sl), torch.view_as_real(u).reshape(-1, 3, 6))
        self.refresh()

    def trajectory(self, steps, dt, iterations, energy_iterations):
        backup = self.gauge.clone()
        mom = self.momentum()
        eta = self.random_vector()
        phi = self.schur(eta, dagger=True)
        del eta
        h0 = self.energy(phi, mom, energy_iterations)
        for i in range(steps):
            self.kick(mom, phi, dt*(.5 if i == 0 else 1.), iterations)
            self.drift(mom, dt)
        self.kick(mom, phi, dt*.5, iterations)
        h1 = self.energy(phi, mom, energy_iterations)
        delta = float(h1-h0)
        accepted = math.isfinite(delta) and math.log(max(float(torch.rand((), device='cuda')), 1.e-30)) < -delta
        if not accepted:
            self.gauge = backup
            self.refresh()
        return dict(h_initial=float(h0), h_final=float(h1), delta_h=delta, accepted=accepted)

    def checkpoint(self, path, config, completed):
        payload = dict(format='compact-hmc-v1', config=config, completed=completed,
                       gauge_q=self.gauge.q, gauge_s=self.gauge.s,
                       rng=torch.cuda.get_rng_state(), experimental=True)
        tmp = Path(str(path)+'.tmp')
        torch.save(payload, tmp)
        tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dims', type=int, nargs=4, default=[4, 4, 4, 4])
    p.add_argument('--precision', choices=PRECISIONS, default='fp4')
    p.add_argument('--kappa', type=float, default=.156)
    p.add_argument('--beta', type=float, default=5.6)
    p.add_argument('--boundary', type=int, choices=(-1, 1), default=1)
    p.add_argument('--tile', type=int, default=8192)
    p.add_argument('--solver-iters', type=int, default=8)
    p.add_argument('--energy-iters', type=int, default=None)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--dt', type=float, default=.01)
    p.add_argument('--trajectories', type=int, default=1)
    p.add_argument('--seed', type=int, default=45)
    p.add_argument('--checkpoint', type=Path, default=Path('compact_hmc.pt'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--log', type=Path, default=Path('compact_hmc.jsonl'))
    a = p.parse_args()
    if min(a.solver_iters, a.steps, a.trajectories, a.tile) < 1 or not math.isfinite(a.dt) or a.dt <= 0:
        p.error('Positive iteration counts, tile and finite dt required')
    if a.energy_iters is None:
        a.energy_iters = a.solver_iters
    if a.energy_iters < 1:
        p.error('Positive energy iteration count required')
    config = {k: v for k, v in vars(a).items() if k not in ('checkpoint', 'resume', 'log')}
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = a.precision == 'tf32'
    saved = torch.load(a.resume, map_location='cpu', weights_only=True) if a.resume else None
    if saved:
        if saved.get('format') != 'compact-hmc-v1':
            p.error('Unknown checkpoint format')
        for k in ('dims', 'precision', 'kappa', 'beta', 'boundary', 'tile', 'solver_iters', 'energy_iters', 'steps', 'dt'):
            if saved['config'][k] != config[k]:
                p.error(f'Resume mismatch for {k}: supply the original settings')
    h = HMC(a.dims, a.precision, a.kappa, a.beta, a.tile, a.boundary)
    completed = 0
    if saved:
        h.gauge.q.copy_(saved['gauge_q'])
        h.gauge.s.copy_(saved['gauge_s'])
        h.refresh()
        torch.cuda.set_rng_state(saved['rng'])
        completed = saved['completed']
        del saved
    else:
        h.cold()
    a.log.parent.mkdir(parents=True, exist_ok=True)
    a.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with a.log.open('a') as log:
        for i in range(a.trajectories):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            h.breakdowns.zero_()
            start = time.perf_counter()
            result = h.trajectory(a.steps, a.dt, a.solver_iters, a.energy_iters)
            torch.cuda.synchronize()
            result.update(trajectory=completed+i+1, precision=a.precision, dims=a.dims,
                          seconds=time.perf_counter()-start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          gauge_bytes=h.gauge.bytes(), operator_bytes=sum(t.numel()*t.element_size() for t in h.op.buffers()),
                          solver_breakdowns=int(h.breakdowns), experimental=True, config=config)
            # Strict JSON: nonfinite energies become null and are always rejected.
            result = {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in result.items()}
            line = json.dumps(result, allow_nan=False)
            print(line, flush=True)
            log.write(line+'\n')
            log.flush()
            h.checkpoint(a.checkpoint, config, completed+i+1)


if __name__ == '__main__':
    main()
