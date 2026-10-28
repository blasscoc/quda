"""GPU contract tests; small independent FP32 force oracle, no QUDA."""
import gc
import json
import tempfile
from pathlib import Path

import torch
from compact_hmc import HMC, Field, PRECISIONS
from wilson_network import WilsonNormalNetwork


def dense_gauge(h):
    return torch.stack([torch.stack([torch.view_as_real(h.u(mu, p, torch.arange(h.v, device='cuda')))
                                   for p in (0, 1)]) for mu in range(4)])


def oracle_gauge_action(u, h):
    out = torch.zeros((), device='cuda')
    ids = torch.arange(h.v, device='cuda')
    c = torch.view_as_complex(u.contiguous())
    for p in (0, 1):
        for mu in range(4):
            jm, _ = h.shift(p, ids, mu, 1)
            for nu in range(mu+1, 4):
                jn, _ = h.shift(p, ids, nu, 1)
                plaquette = c[mu, p]@c[nu, 1-p, jm]@c[mu, 1-p, jn].mH@c[nu, p].mH
                out += h.beta/3*(3-plaquette.diagonal(dim1=-2, dim2=-1).real.sum(-1)).sum()
    return out


def check_force(boundary):
    torch.backends.cuda.matmul.allow_tf32 = False
    h = HMC((2, 4, 2, 4), 'tf32', tile=7, boundary=boundary)
    raw = torch.randn(4, 2, h.v, 3, 3, device='cuda', dtype=torch.complex64)
    ah = (raw-raw.mH)*.12
    ah -= ah.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]/3*torch.eye(3, device='cuda')
    u = torch.view_as_real(torch.linalg.matrix_exp(ah)).contiguous()
    h.gauge.put(..., u.flatten(-2))
    x = torch.randn(1, h.v, 4, 3, 2, device='cuda')*.2
    net = WilsonNormalNetwork(u, h.dims, h.kappa, boundary)
    z = net.dslash(x, 1)
    y = x-h.kappa**2*net.dslash(z, 0)
    w = net.dslash(y, 1, True)
    fields = []
    for t in (x, z, y, w):
        f = h.vector()
        f.put(..., t.flatten(-2))
        fields.append(f)
    xf, zf, yf, wf = fields
    theta = torch.zeros(4, 2, h.v, 8, device='cuda', requires_grad=True)
    uc = torch.view_as_complex(u)
    perturbed = uc+torch.einsum('...a,aij->...ij', theta.to(torch.complex64), h.T)@uc
    ur = torch.view_as_real(perturbed)
    ref = WilsonNormalNetwork(ur, h.dims, h.kappa, boundary)
    value = oracle_gauge_action(ur, h)-ref.schur(x).square().sum()
    expected = torch.autograd.grad(value, theta)[0]
    max_error = 0.
    for sl, ids in h.tiles():
        for mu in range(4):
            for p in (0, 1):
                actual = h.force_tile(mu, p, ids, (yf, wf), (xf, zf))
                error = (actual-expected[mu, p, sl]).abs().max().item()
                max_error = max(max_error, error)
                torch.testing.assert_close(actual, expected[mu, p, sl], atol=4e-5, rtol=4e-5)
    return max_error


def check_mode(precision):
    torch.manual_seed(45)
    torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32'
    h = HMC((4, 4, 4, 4), precision, tile=64)
    h.cold()
    rhs = h.random_vector()
    n = [0]
    apply = h.apply
    def counted(x):
        n[0] += 1
        return apply(x)
    h.apply = counted
    x = h.solve(rhs, 3)
    assert n[0] == 3
    assert torch.isfinite(x.get()).all()
    h.apply = apply
    # Exercise cache refresh on a changed gauge and compare to the independent
    # floating Wilson operator at the decoded gauge/input (outside trajectory).
    probe_momentum = h.momentum()
    h.drift(probe_momentum, .02)
    actual = h.apply(rhs).get().reshape(1, h.v, 4, 3, 2)
    gauge = dense_gauge(h)
    ref = WilsonNormalNetwork(gauge, h.dims, h.kappa, h.boundary)
    expected = ref(rhs.get().reshape(1, h.v, 4, 3, 2))
    forward_error = float((actual-expected).norm()/expected.norm())
    assert forward_error < {'fp4': .4, 'fp8': .12, 'bf16': .03, 'fp16': .003, 'tf32': .003}[precision], (precision, forward_error)
    del probe_momentum, actual, gauge, ref, expected
    h.cold()
    # Packed vectors, momenta and gauge: no volume-sized FP32 fields in FP4 mode.
    mom = h.momentum()
    if precision == 'fp4':
        for f in (rhs, x, mom, h.gauge, h.weights):
            assert f.q.dtype == torch.uint8 and f.s.dtype == torch.float8_e4m3fn
    for sl, _ in h.tiles():
        for mu in range(4):
            for p in (0, 1):
                assert (mom.get((mu, p, sl)).flatten(1)[:, 8:] == 0).all()
    # Forced rejection must exactly restore packed gauge bits.
    before = h.gauge.clone()
    energy = h.energy
    calls = [0]
    def rejected(*args):
        calls[0] += 1
        return torch.tensor(0. if calls[0] == 1 else float('inf'), device='cuda')
    h.energy = rejected
    result = h.trajectory(1, .001, 2, 2)
    assert not result['accepted']
    assert torch.equal(h.gauge.q, before.q)
    assert torch.equal(h.gauge.s.float(), before.s.float())
    h.energy = energy
    result = h.trajectory(1, .001, 2, 2)
    assert all(torch.isfinite(torch.tensor(result[k])) for k in ('h_initial', 'h_final', 'delta_h'))
    # Restoring packed state and RNG reproduces the next trajectory exactly.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)/'state.pt'
        h.checkpoint(path, {}, 1)
        a = h.trajectory(1, .001, 2, 2)
        final = h.gauge.clone()
        saved = torch.load(path, weights_only=True, map_location='cpu')
        h.gauge.q.copy_(saved['gauge_q'])
        h.gauge.s.copy_(saved['gauge_s'])
        h.refresh()
        torch.cuda.set_rng_state(saved['rng'])
        b = h.trajectory(1, .001, 2, 2)
        assert a == b, (precision, a, b)
        assert torch.equal(h.gauge.q, final.q)
        assert torch.equal(h.gauge.s.float(), final.s.float())
    return dict(precision=precision, forward_error=forward_error, gauge_bytes=h.gauge.bytes(), vector_bytes=rhs.bytes(),
                momentum_bytes=mom.bytes(), trajectory=result)


if __name__ == '__main__':
    torch.manual_seed(7)
    evidence = dict(force_errors={str(b): check_force(b) for b in (1, -1)}, modes=[])
    print(json.dumps({'force_errors': evidence['force_errors']}), flush=True)
    for mode in PRECISIONS:
        row = check_mode(mode)
        evidence['modes'].append(row)
        print(json.dumps(row), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    sizes = {r['precision']: r for r in evidence['modes']}
    for key in ('gauge_bytes', 'vector_bytes', 'momentum_bytes'):
        assert sizes['fp16'][key] == 3*sizes['fp4'][key]
        assert sizes['tf32'][key] == 6*sizes['fp4'][key]
    Path('compact_hmc_validation.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print('PASS: force, fixed iteration count, all precision modes, rejection, restart, storage ratios', flush=True)
