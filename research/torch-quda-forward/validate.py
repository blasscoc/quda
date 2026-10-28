"""Independent QUDA comparison, autograd checks, and an optimizer smoke test."""
import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np
import torch

from wilson_network import WilsonNormalNetwork, from_quda_native, to_quda_native


def rel(a, b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quda-bridge', required=True)
    parser.add_argument('--output', default='validation.json')
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    dims = (4, 6, 4, 8)  # Anisotropic, to catch ordering/coordinate mistakes.
    vcb = int(np.prod(dims))//2
    h = torch.randn(4, 2, vcb, 3, 3, dtype=torch.complex128)
    h = (h+h.mH)/2
    h = h - h.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]/3*torch.eye(3)
    u = torch.view_as_real(torch.linalg.matrix_exp(0.3j*h)).contiguous()
    gauge_path = Path('validation_gauge.bin').resolve()
    u.numpy().astype('<f8').tofile(gauge_path)

    lib = ctypes.CDLL(args.quda_bridge)
    lib.quda_replay_init.argtypes = [ctypes.c_int]*4 + [ctypes.c_char_p, ctypes.c_double,
                                                     ctypes.c_double, ctypes.c_ulonglong]
    lib.quda_replay_init.restype = ctypes.c_int
    lib.quda_replay_num_doubles.restype = ctypes.c_size_t
    lib.quda_replay_finalize.restype = None
    rc = lib.quda_replay_init(*dims, str(gauge_path).encode(), 5.6, .156, 123)
    if rc:
        raise RuntimeError(f'QUDA initialization failed: {rc}')
    try:
        assert lib.quda_replay_num_doubles() == vcb*24
        net = WilsonNormalNetwork(u.cuda(), dims)
        x = torch.randn(1, vcb, 4, 3, 2, device='cuda', dtype=torch.float64)
        xq = to_quda_native(x)
        checks = {}
        with torch.no_grad():
            for name, method in [('m', lambda z: net.schur(z)),
                                 ('mdag', lambda z: net.schur(z, dagger=True)),
                                 ('mdagm', net)]:
                fn = getattr(lib, f'quda_replay_apply_{name}_d2d')
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                fn.restype = None
                yq = torch.empty_like(xq)
                fn(xq.data_ptr(), yq.data_ptr())
                ref = from_quda_native(yq)
                y = method(x)
                err = rel(y, ref)
                checks[name + '_relative_error_vs_quda'] = err
                print(name, err, flush=True)
                assert err < 2e-12, (name, err)
            y64 = net(x)
            precision_errors = {}
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                low = WilsonNormalNetwork(u.cuda().to(dtype), dims)
                yp = low(x.to(dtype))
                assert torch.isfinite(yp).all()
                precision_errors[str(dtype)] = rel(yp, y64)
        # Actual autograd, including every gather/projector/color multiplication.
        b = torch.randn_like(x)
        x.requires_grad_(True)
        grad = torch.autograd.grad(net.energy(x, b), x)[0]
        checks['energy_autograd_error'] = rel(grad, net(x)-b)
        assert checks['energy_autograd_error'] < 2e-12
        probe = torch.randn_like(x)
        vjp = torch.autograd.grad((net(x)*probe).sum(), x)[0]
        checks['forward_vjp_error'] = rel(vjp, net(probe))
        assert checks['forward_vjp_error'] < 2e-12
        net.checkpoint_stages = True
        checkpoint_grad = torch.autograd.grad(net.energy(x, b), x)[0]
        checks['checkpoint_gradient_error'] = rel(checkpoint_grad, grad)
        assert checks['checkpoint_gradient_error'] < 2e-12
        net.checkpoint_stages = False
        # A second RHS must produce the same answer independently or batched.
        with torch.no_grad():
            checks['batch_error'] = rel(net(torch.cat([x, probe])),
                                        torch.cat([net(x), net(probe)]))
        assert checks['batch_error'] < 2e-12
        # Save/reload network and input/output independently of QUDA.
        net.save('network.pt')
        torch.save({'x': x.detach().cpu(), 'rhs': b.cpu(), 'Ax': y64.cpu()}, 'example_io.pt')
        restored = WilsonNormalNetwork.load('network.pt', 'cuda')
        checks['reload_error'] = rel(restored(x), y64)
        assert checks['reload_error'] == 0
        # Library optimizer on the NEW pure-Torch forward graph.
        candidate = torch.nn.Parameter(torch.zeros_like(x))
        opt = torch.optim.SGD([candidate], lr=.15)
        residuals = []
        start = time.perf_counter()
        for i in range(20):
            opt.zero_grad(set_to_none=True)
            loss = net.energy(candidate, b)
            loss.backward()
            assert torch.isfinite(candidate.grad).all()
            residuals.append((candidate.grad.norm()/b.norm()).item())
            opt.step()
        torch.cuda.synchronize()
        sgd_seconds = time.perf_counter()-start
        assert residuals[-1] < residuals[0]
        # Tiny independent finite-difference gradient check, including AP time BC.
        tiny_gauge = u[:, :, :8].clone()
        tiny = WilsonNormalNetwork(tiny_gauge, (2,2,2,2), temporal_boundary=-1)
        tx = torch.randn(1,8,4,3,2,dtype=torch.float64,requires_grad=True)
        checks['gradcheck_antiperiodic'] = torch.autograd.gradcheck(tiny, (tx,), fast_mode=True)
        report = {'status': 'passed', 'dims': dims, 'dtype': 'float64', 'gamma_basis': 'UKQCD',
                  'temporal_boundary': 'periodic', 'checks': checks,
                  'reduced_precision_forward_errors': precision_errors,
                  'sgd': {'steps':20, 'residuals':residuals,
                          'wall_seconds':sgd_seconds},
                  'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(),
                  'scope':'Pure torch forward/Schur and autograd, not a QUDA wrapper. Small correctness test, not production performance.'}
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        lib.quda_replay_finalize()


if __name__ == '__main__':
    main()
