"""Packed linear solves with a common policy and measured stopping contract.

Persistent vectors/moments use backend.vector(), including the FP4 expansion.
Torch optimizers see one decoded tile at a time, never a full precision lattice.
The gradient is A(x)-b for the quadratic system; quantized A is approximate.
Muon orthogonalizes each tile's (sites,24) matrix independently: tile size is
part of that algorithm, unlike the elementwise SGD/Adam adapters.
"""
import copy
import json
import math
import time

import torch

from hmc_config import SOLVER_DEFAULTS, merge


def zero_vector(backend):
    x = backend.vector()
    x.q.zero_()
    if x.s.numel():
        x.s.fill_(1)
    if hasattr(x, 'low'):
        x.low.q.zero_()
        x.low.s.fill_(1)
    return x


def residual_norm(backend, rhs, ax):
    """Decode bounded tiles; do NOT repack/round the residual before measuring.

This measures the configured packed operator, not an independent FP64 operator.
FP64 scalar reduction prevents the norm sum losing small tile contributions.
"""
    total = torch.zeros((), device=rhs.q.device, dtype=torch.float64)
    for sl, _ in backend.tiles():
        ix = (slice(None), sl)
        residual = rhs.get(ix)-ax.get(ix)
        total += residual.square().sum(dtype=torch.float64)
    return float(total.sqrt())


class StopPolicy:
    def __init__(self, settings, rhs_norm):
        self.settings = settings
        self.rhs_norm = rhs_norm
        self.tolerance = settings['rtol'] is not None or settings['atol'] is not None
        self.threshold = max((settings['atol'] or 0.), (settings['rtol'] or 0.)*rhs_norm)
        self.enabled = self.tolerance or settings['stagnation_checks'] > 0 or settings['divergence_factor'] is not None
        self.best, self.stale = rhs_norm, 0

    def due(self, iteration, maximum):
        return self.enabled and (iteration % self.settings['check_every'] == 0 or iteration == maximum)

    def evaluate(self, norm, iteration):
        if not math.isfinite(norm):
            return 'nonfinite'
        if iteration < self.settings['min_iterations']:
            return None
        if self.tolerance and norm <= self.threshold:
            return 'converged'
        factor = self.settings['divergence_factor']
        if factor is not None and norm > factor*self.rhs_norm:
            return 'diverged'
        if iteration == 0:
            return None
        if norm < self.best*(1.-self.settings['min_relative_improvement']):
            self.best, self.stale = norm, 0
        else:
            self.stale += 1
        if self.settings['stagnation_checks'] and self.stale >= self.settings['stagnation_checks']:
            return 'stagnated'
        return None


class PackedTorchOptimizer:
    """One reusable Torch parameter/optimizer; state is paged from packed fields.

All tiles use the same global iteration counter and the same pre-update Ax.
There is no weight decay: it would change the requested linear system.
"""
    def __init__(self, backend, settings):
        self.backend, self.settings = backend, settings
        self.parameter = torch.nn.Parameter(torch.empty((1, 24), device=backend.gauge.q.device))
        common = dict(lr=settings['learning_rate'], weight_decay=0.)
        name = settings['algorithm']
        if name == 'sgd':
            self.optimizer = torch.optim.SGD([self.parameter], **common, momentum=settings['momentum'],
                                             nesterov=settings['nesterov'], foreach=False)
            keys = ['momentum_buffer'] if settings['momentum'] else []
        elif name == 'adam':
            self.optimizer = torch.optim.Adam([self.parameter], **common, betas=tuple(settings['betas']),
                                              eps=settings['epsilon'], foreach=False, fused=False)
            keys = ['exp_avg', 'exp_avg_sq']
        elif name == 'muon':
            self.optimizer = torch.optim.Muon([self.parameter], **common, momentum=settings['momentum'],
                                              nesterov=settings['nesterov'], eps=settings['epsilon'],
                                              ns_steps=settings['muon_ns_steps'])
            keys = ['momentum_buffer']
        else:
            raise ValueError(name)
        self.state = {key: zero_vector(backend) for key in keys}

    @torch.no_grad()
    def update(self, x, rhs, ax, iteration):
        for sl, _ in self.backend.tiles():
            ix = (slice(None), sl)
            value = x.get(ix).clone()  # TF32 decode may be a view of packed storage.
            shape = value.shape
            self.parameter.data = value.reshape(-1, 24)
            self.parameter.grad = (ax.get(ix)-rhs.get(ix)).reshape(-1, 24)
            state = {}
            # SGD initializes momentum to the gradient on its first step.
            if iteration > 1 or self.settings['algorithm'] != 'sgd':
                state = {key: field.get(ix).clone().reshape(-1, 24) for key, field in self.state.items()}
            if self.settings['algorithm'] == 'adam':
                state['exp_avg_sq'].clamp_min_(0.)
                state['step'] = torch.tensor(float(iteration-1))
            self.optimizer.state[self.parameter] = state
            self.optimizer.step()
            x.put(ix, self.parameter.detach().reshape(shape))
            for key, field in self.state.items():
                field.put(ix, self.optimizer.state[self.parameter][key].reshape(shape))
        # Do not retain the last decoded tile between calls.
        self.optimizer.state.clear()
        self.parameter.grad = None
        self.parameter.data = torch.empty((0, 24), device=x.q.device)


def solve_packed(backend, rhs, iterations, settings=None, operator=None):
    settings = merge(SOLVER_DEFAULTS, settings or {})
    settings['iterations'] = iterations
    operator = operator or backend.apply
    sync = lambda: torch.cuda.synchronize(rhs.q.device) if rhs.q.is_cuda else None
    sync()
    started = time.perf_counter()
    x = zero_vector(backend)
    rhs_norm = residual_norm(backend, rhs, x)
    policy = StopPolicy(settings['stopping'], rhs_norm)
    name = settings['algorithm']
    record = dict(index=len(backend.solves)+1, label=getattr(backend, 'solve_label', 'unspecified'),
                  algorithm=name, variant='tile_muon' if name == 'muon' else name,
                  settings=copy.deepcopy(settings), rhs_norm=rhs_norm, iterations=[],
                  max_iterations=iterations, starting_relative_residual=0. if rhs_norm == 0 else 1.,
                  residual_definition='norm(decoded b - decoded packed A(x)) / norm(decoded b)',
                  stopping_threshold=policy.threshold if policy.tolerance else None,
                  persistent_state_bytes=0, operator_applications=0, check_operator_applications=0)
    backend.solves.append(record)
    verbose = backend.diagnostics
    if verbose:
        print(json.dumps(dict(event='solve_start', index=record['index'], label=record['label'], algorithm=name)), flush=True)
    reason = 'converged' if rhs_norm == 0 else policy.evaluate(rhs_norm, 0)
    last_norm, measured_at = rhs_norm, 0
    adapter = None
    if not reason:
        if name == 'cg':
            r, p = rhs.clone(), rhs.clone()
            rr = backend.dot(r, r)
            record['persistent_state_bytes'] = x.bytes()+r.bytes()+p.bytes()
        else:
            adapter = PackedTorchOptimizer(backend, settings)
            record['persistent_state_bytes'] = x.bytes()+sum(v.bytes() for v in adapter.state.values())
    else:
        record['persistent_state_bytes'] = x.bytes()
    for iteration in range(1, iterations+1):
        if reason:
            break
        sync()
        begin = time.perf_counter()
        ap = operator(p if name == 'cg' else x)
        record['operator_applications'] += 1
        sync()
        operator_seconds = time.perf_counter()-begin
        extras = {}
        if name == 'cg':
            pap = backend.dot(p, ap)
            good = torch.isfinite(pap) & (pap > 1.e-30) & torch.isfinite(rr)
            backend.breakdowns += (~good).long()
            alpha = torch.where(good, rr/pap.clamp_min(1.e-30), 0.)
            backend.axpy(x, 1., x, alpha, p)
            backend.axpy(r, 1., r, -alpha, ap)
            interval = settings['residual_refresh_every']
            refreshed = bool(interval and iteration % interval == 0)
            if refreshed:
                del ap
                sync(); op_start = time.perf_counter()
                ap = operator(x)
                record['operator_applications'] += 1
                sync(); operator_seconds += time.perf_counter()-op_start
                backend.axpy(r, 1., rhs, -1., ap)
            new = backend.dot(r, r)
            ratio = torch.zeros_like(rr) if refreshed else torch.where(rr > 1.e-30, new/rr.clamp_min(1.e-30), 0.)
            backend.axpy(p, 1., r, ratio, p)
            rr = new
            extras = dict(alpha=float(alpha), beta=float(ratio), pAp=float(pap), denominator_ok=bool(good),
                          residual_refreshed=refreshed,
                          recursive_relative_residual=float(rr.sqrt())/rhs_norm)
            if not bool(good):
                reason = 'breakdown'
        else:
            adapter.update(x, rhs, ap, iteration)
        del ap
        sync()
        algorithm_seconds = time.perf_counter()-begin
        check_start = time.perf_counter()
        period = getattr(backend, 'residual_every', 1)
        measure = (verbose and period > 0 and (iteration % period == 0 or iteration == iterations)) or policy.due(iteration, iterations) or bool(reason)
        norm = None
        if measure:
            ax = operator(x)
            record['operator_applications'] += 1
            record['check_operator_applications'] += 1
            norm = residual_norm(backend, rhs, ax)
            del ax
            last_norm, measured_at = norm, iteration
            # Logging cadence must not change a configured stopping decision.
            if policy.due(iteration, iterations) or reason:
                reason = policy.evaluate(norm, iteration) or reason
            if not math.isfinite(norm):
                reason = 'nonfinite'
        sync()
        row = dict(iteration=iteration, seconds=algorithm_seconds, operator_seconds=operator_seconds,
                   diagnostic_seconds=time.perf_counter()-check_start,
                   recomputed_relative_residual=norm/rhs_norm if norm is not None else None, **extras)
        record['iterations'].append(row)
        if verbose:
            print(json.dumps(dict(event='solver_iteration', solve=record['index'], **row)), flush=True)
    sync()
    count = len(record['iterations'])
    final_norm = last_norm if measured_at == count else None
    relative = final_norm/rhs_norm if final_norm is not None and rhs_norm else (0. if rhs_norm == 0 else None)
    record.update(actual_iterations=count, stop_reason=reason or 'max_iterations',
                  converged=(final_norm <= policy.threshold if final_norm is not None else False) if policy.tolerance else (True if rhs_norm == 0 else None),
                  ending_relative_residual=relative, starting_normalized_loss=.5*record['starting_relative_residual']**2,
                  ending_normalized_loss=.5*relative**2 if relative is not None else None,
                  seconds=time.perf_counter()-started,
                  seconds_per_iteration=sum(r['seconds'] for r in record['iterations'])/count if count else None)
    record['seconds_per_iteration_with_checks'] = record['seconds']/count if count else None
    if verbose:
        print(json.dumps(dict(event='solve_end', **{k:v for k,v in record.items() if k not in ('iterations','settings')})), flush=True)
    return x
