"""Run the saved forward network or optimize its input with torch.optim."""
import argparse
import json
from pathlib import Path

import torch

from wilson_network import WilsonNormalNetwork


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--network', default='network.pt')
    p.add_argument('--io', default='example_io.pt')
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['fp64','fp32','fp16','bf16'], default='fp64')
    p.add_argument('--optimizer', choices=['sgd','adam','adamw'], default='sgd')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--lr', type=float, default=.15)
    p.add_argument('--checkpoint-stages', action='store_true')
    p.add_argument('--output', default='replay_result.json')
    a = p.parse_args()
    if a.steps < 0:
        p.error('--steps must be nonnegative')
    torch.set_float32_matmul_precision('highest')
    dtype = {'fp64':torch.float64,'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[a.precision]
    net = WilsonNormalNetwork.load(a.network, a.device).to(dtype=dtype)
    net.checkpoint_stages = a.checkpoint_stages
    data = torch.load(a.io, map_location=a.device, weights_only=True)
    x, b = data['x'].to(dtype), data['rhs'].to(dtype)
    with torch.no_grad():
        y = net(x)
        err = ((y.double()-data['Ax'].double()).norm()/data['Ax'].double().norm()).item()
    candidate = torch.nn.Parameter(torch.zeros_like(x))
    cls = {'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[a.optimizer]
    opt = cls([candidate], lr=a.lr, weight_decay=0)
    history = []
    for i in range(a.steps):
        opt.zero_grad(set_to_none=True)
        loss = net.energy(candidate, b)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(candidate.grad).all():
            raise RuntimeError('Nonfinite loss/gradient')
        history.append({'step':i, 'loss':loss.item()})
        opt.step()
    with torch.no_grad():
        residual = ((net(candidate).double()-b.double()).norm()/b.double().norm()).item()
    r = {'forward_relative_error_vs_saved':err,'optimizer':a.optimizer,
         'steps':a.steps,'precision':a.precision,'final_true_relative_residual':residual,'history':history}
    Path(a.output).write_text(json.dumps(r, indent=2))
    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
