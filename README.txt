Torch Wilson forward network
===========================

Standalone experimental HMC with packed gauge, momenta and solver state is now
available: see COMPACT_HMC.txt and run_compact_hmc.sh. Supports BF16/FP16/TF32/
FP8/FP4, fixed solve budgets, and checkpoint/restart; no QUDA dependency.

Packed FP4 forward is now available: see PACKED_FP4_RESULTS.txt.
Run ./run_packed_forward.sh to read the saved packed model/input and write packed
FP4 output directly. No floating-vector conversions occur in that forward path.
For clean operator-only timing/memory across FP4/FP8/FP16/BF16, see
ISOLATED_FORWARD_RESULTS.txt and run_isolated_forward.sh. Those measurements
exclude the optimizer, QUDA validation and all reference-vector allocations.

A new standalone torch.nn.Module implementing the actual unimproved Wilson
even/odd normal operator A = M_even^dagger M_even. This is not a surrogate
and does not call QUDA during forward or backward. No custom CUDA or custom
autograd.Function: gathers, real matrix multiplications and sums are ordinary
PyTorch operations. Gauge matrices are fixed network buffers; the solution
spinor is the input being optimized by torch.optim.

Architecture:
  x_even -> [D_oe -> D_eo -> x - kappa^2 D_eo D_oe x]
         -> [adjoint Schur block] -> A x_even
Each D has eight neighbour/transport/projector branches, added together.
No nonlinear activations: the physical operator is linear for fixed gauge.

Minimal use:
  net = WilsonNormalNetwork(gauge, dims=(Lx,Ly,Lz,Lt), kappa=0.156)
  y = net(x)
  loss = net.energy(x, rhs)   # 0.5 ||M x||^2 - <rhs,x>
  loss.backward()           # gradient wrt x is A x - rhs

Tensor conventions:
  gauge [4,2,V/2,3,3,2], spinor [batch,V/2,4,3,2], final axis real/imag.
  Dimensions (x,y,z,t), x fastest, all even, UKQCD spin basis.
  Gauge uses direction/parity/checkerboard ordering, like raw QDP checkpoints.
  Default periodic fermion boundary; temporal_boundary=-1 is also supported.
  load_qdp_gauge(path,dims) loads a full raw FP64 checkpoint with size checking.
  Native QUDA spinor vector order is BUILD DEPENDENT. The tested bridge uses
  four reals per site per vector; conversion helpers expose vector_reals.
  Do not treat raw native spinors as ordinary site-major arrays.

Run inside the installed container (from this directory):
  docker run --rm --gpus all --ipc=host -v "$PWD:/work" -w /work nvcr.io/nvidia/pytorch:26.09-py3 python replay.py --steps 20
  # Adam alternative:
  # ... python replay.py --optimizer adam --lr 0.05 --steps 40

Validation requires the existing QUDA bridge only as an independent oracle:
  docker run --rm --gpus all --ipc=host -v "$PWD:/work" -v /home/vjordan/.local/state/experiment-playground/teams/research-lab:/home/vjordan/.local/state/experiment-playground/teams/research-lab:ro -w /work nvcr.io/nvidia/pytorch:26.09-py3 python validate.py --quda-bridge /home/vjordan/.local/state/experiment-playground/teams/research-lab/experiments/fermion-force-research/ckpt_replay_pytorch/libquda_replay_op.so

validate.py writes network.pt and example_io.pt, then replay.py is independent
of QUDA. validation.json records direct M, Mdagger and MdaggerM comparisons,
gradient/VJP checks, batching, checkpointing, save/reload, reduced precision,
finite-difference gradcheck and an SGD smoke test. Tests use a nontrivial random
SU(3) field on 4x6x4x8, not an identity field.

FP64/FP32/FP16/BF16 are exposed via standard torch dtypes. The original module
stays pure PyTorch. Experimental native FP4 kernels now live in fp4_color.py,
used by optimized_network.py. Ordinary torch.matmul does not become FP4.
See BENCHMARKS.txt for measurements and important precision limitations.

This is a correctness-first executable network, not a production-performance
replacement. Full 56^3x96 training has not been memory/performance validated.
checkpoint_stages=True recomputes Dslash stages on backward to save activations.
Reductions promote FP16/BF16 to FP32. No clover, force-output or full HMC chain
is implemented here. Raw checkpoint metadata must match this operator.

Conventions checked against local QUDA include/gamma.cuh (UKQCD basis),
include/gauge_field_order.h (QDP order), lib/dirac_wilson.cpp (Schur complement)
and the existing replay bridge. The implementation is a separate PyTorch model.

FP4 roundoff research demo: FP4_ROUNDOFF_RESEARCH.txt develops the formalism
and cites AI literature; FP4_ROUNDOFF_RESULTS.txt and FP4_ROUNDOFF_SOLVERS.csv
contain measured 32^4 ablations and the 128^4 memory model. Run
run_fp4_roundoff.sh for individual cases; production defaults remain unchanged.
