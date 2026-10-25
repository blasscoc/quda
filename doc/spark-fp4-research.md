# Spark FP4 research snapshot

This experimental branch imports the committed QUDA source snapshot from the
Playground research repository at `89c6bc2200119df208f4e38eb7c249f6e5e6a3c9`.
Its upstream-compatible HMC base is
`0dc8468a0bb2e2e2084e4d987b74c8610dfc6985`.

The original experiment kept QUDA under `src/`; those tracked sources are
mapped to the repository root here. Existing upstream vendor headers and test
scripts absent from that experiment snapshot are retained. Checkpoints, gauge
configurations, cached build products, and untracked files are not included.

The snapshot includes experimental Blackwell FP4 kernels, numerical comparison
and probe programs, the ensemble runner, and the in-place gauge-update tuning
state fix. This is a research checkpoint, not a claim that every experimental
kernel improves performance or has production-level validation. Rejected
experimental branches are not merged here.

The import was checked against the committed source snapshot. No fresh full
QUDA build or GPU validation was performed as part of publishing this branch.
Historical local build recipes depended on cached objects; reproducibility
from a clean upstream build remains a separate validation task.
