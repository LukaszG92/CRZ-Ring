# CRZRing — fidelity quantum kernels beyond exponential concentration

Code and results for the paper **"CRZRing: Scalable Fidelity Quantum Kernels Beyond
Exponential Concentration"**, currently under review.

The paper introduces CRZRing, a fidelity feature map that encodes `2n` SVD components on
`n` qubits through a layer of single-qubit `RY` rotations followed by a nearest-neighbour
ring of data-dependent `CRZ` gates. This repository contains the two experiments reported
in the paper — a kernel-concentration analysis and a QSVM classification benchmark — on
SST-2 and AG News.

## Feature maps

Six kernels are compared at equal qubit count `n`:

| Name in the code | Name in the paper | SVD components | Notes |
| --- | --- | --- | --- |
| `W2K-Block` | CRZRing | `2n` | `RY` layer + closed `CRZ` ring, `2n` CNOTs, depth `O(1)` |
| `RBF-Class` | RBF-Classic | `n` | Classical Gaussian RBF, matches the quantum baselines' input budget |
| `RBF-Class-2n` | RBF-Classic-2n | `2n` | Same kernel, matches CRZRing's input budget |
| `IQP` | IQP | `n` | Complete-graph `ZZ` phases, 2 repetitions |
| `HEE` | HEE | `n` | CNOT chain ansatz, 2 repetitions |
| `ZZ` | ZZ-Ising | `n` | Complete graph, `(π−x_i)(π−x_j)` phase function |

The two RBF arms differ only in how many SVD components they receive; the bandwidth
heuristic `γ = 1/(m · Var(z))` follows that budget `m`.

## Scripts

Three modules, each runnable on its own:

- **`kernels.py`** — feature preprocessing and kernel evaluation for every map. SVD
  projection, percentile `tanh` rescaling, PennyLane/JAX circuits compiled once per
  `(map, n)`, and memory-bounded batching of the statevector Gram computation
  (8 GB budget, so the full state matrix fits up to `n = 18`).
- **`concentration_evaluator.py`** — concentration experiment. Measures the variance of
  the off-diagonal Gram entries on a pool of `N = 2000` inputs for `n ∈ {4, 6, …, 16}`
  over 3 seeds, then fits the slope `α` of `log Var(K_off)` against `n` with a 95% CI.
  Also holds the shared data loading, SBERT caching and stratified subsampling used by
  the other scripts.
- **`qsvm.py`** — classification experiment. Precomputes each Gram matrix, tunes `C` and
  (for the RBF arms) `γ` by 3-fold CV on the training block, and reports test accuracy,
  F1 and MCC over a grid of 5 pool sizes × 5 qubit counts × 3 seeds.

```bash
python concentration_evaluator.py --dataset sst2       # or ag_news
python qsvm.py --dataset sst2                          # --quick for a reduced grid
```

Text is embedded with `all-MiniLM-L6-v2` (384-d, unit-normalised) and projected by SVD.
Embeddings are cached in `cache_concentration/` on first use; everything downstream reads
that cache, so a rerun does not re-encode the corpus.

## Results

`results/` holds the data behind the paper's tables and figures:

| File | Content |
| --- | --- |
| `concentration_{dataset}_N_2000.csv` | Per-map slope `α`, 95% CI and per-qubit factor `e^α` (paper Table 2) |
| `concentration_{dataset}_N_2000_raw.csv` | Per-`(map, n, seed)` off-diagonal variance |
| `qsvm_{dataset}_aggregated_results.csv` | Mean and s.d. of accuracy, F1 and MCC per `(map, N, n)` (paper Table 3) |
| `qsvm_{dataset}_raw_results.csv` | Per-seed values behind the aggregates |
| `figs/` | Variance vs. qubit count, and accuracy/MCC vs. pool size |

Provenance note: the `RBF-Class-2n` rows of the QSVM results come from a later control
run, added to the original sweep. That control run also recomputed `RBF-Class`, and on
SST-2 its numbers differ from the original sweep; the `RBF-Class` rows kept here are the
original ones, which are the numbers printed in the paper.

## Requirements

Python 3.14 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pennylane`
(`lightning.qubit`), `jax`, `sentence-transformers` and `datasets`. The results in
`results/` were produced with PennyLane 0.45, JAX 0.10.1 and scikit-learn 1.8.

All experiments are exact statevector simulations — no shot noise, no hardware backend.
The `n = 20` configurations of the QSVM grid are the expensive ones; the concentration
sweep stops at `n = 16`.
