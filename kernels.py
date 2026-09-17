import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml
from sklearn.metrics.pairwise import rbf_kernel

jax.config.update("jax_enable_x64", True)

N_REPS = 2


def _svd_projections(X, n):
    x_bar = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - x_bar, full_matrices=False)
    return (X - x_bar) @ Vt[:n].T


def _scaled_features(X, n):
    proj = _svd_projections(X, n)
    sigma95 = np.percentile(np.abs(proj), 95, axis=0) + 1e-12
    return np.pi * np.tanh(proj / sigma95)


@jax.jit
def _states_to_kernel(states):
    return jnp.abs(states.conj() @ states.T) ** 2


# Single-process budget: full state matrix fits up to n=18 (8.4 GB);
# n=20 (33.6 GB) still falls back to memory-bounded batching.
_BATCH_MEM_BYTES = 8 * 1024 * 1024 * 1024


def _batch_size(n_qubits):
    state_bytes = (2 ** n_qubits) * 16
    return max(1, _BATCH_MEM_BYTES // state_bytes)


def _kernel_from_circuit(batched_fn, feats, n_qubits):
    n = feats.shape[0]
    bs = _batch_size(n_qubits)
    feats_jax = jnp.array(feats)

    if bs >= n:
        return np.array(_states_to_kernel(batched_fn(feats_jax)))

    K = np.zeros((n, n))
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        si = batched_fn(feats_jax[i0:i1])
        K[i0:i1, i0:i1] = np.array(_states_to_kernel(si))
        for j0 in range(i1, n, bs):
            j1 = min(j0 + bs, n)
            sj = batched_fn(feats_jax[j0:j1])
            block = np.array(jnp.abs(si.conj() @ sj.T) ** 2)
            K[i0:i1, j0:j1] = block
            K[j0:j1, i0:i1] = block.T
    return K


# ── Feature preprocessing ──────────────────────────────────────────────────────
# One call per (fm, n); results are reused across seeds in the evaluator.

FEATURE_MAPPING = {
    "W2K-Block": lambda X, n: np.arccos(np.tanh(_svd_projections(X, 2 * n))),
    "IQP":       _scaled_features,
    "HEE":       _scaled_features,
    "ZZ":        _scaled_features,
    "RBF-Class": lambda X, n: _svd_projections(X, n),
    "RBF-Class-2n": lambda X, n: _svd_projections(X, 2 * n),
}

# SVD components consumed by each map; the RBF arms differ only in this budget.
RBF_COMPONENTS = {"RBF-Class": lambda n: n, "RBF-Class-2n": lambda n: 2 * n}


def get_kernel_fn(fm, n):
    """Return a compiled kernel evaluator for (fm, n).

    Circuit and JIT compilation happen once here; the returned callable only
    takes precomputed features and returns the Gram matrix as a numpy array.
    """
    if fm in RBF_COMPONENTS:
        m = RBF_COMPONENTS[fm](n)

        def rbf_fn(feats):
            variance = feats.var()
            gamma_val = 1.0 / (m * variance) if variance > 0 else 1.0 / m
            return rbf_kernel(feats, gamma=gamma_val)
        return rbf_fn

    dev = qml.device("lightning.qubit", wires=n)

    if fm == "W2K-Block":
        @qml.qnode(dev, interface="jax")
        def circuit(t):
            for i in range(n):
                qml.RY(t[i], wires=i)
            for i in range(n):
                qml.CRZ(t[2 * n - 1 - i], wires=[i, (i + 1) % n])
            return qml.state()

    elif fm == "IQP":
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.IQPEmbedding(x, wires=range(n), n_repeats=N_REPS)
            return qml.state()

    elif fm == "HEE":
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            for _ in range(N_REPS):
                for i in range(n):
                    qml.RY(x[i], wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()

    elif fm == "ZZ":
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            for _ in range(N_REPS):
                for i in range(n):
                    qml.Hadamard(wires=i)
                for i in range(n):
                    qml.RZ(2.0 * x[i], wires=i)
                for i in range(n):
                    for j in range(i + 1, n):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                        qml.CNOT(wires=[i, j])
            return qml.state()

    batched = jax.jit(jax.vmap(circuit))
    return lambda feats: _kernel_from_circuit(batched, feats, n)


# ── Backward-compatible wrappers (used by qsvm.py) ────────────────────────────

def kernel_rbf_classic(X, n, gamma="scale", m=None):
    m = n if m is None else m
    proj = _svd_projections(X, m)
    variance = proj.var()
    gamma_val = 1.0 / (m * variance) if (gamma == "scale" and variance > 0) else (1.0 / m if gamma == "scale" else gamma)
    return rbf_kernel(proj, gamma=gamma_val)


def _compat(fm):
    def wrapper(X, n):
        return get_kernel_fn(fm, n)(FEATURE_MAPPING[fm](X, n))
    return wrapper


KERNEL_MAPPING = {
    "W2K-Block": _compat("W2K-Block"),
    "IQP":       _compat("IQP"),
    "HEE":       _compat("HEE"),
    "ZZ":        _compat("ZZ"),
    "RBF-Class": kernel_rbf_classic,
    "RBF-Class-2n": lambda X, n, gamma="scale": kernel_rbf_classic(X, n, gamma, m=2 * n),
}
