import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
from kernels import FEATURE_MAPPING, get_kernel_fn

N = 2000
NS = [4, 6, 8, 10, 12, 14, 16]
SEEDS = 3
MAPS = ["RBF-Class", "RBF-Class-2n", "W2K-Block", "IQP", "HEE", "ZZ"]
CACHE_DIR = "./cache_concentration"

DATASETS = {
    "sst2":    dict(hf=("nyu-mll/glue", "sst2"), text="sentence", shuffle=False),
    "ag_news": dict(hf=("fancyzhx/ag_news",),    text="text",     shuffle=True),
}

def load_data(n_samples, dataset="sst2"):
    cfg = DATASETS[dataset]
    cd = Path(CACHE_DIR)
    cd.mkdir(exist_ok=True, parents=True)
    cache = cd / f"sbert_{dataset}_n{n_samples}.npz"
    if cache.exists():
        data = np.load(cache)
        return data["X"], data["y"]

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize

    ds = load_dataset(*cfg["hf"])["train"]
    if cfg["shuffle"]:
        ds = ds.shuffle(seed=0)
    sents = ds[cfg["text"]][:n_samples]
    labels = np.array(ds["label"][:n_samples])
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = normalize(model.encode(sents, batch_size=64, show_progress_bar=True), axis=1)
    np.savez(cache, X=X, y=labels)
    return X, labels


def subsample(X, y, n_use, rng):
    """Stratified subsample: preserves the pool's class proportions per draw."""
    n_use = min(n_use, len(X))
    classes, counts = np.unique(y, return_counts=True)
    quotas = np.floor(n_use * counts / len(y)).astype(int)
    for i in np.argsort(-counts):  # distribute the rounding remainder
        if quotas.sum() >= n_use: break
        quotas[i] += 1
    idx = np.concatenate([
        rng.choice(np.where(y == c)[0], size=q, replace=False)
        for c, q in zip(classes, quotas)
    ])
    rng.shuffle(idx)
    return X[idx], y[idx]


def slope_with_ci(ns, var_per_seed):
    ns = np.asarray(ns, dtype=float)
    slopes = []
    for v in var_per_seed.values():
        v = np.asarray(v, dtype=float)
        m = v > 0
        if m.sum() >= 2:
            slopes.append(np.polyfit(ns[m], np.log(v[m]), 1)[0])
    slopes = np.asarray(slopes)
    if len(slopes) == 0:
        return np.nan, (np.nan, np.nan)
    mean = float(slopes.mean())
    if len(slopes) >= 2:
        se = slopes.std(ddof=1) / np.sqrt(len(slopes))
        t = st.t.ppf(0.975, df=len(slopes) - 1)
        ci = (mean - t * se, mean + t * se)
    else:
        ci = (mean, mean)
    return mean, ci


def run_experiment(X_pool, y_pool, dataset):
    total_tasks = len(NS) * len(MAPS) * SEEDS
    counter = 0
    out = []
    iu = np.triu_indices(N, k=1)

    print(f"=== Esecuzione Esperimenti di Concentrazione con N={N} ===")
    print(f"Generati {total_tasks} task totali da elaborare.")
    print("Inizio computazione...\n" + "-" * 70)

    for n in sorted(NS):
        for fm in MAPS:
            feats_pool = FEATURE_MAPPING[fm](X_pool, n)
            kernel_fn = get_kernel_fn(fm, n)

            for seed in range(SEEDS):
                t0 = time.time()
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(feats_pool), size=min(N, len(feats_pool)), replace=False)

                K = kernel_fn(feats_pool[idx])
                off_var = float(K[iu].var())
                elapsed = time.time() - t0

                counter += 1
                pct = counter / total_tasks * 100
                print(f"[{pct:6.2f}%] ({counter}/{total_tasks}) -> {fm:<10s} | n={n:<2d} | seed={seed} | Var={off_var:.3e} in {elapsed:.1f}s")
                sys.stdout.flush()

                out.append({"feature_map": fm, "n": n, "seed": seed, "off_diag_var": off_var})

        pd.DataFrame(out).to_csv(
            Path("results", f"concentration_{dataset}_N_{N}_checkpoint_n{n}.csv"), index=False
        )

    print("-" * 70 + "\nComputazione completata con successo.")
    raw = pd.DataFrame(out)

    table1 = []
    for fm in MAPS:
        sub = raw[raw.feature_map == fm]
        vps = {sd: sub[sub.seed == sd].sort_values("n")["off_diag_var"].values for sd in range(SEEDS)}
        slope, (lo, hi) = slope_with_ci(NS, vps)
        table1.append({
            "feature_map": fm,
            "slope_logVar_vs_n": slope,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "per_qubit_factor": float(np.exp(slope))
        })
    table1 = pd.DataFrame(table1)

    print("\n--- Tabella 1: pendenza di log Var(K) vs n ---")
    print(f"  {'Feature map':<12s} {'slope':>8s}   {'95% CI':>20s}   {'fattore/qubit':>14s}")
    for _, r in table1.iterrows():
        print(f"  {r.feature_map:<12s} {r.slope_logVar_vs_n:+8.3f}   "
              f"[{r.ci95_lo:+7.3f}, {r.ci95_hi:+7.3f}]   {r.per_qubit_factor:13.3f}x")
    return raw, table1


def draw_figure(raw, dataset):
    agg = raw.groupby(["feature_map", "n"])["off_diag_var"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.colormaps["viridis"].resampled(len(MAPS))
    for i, fm in enumerate(MAPS):
        sub = agg[agg.feature_map == fm].sort_values("n")
        v = sub["off_diag_var"].values
        m = v > 0
        ax.plot(sub["n"].values[m], v[m], marker="o", color=cmap(i), label=fm)
    ax.set_yscale("log")
    ax.set_xlabel("numero di qubit  n")
    ax.set_ylabel(r"Var($K_{\mathrm{off}}$)")
    ax.set_title(f"Stage 1 -- Analisi di Concentrazione ({dataset})")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()

    p = Path("results", "figs")
    p.mkdir(exist_ok=True, parents=True)
    plt.savefig(Path(p, f"concentration_{dataset}_N_{N}.png"), dpi=150)
    plt.close()
    print(f"[fig] {p}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="sst2")
    args = parser.parse_args()

    print("=== Configurazione dell'esperimento ===")
    print(f"  dataset={args.dataset}  N={N}  ns={NS}  seeds={SEEDS}  maps={MAPS}")

    X_pool, y_pool = load_data(N, args.dataset)
    _, counts = np.unique(y_pool, return_counts=True)
    print(f"  pool: X{X_pool.shape}  (classi: {counts.tolist()})")

    raw1, table1 = run_experiment(X_pool, y_pool, args.dataset)
    draw_figure(raw1, args.dataset)
    raw1.to_csv(Path("results", f"concentration_{args.dataset}_N_{N}_raw.csv"), index=False)
    table1.to_csv(Path("results", f"concentration_{args.dataset}_N_{N}.csv"), index=False)



if __name__ == "__main__":
    main()
