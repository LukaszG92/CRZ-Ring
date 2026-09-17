import matplotlib
import sys, time, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from kernels import FEATURE_MAPPING, RBF_COMPONENTS, get_kernel_fn
from concentration_evaluator import DATASETS, load_data, subsample
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

matplotlib.use("Agg")

N_POOL_VALS  = [400, 800, 1200, 1600, 2000]
N_QUBIT_VALS = [4, 8, 12, 16, 20]
SEEDS = 3

OUT_DIR = Path("./results")
FIG_DIR = OUT_DIR / "figs"

MAPS = ["RBF-Class", "RBF-Class-2n", "W2K-Block", "IQP", "HEE", "ZZ"]


def split_kernel(K_full, n_tr):
    return K_full[:n_tr, :n_tr], K_full[n_tr:, :n_tr]


def evaluate_task(fm, n_q, kernel_fn, X, y, seed):
    """Train/tune/test one QSVM task. Quantum kernels are computed once;
    only RBF recomputes per gamma candidate (cheap, classical)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    X_full = np.vstack([X_tr, X_te])
    n_tr = len(X_tr)

    if fm in RBF_COMPONENTS:
        m = RBF_COMPONENTS[fm](n_q)
        proj = FEATURE_MAPPING[fm](X_full, n_q)
        variance = proj.var()
        scale_val = 1.0 / (m * variance) if variance > 0 else 1.0 / m
        gamma_candidates = [("scale", scale_val), (0.001, 0.001), (0.01, 0.01), (0.1, 0.1), (1.0, 1.0)]
        kernels = {name: split_kernel(rbf_kernel(proj, gamma=val), n_tr) for name, val in gamma_candidates}
    else:
        feats = FEATURE_MAPPING[fm](X_full, n_q)
        kernels = {"scale": split_kernel(kernel_fn(feats), n_tr)}

    best = (-1, 1.0, "scale")
    for c_cand in [0.1, 1.0, 10.0]:
        for g_name, (K_tr, _) in kernels.items():
            clf = SVC(kernel="precomputed", C=c_cand)
            sc = cross_val_score(clf, K_tr, y_tr, cv=StratifiedKFold(3, shuffle=True, random_state=seed)).mean()
            if sc > best[0]:
                best = (sc, c_cand, g_name)

    _, best_c, best_gamma = best
    K_train, K_test = kernels[best_gamma]
    clf = SVC(kernel="precomputed", C=best_c)
    clf.fit(K_train, y_tr)
    preds = clf.predict(K_test)

    f1_avg = "binary" if len(np.unique(y)) <= 2 else "macro"
    return (accuracy_score(y_te, preds),
            f1_score(y_te, preds, average=f1_avg, zero_division=0),
            matthews_corrcoef(y_te, preds))


def run_experiment(X_pool, y_pool, n_pools, n_qubits, seeds, dataset):
    total_tasks = len(MAPS) * len(n_pools) * len(n_qubits) * seeds
    counter = 0
    out = []

    print(f"Generati {total_tasks} task totali da elaborare.")
    print("Inizio computazione...\n" + "-" * 70)

    for n_q in sorted(n_qubits):
        for fm in MAPS:
            # Circuit built and JIT-compiled once per (fm, n_q)
            kernel_fn = None if fm in RBF_COMPONENTS else get_kernel_fn(fm, n_q)

            for n_pool in n_pools:
                for seed in range(seeds):
                    t0 = time.time()
                    rng = np.random.RandomState(seed)
                    X, y = subsample(X_pool, y_pool, n_pool, rng)

                    acc, f1, mcc = evaluate_task(fm, n_q, kernel_fn, X, y, seed)
                    elapsed = time.time() - t0

                    counter += 1
                    pct = counter / total_tasks * 100
                    print(f"[{pct:6.2f}%] ({counter}/{total_tasks}) -> {fm:<10s} | N={n_pool:<4d} | Qubits={n_q:<2d} | Seed={seed} | Acc={acc:.4f} in {elapsed:.1f}s")
                    sys.stdout.flush()

                    out.append({"feature_map": fm, "n_pool": n_pool, "n_qubits": n_q,
                                "seed": seed, "acc": acc, "f1": f1, "mcc": mcc})

        OUT_DIR.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(out).to_csv(OUT_DIR / f"qsvm_{dataset}_checkpoint_q{n_q}.csv", index=False)

    print("-" * 70 + "\nComputazione completata con successo.")
    return pd.DataFrame(out)


def aggregate(raw):
    gp = ["feature_map", "n_pool", "n_qubits"]
    agg = raw.groupby(gp).agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std")
    ).reset_index()
    return agg


def draw_figure(agg, m_mean, m_std, label, title, fname):
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    nq_vals = sorted(agg.n_qubits.unique())
    fig, axes = plt.subplots(1, len(nq_vals), figsize=(4 * len(nq_vals), 4.5), sharey=True)
    if len(nq_vals) == 1: axes = [axes]

    cmap = plt.colormaps["tab10"].resampled(len(MAPS))
    handles = []

    for idx, nq in enumerate(nq_vals):
        ax = axes[idx]
        sub_nq = agg[agg.n_qubits == nq]
        for i, fm in enumerate(MAPS):
            df = sub_nq[sub_nq.feature_map.astype(str) == fm].sort_values("n_pool")
            if df.empty: continue
            h = ax.errorbar(df.n_pool, df[m_mean], yerr=df[m_std], fmt="-o",
                            color=cmap(i), label=fm, capsize=3, alpha=0.8)
            if idx == 0:
                handles.append(h)
        ax.set_xlabel("N_pool (Train+Test)")
        ax.set_title(f"Qubit n = {nq}")
        ax.grid(True, ls=":", alpha=0.5)
        if idx == 0:
            ax.set_ylabel(label)

    fig.legend(handles=handles, labels=MAPS, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=len(MAPS), fontsize=9)
    plt.suptitle(title, y=0.88, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p = FIG_DIR / fname
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Grafico Salvato] -> {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Esegue un test rapido ridotto")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="sst2")
    args = parser.parse_args()

    if args.quick:
        n_pool_vals = [400]
        n_qubit_vals = [4, 6]
        seeds = 1
        print("=== MODALITÀ VELOCE (QUICK TEST) ATTIVATA ===")
    else:
        n_pool_vals = N_POOL_VALS
        n_qubit_vals = N_QUBIT_VALS
        seeds = SEEDS
        print("=== CONFIGURAZIONE COMPLETA SVOLGIMENTO ESPERIMENTO ===")

    print(f"Parametri: dataset={args.dataset} | N_pool={n_pool_vals} | Qubits={n_qubit_vals} | Seed={seeds}")

    max_N = max(n_pool_vals)
    X_pool, y_pool = load_data(max_N, args.dataset)

    raw = run_experiment(X_pool, y_pool, n_pool_vals, n_qubit_vals, seeds, args.dataset)

    OUT_DIR.mkdir(exist_ok=True, parents=True)

    out_raw = OUT_DIR / f"qsvm_{args.dataset}_raw_results.csv"
    out_agg = OUT_DIR / f"qsvm_{args.dataset}_aggregated_results.csv"

    raw.to_csv(out_raw, index=False)
    print(f"\n[Fatto] Dati grezzi salvati in -> {out_raw}")

    agg = aggregate(raw)
    agg.to_csv(out_agg, index=False)
    print(f"[Fatto] Medie aggregate salvate in -> {out_agg}")

    ds_label = args.dataset.upper().replace("_", " ")
    draw_figure(agg, "acc_mean", "acc_std", "Accuracy", f"Confronto Accuratezza QSVM ({ds_label})",
                    f"fig_scaled_acc_vs_N_{args.dataset}.png")
    draw_figure(agg, "mcc_mean", "mcc_std", "Matthews Corr (MCC)", f"Stabilità dei Modelli Quantistici (MCC, {ds_label})",
                    f"fig_scaled_mcc_vs_N_{args.dataset}.png")


if __name__ == "__main__":
    main()
