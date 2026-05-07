"""
Quantum Kernel + Dimensionality Reduction — Results Plotter
============================================================
Usage:
    python plot_results.py                        # reads all *.csv in current dir
    python plot_results.py --csv_dir results/     # custom directory
    python plot_results.py --out_dir figures/     # custom output directory
    python plot_results.py --datasets gaussian mnist   # filter datasets

Per-dataset figures  (identical layout for every dataset):
    {ds}_exp1_accuracy_kta.png          — Exp 1: Accuracy & KTA vs d' (all DR methods)
    {ds}_exp2_method_bars.png           — Exp 2: DR method bar comparison + heatmaps
    {ds}_exp3_expressivity.png          — Exp 3: Kernel variance / eff-rank / Frob norm vs d'
    {ds}_exp4_stability.png             — Exp 4: KTA std & gen-gap vs d'

Cross-dataset overview figures:
    overview_accuracy_heatmap.png       — mean test accuracy: dataset x DR method
    overview_kta_heatmap.png            — mean KTA:          dataset x DR method
    overview_best_per_dataset.png       — best test accuracy per dataset (bar)
    overview_radar.png                  — normalised metric radar per DR method
    overview_compute_scaling.png        — wall-clock time vs d' across datasets
"""

import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────
DR_COLOR  = {"pca": "#4C72B0", "rp": "#DD8452", "umap": "#55A868",
             "fs":  "#C44E52", "ae": "#8172B3"}
DR_MARKER = {"pca": "o",  "rp": "s", "umap": "^", "fs": "D", "ae": "P"}
DR_LABEL  = {"pca": "PCA", "rp": "Random Projection",
             "umap": "UMAP", "fs": "Feature Selection", "ae": "Autoencoder"}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
})

# ─────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────

def load_all_csvs(csv_dir):
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in: {csv_dir}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  {os.path.basename(f):45s}  ({len(df):,} rows)")
        except Exception as e:
            print(f"  [WARN] {f}: {e}")
    out = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(out):,} rows\n")
    return out


def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    rename = {"dimensionality": "dimension", "dim": "dimension",
              "kta": "final_alignment", "alignment": "final_alignment",
              "test_acc": "test_accuracy", "train_acc": "train_accuracy",
              "var": "kernel_variance", "eff_rank": "effective_rank",
              "wall_time": "wall_clock_time"}
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    df["dr_method"] = df["dr_method"].str.lower().str.strip()
    df["dataset"]   = df["dataset"].str.lower().str.strip()
    nums = ["dimension", "final_alignment", "test_accuracy", "train_accuracy",
            "balanced_test_accuracy", "balanced_train_accuracy", "kernel_variance",
            "effective_rank", "fro_norm", "centered_alignment", "generalization_gap",
            "wall_clock_time", "circuit_executions", "f1_score", "precision",
            "recall", "train_margin", "test_margin"]
    for c in nums:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────────────────────
# Drawing primitives
# ─────────────────────────────────────────────────────────────

def draw_lines(ax, sub, dims, col, dr_methods, ylabel="", title=""):
    """Plot all DR methods on one axis with shaded error bands (mean +/- std over seeds)."""
    for dr in dr_methods:
        c = DR_COLOR.get(dr, "gray")
        m = DR_MARKER.get(dr, "o")
        grp = sub[sub["dr_method"] == dr].groupby("dimension")[col]
        mn  = grp.mean().reindex(dims)
        sd  = grp.std().reindex(dims)
        ax.plot(dims, mn.values, color=c, marker=m, linewidth=2,
                markersize=6, label=DR_LABEL.get(dr, dr.upper()), zorder=3)
        ax.fill_between(dims,
                        mn.values - sd.fillna(0).values,
                        mn.values + sd.fillna(0).values,
                        color=c, alpha=0.12, zorder=2)
    ax.set_xticks(dims)
    ax.set_xlabel("Reduced Dimension d'")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")


def dr_legend(fig, dr_methods, ncol=None):
    handles = [
        plt.Line2D([0], [0], color=DR_COLOR.get(d, "gray"),
                   marker=DR_MARKER.get(d, "o"), linewidth=2,
                   markersize=6, label=DR_LABEL.get(d, d.upper()))
        for d in dr_methods
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=ncol or len(dr_methods),
               bbox_to_anchor=(0.5, -0.04), frameon=True)


# ─────────────────────────────────────────────────────────────
# PER-DATASET: Exp 1 — Accuracy & KTA vs d'
# ─────────────────────────────────────────────────────────────

def plot_exp1(sub, dims, dr_methods, dataset, out_dir):
    cols   = [c for c in ["test_accuracy", "final_alignment"] if c in sub.columns]
    labels = {"test_accuracy": "Test Accuracy", "final_alignment": "KTA (Alignment)"}
    if not cols:
        return

    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4.2))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        draw_lines(ax, sub, dims, col, dr_methods,
                   ylabel=labels[col], title=labels[col])
        if col == "test_accuracy":
            ax.set_ylim(0, 1.05)

    fig.suptitle(f"Exp 1 · Compression vs Performance   [{dataset.upper()}]",
                 fontsize=13, fontweight="bold")
    dr_legend(fig, dr_methods)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, os.path.join(out_dir, f"{dataset}_exp1_accuracy_kta.png"))


# ─────────────────────────────────────────────────────────────
# PER-DATASET: Exp 2 — DR Method Bar Comparison
# ─────────────────────────────────────────────────────────────

def plot_exp2(sub, dims, dr_methods, dataset, out_dir):
    metric_pairs = [(c, l) for c, l in
                    [("test_accuracy",   "Test Accuracy"),
                     ("final_alignment", "KTA (Alignment)")]
                    if c in sub.columns]
    if not metric_pairs:
        return

    n = len(metric_pairs)
    fig = plt.figure(figsize=(6.5 * n, 9))
    gs  = fig.add_gridspec(2, n, hspace=0.45, wspace=0.35)

    cmap_bar  = plt.cm.viridis(np.linspace(0.15, 0.85, len(dims)))
    width     = 0.8 / len(dims)
    x         = np.arange(len(dr_methods))
    xlabels   = [DR_LABEL.get(d, d.upper()) for d in dr_methods]

    for col_idx, (col, ylabel) in enumerate(metric_pairs):
        # grouped bar
        ax_bar = fig.add_subplot(gs[0, col_idx])
        for i, dim in enumerate(dims):
            means = [sub[(sub["dr_method"] == dr) & (sub["dimension"] == dim)][col].mean()
                     for dr in dr_methods]
            stds  = [sub[(sub["dr_method"] == dr) & (sub["dimension"] == dim)][col].std()
                     for dr in dr_methods]
            offset = (i - len(dims) / 2 + 0.5) * width
            ax_bar.bar(x + offset, means, width * 0.88, yerr=stds, capsize=3,
                       color=cmap_bar[i], label=f"d'={dim}", alpha=0.88,
                       error_kw={"elinewidth": 1.2})
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(xlabels, rotation=18, ha="right")
        ax_bar.set_ylabel(ylabel)
        ax_bar.set_title(f"{ylabel} by Method & Dimension", fontweight="bold")
        if col_idx == n - 1:
            ax_bar.legend(title="Dimension", fontsize=8, title_fontsize=8,
                          loc="upper right")

        # heatmap
        ax_hm = fig.add_subplot(gs[1, col_idx])
        pivot = np.array([
            [sub[(sub["dr_method"] == dr) & (sub["dimension"] == dim)][col].mean()
             for dim in dims]
            for dr in dr_methods
        ])
        valid = pivot[~np.isnan(pivot)]
        im = ax_hm.imshow(pivot, aspect="auto", cmap="RdYlGn",
                          vmin=valid.min() if len(valid) else 0,
                          vmax=valid.max() if len(valid) else 1)
        ax_hm.set_xticks(range(len(dims)))
        ax_hm.set_xticklabels(dims)
        ax_hm.set_yticks(range(len(dr_methods)))
        ax_hm.set_yticklabels(xlabels)
        ax_hm.set_xlabel("Dimension d'")
        ax_hm.set_title(f"{ylabel} Heatmap", fontweight="bold")
        plt.colorbar(im, ax=ax_hm, shrink=0.85, label=ylabel)
        for r in range(len(dr_methods)):
            for c2 in range(len(dims)):
                v = pivot[r, c2]
                if not np.isnan(v):
                    ax_hm.text(c2, r, f"{v:.3f}", ha="center", va="center",
                               fontsize=8.5,
                               color="white" if len(valid) and v > valid.mean() + 0.15 * valid.std() else "black")

    fig.suptitle(f"Exp 2 · DR Method Effect   [{dataset.upper()}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, os.path.join(out_dir, f"{dataset}_exp2_method_bars.png"))


# ─────────────────────────────────────────────────────────────
# PER-DATASET: Exp 3 — Kernel Expressivity
# ─────────────────────────────────────────────────────────────

def plot_exp3(sub, dims, dr_methods, dataset, out_dir):
    candidates = [
        ("kernel_variance", "Kernel Variance",  "Concentration down  (lower = more concentrated)"),
        ("effective_rank",  "Effective Rank",   "Expressivity up     (higher = richer kernel)"),
        ("fro_norm",        "Frobenius Norm",   "Kernel matrix scale"),
    ]
    avail = [(c, l, s) for c, l, s in candidates if c in sub.columns]
    if not avail:
        print(f"  [WARN] No expressivity metrics for {dataset}")
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(5.5 * len(avail), 4.2))
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, ylabel, subtitle) in zip(axes, avail):
        draw_lines(ax, sub, dims, col, dr_methods,
                   ylabel=ylabel, title=f"{ylabel}\n{subtitle}")

    fig.suptitle(f"Exp 3 · Kernel Expressivity   [{dataset.upper()}]",
                 fontsize=13, fontweight="bold")
    dr_legend(fig, dr_methods)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, os.path.join(out_dir, f"{dataset}_exp3_expressivity.png"))


# ─────────────────────────────────────────────────────────────
# PER-DATASET: Exp 4 — Alignment Stability
# ─────────────────────────────────────────────────────────────

def plot_exp4(sub, dims, dr_methods, dataset, out_dir):
    panels = []
    if "final_alignment" in sub.columns:
        std_tbl = (sub.groupby(["dr_method", "dimension"])["final_alignment"]
                   .std().reset_index().rename(columns={"final_alignment": "kta_std"}))
        panels.append(("kta_std", std_tbl, "KTA Std across Seeds",
                        "Alignment Stability (lower = more stable)"))
    if "generalization_gap" in sub.columns:
        panels.append(("generalization_gap", sub,
                        "Generalization Gap", "Train Acc - Test Acc (lower = better)"))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, data, ylabel, title) in zip(axes, panels):
        for dr in dr_methods:
            c = DR_COLOR.get(dr, "gray")
            m = DR_MARKER.get(dr, "o")
            grp = data[data["dr_method"] == dr].groupby("dimension")[col]
            mn  = grp.mean().reindex(dims)
            sd  = grp.std().reindex(dims)
            ax.plot(dims, mn.values, color=c, marker=m, linewidth=2,
                    markersize=6, label=DR_LABEL.get(dr, dr.upper()), zorder=3)
            if col != "kta_std":
                ax.fill_between(dims,
                                mn.values - sd.fillna(0).values,
                                mn.values + sd.fillna(0).values,
                                color=c, alpha=0.12, zorder=2)
        ax.set_xticks(dims)
        ax.set_xlabel("Reduced Dimension d'")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")

    fig.suptitle(f"Exp 4 · Alignment Stability & Generalisation   [{dataset.upper()}]",
                 fontsize=13, fontweight="bold")
    dr_legend(fig, dr_methods)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, os.path.join(out_dir, f"{dataset}_exp4_stability.png"))


# ─────────────────────────────────────────────────────────────
# OVERVIEW: heatmaps — dataset x DR method
# ─────────────────────────────────────────────────────────────

def plot_overview_heatmaps(df, datasets, dr_methods, out_dir):
    for col, label, fname in [
        ("test_accuracy",   "Mean Test Accuracy",  "overview_accuracy_heatmap.png"),
        ("final_alignment", "Mean KTA",            "overview_kta_heatmap.png"),
    ]:
        if col not in df.columns:
            continue
        pivot = np.array([
            [df[(df["dataset"] == ds) & (df["dr_method"] == dr)][col].mean()
             for dr in dr_methods]
            for ds in datasets
        ])
        valid = pivot[~np.isnan(pivot)]
        if not len(valid):
            continue

        fig, ax = plt.subplots(figsize=(max(5, len(dr_methods) * 1.6),
                                         max(3, len(datasets) * 0.85) + 1.5))
        im = ax.imshow(pivot, aspect="auto", cmap="YlGnBu",
                       vmin=valid.min(), vmax=valid.max())
        ax.set_xticks(range(len(dr_methods)))
        ax.set_xticklabels([DR_LABEL.get(d, d.upper()) for d in dr_methods],
                           rotation=20, ha="right")
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels([d.upper() for d in datasets])
        ax.set_xlabel("DR Method")
        ax.set_ylabel("Dataset")
        ax.set_title(f"Overview · {label}  (all dims & seeds averaged)",
                     fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)
        for r in range(len(datasets)):
            for c in range(len(dr_methods)):
                v = pivot[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=9,
                            color="white" if v > (valid.max() * 0.65) else "black")
        plt.tight_layout()
        save(fig, os.path.join(out_dir, fname))


# ─────────────────────────────────────────────────────────────
# OVERVIEW: best accuracy per dataset (grouped bar)
# ─────────────────────────────────────────────────────────────

def plot_overview_best_per_dataset(df, datasets, dr_methods, out_dir):
    if "test_accuracy" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(datasets) * 1.5), 5))
    x     = np.arange(len(datasets))
    width = 0.8 / len(dr_methods)
    cmap  = plt.cm.tab10(np.linspace(0, 0.8, len(dr_methods)))

    for i, dr in enumerate(dr_methods):
        vals = [df[(df["dataset"] == ds) & (df["dr_method"] == dr)]["test_accuracy"].max()
                for ds in datasets]
        offset = (i - len(dr_methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, color=cmap[i], alpha=0.88,
               label=DR_LABEL.get(dr, dr.upper()))

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], rotation=20, ha="right")
    ax.set_ylabel("Best Test Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Overview · Best Test Accuracy per Dataset & DR Method",
                 fontweight="bold")
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, "overview_best_per_dataset.png"))


# ─────────────────────────────────────────────────────────────
# OVERVIEW: radar chart — DR method profiles
# ─────────────────────────────────────────────────────────────

def plot_overview_radar(df, dr_methods, out_dir):
    radar_cols = ["test_accuracy", "final_alignment", "kernel_variance",
                  "effective_rank", "f1_score"]
    avail = [c for c in radar_cols if c in df.columns]
    if len(avail) < 3:
        return

    norm = df[["dr_method"] + avail].copy()
    for m in avail:
        mn, mx = norm[m].min(), norm[m].max()
        norm[m] = (norm[m] - mn) / (mx - mn) if mx > mn else 0.5
    if "kernel_variance" in norm.columns:
        norm["kernel_variance"] = 1 - norm["kernel_variance"]

    means = norm.groupby("dr_method")[avail].mean()

    labels_map = {"test_accuracy": "Test Acc", "final_alignment": "KTA",
                  "kernel_variance": "Kern Var\n(inv)", "effective_rank": "Eff Rank",
                  "f1_score": "F1 Score"}

    n   = len(avail)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang_c = ang + ang[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for dr in dr_methods:
        if dr not in means.index:
            continue
        vals = means.loc[dr, avail].tolist() + [means.loc[dr, avail[0]]]
        c = DR_COLOR.get(dr, "gray")
        ax.plot(ang_c, vals, color=c, linewidth=2,
                marker="o", markersize=5, label=DR_LABEL.get(dr, dr.upper()))
        ax.fill(ang_c, vals, color=c, alpha=0.07)

    ax.set_xticks(ang)
    ax.set_xticklabels([labels_map.get(m, m) for m in avail], size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Overview · DR Method Profiles (normalised)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=9)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, "overview_radar.png"))


# ─────────────────────────────────────────────────────────────
# OVERVIEW: compute scaling across datasets
# ─────────────────────────────────────────────────────────────

def plot_overview_compute(df, datasets, dr_methods, out_dir):
    if "wall_clock_time" not in df.columns:
        return

    dims = sorted(df["dimension"].unique())
    ncols = min(3, len(datasets))
    nrows = int(np.ceil(len(datasets) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4 * nrows),
                             sharey=False)
    axes_flat = np.array(axes).flatten() if len(datasets) > 1 else [axes]

    for ax, ds in zip(axes_flat, datasets):
        sub = df[df["dataset"] == ds]
        draw_lines(ax, sub, dims, "wall_clock_time", dr_methods,
                   ylabel="Wall-clock Time (s)", title=ds.upper())

    for ax in axes_flat[len(datasets):]:
        ax.set_visible(False)

    fig.suptitle("Overview · Compute Scaling — Wall-clock Time vs d'",
                 fontsize=13, fontweight="bold")
    dr_legend(fig, dr_methods)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    save(fig, os.path.join(out_dir, "overview_compute_scaling.png"))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir",  default=".",       help="Directory with CSV files")
    parser.add_argument("--out_dir",  default="figures", help="Output directory for plots")
    parser.add_argument("--datasets", nargs="*",         help="Filter to specific datasets")
    args = parser.parse_args()

    print("=" * 60)
    print("Quantum Kernel DR Experiment Plotter")
    print("=" * 60)
    print(f"\nLoading CSVs from: {args.csv_dir}\n")

    df = preprocess(load_all_csvs(args.csv_dir))

    datasets   = sorted(df["dataset"].unique())
    dr_methods = sorted(df["dr_method"].unique())

    if args.datasets:
        want = [d.lower() for d in args.datasets]
        datasets = [d for d in datasets if d in want]

    print(f"Datasets   : {datasets}")
    print(f"DR methods : {dr_methods}")
    print(f"Dimensions : {sorted(df['dimension'].unique())}")
    print(f"Seeds      : {sorted(df['seed'].unique()) if 'seed' in df.columns else 'N/A'}")
    print(f"Output dir : {args.out_dir}\n")

    # Per-dataset (same 4 charts for every dataset)
    for ds in datasets:
        print(f"{'─'*55}")
        print(f"  {ds.upper()}")
        print(f"{'─'*55}")
        sub  = df[df["dataset"] == ds]
        dims = sorted(sub["dimension"].unique())
        drs  = sorted(sub["dr_method"].unique())

        plot_exp1(sub, dims, drs, ds, args.out_dir)
        plot_exp2(sub, dims, drs, ds, args.out_dir)
        plot_exp3(sub, dims, drs, ds, args.out_dir)
        plot_exp4(sub, dims, drs, ds, args.out_dir)

    # Cross-dataset overviews
    print(f"\n{'─'*55}")
    print("  OVERVIEW charts")
    print(f"{'─'*55}")
    plot_overview_heatmaps(df, datasets, dr_methods, args.out_dir)
    plot_overview_best_per_dataset(df, datasets, dr_methods, args.out_dir)
    plot_overview_radar(df, dr_methods, args.out_dir)
    plot_overview_compute(df, datasets, dr_methods, args.out_dir)

    print(f"\nDone — all plots in: {args.out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()