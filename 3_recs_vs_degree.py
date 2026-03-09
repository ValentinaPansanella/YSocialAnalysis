import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # MUST be before importing pyplot
import matplotlib.pyplot as plt
import gc # Import garbage collector
from typing import Sequence, Hashable, Optional


from utils import get_db_path, pids_to_list, compute_author_degrees
from utils_figures import apply_plot_style

BASE_DIRS = ['data']
NETWORKS = ['BA', 'ER']
RECSYSS = ['F', 'RC', 'P', 'FP', 'hybrid']
RUNS = list(range(10))

LOG_FILE = "logs/recs_vs_degree.log"
os.makedirs("logs", exist_ok=True)

import matplotlib.pyplot as plt

def create_bins(degrees, n_bins=20, strategy="log"):
    degrees = np.array(degrees)
    min_deg = max(degrees.min(), 1)  # avoid log10(0)
    max_deg = degrees.max()
    
    if strategy == "log":
        bins = np.logspace(np.log10(min_deg), np.log10(max_deg)*1.01, n_bins)
        bins = np.unique(bins)
    elif strategy == "linear":
        bins = np.linspace(min_deg, max_deg*1.01, n_bins)
        bins = np.unique(bins)
    elif strategy == "unique":
        bins = np.sort(np.unique(degrees))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return bins


def plot(
    df,
    base_dir,
    network,
    recsys,
    run,
    out_dir=None,
    mode="scatter_binned",   # scatter | hexbin | binned | scatter_binned | all
    bins=20
):

    apply_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    metrics = ["recommendations", "reach", "origin", "views", "delta"]

    formulas = {
        "recommendations": r"$\mathrm{recommendations}(u)=\sum_{p\in P(u)}\mathrm{rec}(p)$",
        "reach": r"$\mathrm{reach}(u)=|\{v:\exists p\in P(u)\ \mathrm{shown\ to}\ v\}|$",
        "origin": r"$\mathrm{origin}(u)=\sum_{p}\mathbf{1}(u\ \mathrm{viewed}\ p)$",
        "views": r"$\mathrm{views}(u)=|\{a:\exists p_a\ \mathrm{shown\ to}\ u\}|$",
        "delta": r"$\Delta(u)=\mathrm{reach}(u)-\mathrm{views}(u)$"
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # helper: compute log-binned stats
    def compute_binned(df, xcol, ycol, bins):
        """
        Compute mean and std of y for each x-bin
        """
        if bins is None:
            # per-degree no binning
            df_grouped = df.groupby(xcol).agg(
                x_mean=(xcol, "mean"),
                y_mean=(ycol, "mean"),
                y_std=(ycol, "std"),
                n=(ycol, "size")
            ).reset_index(drop=True)
        else:
            # bin degrees
            df["bin"] = pd.cut(df[xcol], bins=bins, include_lowest=True)
            df_grouped = df.groupby("bin").agg(
                x_mean=(xcol, "mean"),
                y_mean=(ycol, "mean"),
                y_std=(ycol, "std"),
                n=(ycol, "size")
            ).reset_index(drop=True)
        return df_grouped

    for i, metric in enumerate(metrics):

        ax = axes[i]

        x = df['author_degree']
        y = df[metric]

        # -----------------------
        # SCATTER
        # -----------------------
        if mode in ["scatter", "scatter_binned", "all"]:

            alpha = 0.6 if mode == "scatter" else 0.1

            ax.scatter(
                x,
                y,
                s=10,
                alpha=alpha
            )

        # -----------------------
        # HEXBIN
        # -----------------------
        if mode in ["hexbin", "all"]:

            hb = ax.hexbin(
                x,
                y,
                gridsize=40,
                bins="log",
                mincnt=1
            )

        # -----------------------
        # BINNED MEAN + STD
        # -----------------------
        if mode in ["binned", "scatter_binned", "all"]:
            # create bins only if needed
            if bins is not None and isinstance(bins, int):
                bins_arr = create_bins(df["author_degree"], n_bins=bins, strategy="log")
            else:
                bins_arr = bins

            grouped = compute_binned(df, "author_degree", metric, bins=bins_arr)

            ax.errorbar(
                grouped["x_mean"],
                grouped["y_mean"],
                yerr=grouped["y_std"],
                fmt="o",
                capsize=3,
                linewidth=1.5
            )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("degree")
        ax.set_ylabel(metric)
        ax.set_title(metric)

        # ---- formula annotation ----
        ax.text(
            0.95,
            0.95,
            formulas[metric],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

    axes[-1].axis("off")

    fig.suptitle(f"{base_dir} | {network} | {recsys} | run {run}")

    plt.tight_layout()

    filename = f"{network}-{recsys}-{run}-degree-analysis-{mode}.png"
    path = os.path.join(out_dir, filename)

    #print(f'Saving plots to {path}')
    plt.savefig(path, dpi=150)
    # Explicitly clear and close the specific figure to free RAM
    fig.clf()
    plt.close(fig)
    gc.collect()

def compute_metric(
    df: pd.DataFrame,
    group_by: Hashable,
    count_col: Hashable,
    all_users: Sequence[Hashable],
    count_unique: bool = False,
) -> pd.Series:
    """
    Generic metric computation.

    Parameters
    ----------
    df : DataFrame
    group_by : column to group on (e.g., 'author_id', 'viewer_id')
    count_col : column to count (e.g., 'post_id', 'viewer_id', 'author_id')
    all_users : list of all user_ids to ensure full index
    count_unique : bool
    normalization : bool (relative frequency within group)

    Returns
    -------
    Series indexed by user_id
    """
    
    grouped = df.groupby(group_by)

    if count_unique:
        counts = grouped[count_col].nunique()
    else:
        counts = grouped[count_col].count()

    # Ensure all users appear
    counts = counts.reindex(all_users, fill_value=0)

    return counts

# ==========================================================
# RUN ANALYSIS
# ==========================================================
def run_analysis(
    base_dir: str,
    network: str,
    recsys: str,
    run: int,
    normalize_recommendations: bool = True
) -> Optional[pd.DataFrame]:

    db_path = get_db_path(base_dir, network, recsys, run)
    if not os.path.exists(db_path):
        return None
    
    #print(f'Processing file {db_path}')

    # --- Load degrees ---
    degrees = compute_author_degrees(base_dir, network, recsys, run, by="id")
    all_users = list(degrees.keys())

    with sqlite3.connect(db_path) as conn:
        # Optimization: Do NOT load the 'round' column since it is unused
        df_rec = pd.read_sql(
            "SELECT post_ids, user_id AS viewer_id FROM recommendations",
            conn,
        )
        df_posts = pd.read_sql(
            "SELECT id AS post_id, user_id AS author_id FROM post",
            conn,
        )

    # --- Explode recommendations ---
    df_rec["post_id"] = df_rec["post_ids"].apply(pids_to_list)
    df_rec = df_rec.explode("post_id").drop(columns=["post_ids"])

    df = df_rec.merge(df_posts, on="post_id", how="left")
    
    # Free up memory immediately
    del df_rec 
    del df_posts
    gc.collect()

    df = df.dropna(subset=["author_id"])

    # AUTHOR SIDE
    recommendations = compute_metric(df, "author_id", "post_id", all_users)
    reach = compute_metric(df, "author_id", "viewer_id", all_users, True)

    # VIEWER SIDE
    origin = compute_metric(df, "viewer_id", "post_id", all_users)
    views = compute_metric(df, "viewer_id", "author_id", all_users, True)

    # Free the massive merged dataframe
    del df
    gc.collect()

    # --------------------------------------------------
    # Assemble final dataframe
    # --------------------------------------------------
    stats = pd.DataFrame({
        "user_id": all_users,
        "author_degree": [degrees[u] for u in all_users],
        "recommendations": recommendations.values,
        "reach": reach.values,
        "origin": origin.values,
        "views": views.values,
    })

    stats["delta"] = stats["reach"] - stats["views"]
    
    #print(f'Stats dataframe computed: {stats.head()}')
    
    fig_dir = f'res/{base_dir}/recs_vs_degree/'
    plot(stats, base_dir, network, recsys, run, out_dir=fig_dir, mode="scatter_binned")

    return stats

# ==========================================================
# BINNING FUNCTION
# ==========================================================

def bin_metrics(df, bins, metrics):

    df = df.copy()
    df["degree_bin"] = pd.cut(df["author_degree"], bins=bins, include_lowest=True)
    #print('after adding degree_bin column tha dataframe looks like:')
    #print(df.head())

    result = {}

    for m in metrics:
        non_empty_bins = df.bin.dropna().unique()
        result[m] = df.groupby('bin')[m].mean().reindex(non_empty_bins).values
        #print(f'for metric {m} i computed:')
        #print(result[m])
    
    return result

# ==========================================================
# AGGREGATE METRICS OVER RUNS
# ==========================================================
def aggregate_runs(base_dir, networks=NETWORKS, recsyss=RECSYSS, runs=RUNS, n_bins=20):
    
    #print('Aggregating runs...')

    metrics = ["recommendations", "reach", "origin", "views", "delta"]

    results = defaultdict(dict)

    total_steps = len(networks)*len(recsyss)*len(runs)
    pbar = tqdm(total=total_steps)

    for network in networks:
        if base_dir == 'old_data':
            recsyss = ['RC', 'F']
        for recsys in recsyss:

            run_dfs = []

            for run in runs:

                df = run_analysis(base_dir, network, recsys, run)
                
                if df is not None:
                    run_dfs.append(df)

                pbar.update(1)

            if not run_dfs:
                continue
            
            #print(run_dfs[0])

            # -----------------------------------------
            # COMMON DEGREE BINS
            # -----------------------------------------
            #print('Computing bins...')
            all_degrees = np.concatenate([df.author_degree.values for df in run_dfs])
            bins = create_bins(all_degrees, n_bins=20, strategy="log")
            #print(bins)

            # -----------------------------------------
            # PER RUN CURVES
            # -----------------------------------------
            #print(f'metrics: {metrics}')
            #print('Computing per run curves:')
            per_run = {m: [] for m in metrics}
            #print('per_run when itialized is:')
            #print(per_run)

            for df in run_dfs:
                #print('for df in run_dfs produces:')
                #print(df.head())
                binned = bin_metrics(df, bins, metrics)
                #print('binned:')
                #print(binned)

                for m in metrics:
                    per_run[m].append(binned[m])
            
            #print('per_run:')
            #print(per_run)

            # -----------------------------------------
            # MEAN ± STD
            # -----------------------------------------
            #print('Computing MEAN ± STD:')
            mean_std = {}

            for m in metrics:

                arr = np.array(per_run[m])
                #print(f'array created from per_run[{m}]:')
                #print(arr)

                mean_std[m] = {
                    "mean": np.nanmean(arr, axis=0).tolist(),
                    "std": np.nanstd(arr, axis=0).tolist()
                }
            #print(mean_std)
            # -----------------------------------------
            # GLOBAL POOLED DATA
            # -----------------------------------------
            #print('Computing concatenated distribs:')
            pooled_df = pd.concat(run_dfs, ignore_index=True)

            global_metrics = bin_metrics(pooled_df, bins, metrics)
            #print(global_metrics)

            # -----------------------------------------
            # STORE
            # -----------------------------------------

            results[network][recsys] = {
                "bins": bins.tolist(),
                "per_run": {m: np.array(per_run[m]).tolist() for m in metrics},
                "mean_std": mean_std,
                "global": {m: global_metrics[m].tolist() for m in metrics}
            }

    pbar.close()
    #print(results)
    return results

# ==========================================================
# SAVE
# ==========================================================

def save_metrics(base_dir, metrics_dict):

    res_dir = f"res/{base_dir}/recs_vs_degree/"
    os.makedirs(res_dir, exist_ok=True)
    
    #print(f'saving metrics to: {os.path.join(res_dir, "metrics_vs_degree.json")}')
    with open(os.path.join(res_dir, "metrics_vs_degree.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)
        
# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    old_stdout = sys.stdout
    log_file = open(LOG_FILE, "w")
    sys.stdout = log_file

    for base_dir in BASE_DIRS:

        #print(f"Processing {base_dir}")

        results = aggregate_runs(base_dir)

        save_metrics(base_dir, results)

    sys.stdout = old_stdout
    log_file.close()