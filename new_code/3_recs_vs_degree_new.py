import argparse
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# Safer defaults for headless/limited server environments.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_new import compute_author_degrees, discover_runs, get_db_path, pids_to_list
from utils_figures_new import apply_plot_style


METRICS = ["recommendations", "reach", "origin", "views", "delta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute recommendation-exposure metrics as a function of author degree."
    )
    parser.add_argument("--base-dirs", nargs="+", default=["data"])
    parser.add_argument("--res-dir", default="res")
    parser.add_argument("--log-file", default="logs/recs_vs_degree.log")
    parser.add_argument("--networks", nargs="+", default=None)
    parser.add_argument("--recsyss", nargs="+", default=None)
    parser.add_argument("--runs", nargs="+", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument(
        "--run-plots",
        action="store_true",
        help="Enable per-run plot generation (disabled by default for efficiency).",
    )
    parser.add_argument(
        "--skip-run-plots",
        action="store_true",
        help="Backward-compatible alias; run plots are skipped by default.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_bins(degrees: np.ndarray, n_bins: int = 20) -> np.ndarray:
    degrees = np.asarray(degrees, dtype=float)
    degrees = degrees[np.isfinite(degrees)]
    degrees = degrees[degrees > 0]

    if degrees.size == 0:
        return np.array([])

    min_deg = float(np.min(degrees))
    max_deg = float(np.max(degrees))

    if min_deg == max_deg:
        return np.array([min_deg, min_deg * 1.01 + 1e-9])

    bins = np.logspace(np.log10(min_deg), np.log10(max_deg) * 1.001, n_bins)
    bins = np.unique(bins)

    if bins.size < 2:
        return np.array([min_deg, max_deg * 1.01])

    return bins


def load_run_stats(base_dir: str, network: str, recsys: str, run: int) -> Optional[pd.DataFrame]:
    db_path = get_db_path(base_dir, network, recsys, run)
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM user_mgmt")
        all_users = [row[0] for row in cur.fetchall()]
        if not all_users:
            return None

        cur.execute("SELECT id, user_id FROM post")
        post_rows = cur.fetchall()

    if not all_users:
        return None

    degrees = compute_author_degrees(base_dir, network, recsys, run, by="id")
    post_to_author: Dict = {}
    for post_id, author_id in post_rows:
        post_to_author[post_id] = author_id
        post_to_author[str(post_id)] = author_id

    recommendations_count = defaultdict(int)
    origin_count = defaultdict(int)
    reach_sets = defaultdict(set)
    views_sets = defaultdict(set)

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id, post_ids FROM recommendations")
        for viewer_id, post_ids_raw in cur:
            post_ids = pids_to_list(post_ids_raw)
            if not post_ids:
                continue

            for post_id in post_ids:
                author_id = post_to_author.get(post_id)
                if author_id is None:
                    author_id = post_to_author.get(str(post_id))
                if author_id is None:
                    continue

                recommendations_count[author_id] += 1
                origin_count[viewer_id] += 1
                reach_sets[author_id].add(viewer_id)
                views_sets[viewer_id].add(author_id)

    n_users = len(all_users)
    recommendations_arr = np.fromiter(
        (recommendations_count.get(user_id, 0) for user_id in all_users), dtype=np.int64, count=n_users
    )
    reach_arr = np.fromiter(
        (len(reach_sets.get(user_id, ())) for user_id in all_users), dtype=np.int64, count=n_users
    )
    origin_arr = np.fromiter(
        (origin_count.get(user_id, 0) for user_id in all_users), dtype=np.int64, count=n_users
    )
    views_arr = np.fromiter(
        (len(views_sets.get(user_id, ())) for user_id in all_users), dtype=np.int64, count=n_users
    )
    degrees_arr = np.fromiter(
        (int(degrees.get(user_id, 0)) for user_id in all_users), dtype=np.int64, count=n_users
    )

    stats = pd.DataFrame(
        {
            "user_id": all_users,
            "author_degree": degrees_arr,
            "recommendations": recommendations_arr,
            "reach": reach_arr,
            "origin": origin_arr,
            "views": views_arr,
        }
    )
    stats["delta"] = stats["reach"] - stats["views"]
    return stats


def _compute_binned(df: pd.DataFrame, metric: str, bins: np.ndarray) -> pd.DataFrame:
    binned = df.copy()
    binned["degree_bin"] = pd.cut(binned["author_degree"], bins=bins, include_lowest=True)

    grouped = (
        binned.groupby("degree_bin", observed=False)
        .agg(x_mean=("author_degree", "mean"), y_mean=(metric, "mean"), y_std=(metric, "std"))
    )
    return grouped


def plot_run(
    df: pd.DataFrame,
    base_name: str,
    network: str,
    recsys: str,
    run: int,
    out_dir: str,
    n_bins: int,
) -> None:
    apply_plot_style()
    ensure_dir(out_dir)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    base_plot_df = df[df["author_degree"] > 0]
    positive_degrees = base_plot_df["author_degree"].to_numpy()
    bins = create_bins(positive_degrees, n_bins=n_bins)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]

        plot_df = base_plot_df

        if metric != "delta":
            plot_df = plot_df[plot_df[metric] > 0]
        else:
            plot_df = plot_df[plot_df[metric] != 0]

        if plot_df.empty:
            ax.set_title(metric)
            ax.set_axis_off()
            continue

        ax.scatter(plot_df["author_degree"], plot_df[metric], s=8, alpha=0.12)

        if bins.size >= 2:
            grouped = _compute_binned(plot_df, metric, bins)
            x = grouped["x_mean"].to_numpy()
            y = grouped["y_mean"].to_numpy()
            y_std = grouped["y_std"].fillna(0).to_numpy()

            valid = np.isfinite(x) & np.isfinite(y)
            if metric != "delta":
                valid &= y > 0

            if np.any(valid):
                ax.errorbar(x[valid], y[valid], yerr=y_std[valid], fmt="o", capsize=2, linewidth=1.0)

        ax.set_xscale("log")
        if metric == "delta":
            ax.set_yscale("symlog", linthresh=1.0)
        else:
            ax.set_yscale("log")

        ax.set_xlabel("degree")
        ax.set_ylabel(metric)
        ax.set_title(metric)

    axes[-1].axis("off")
    fig.suptitle(f"{base_name} | {network} | {recsys} | run {run}")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{network}-{recsys}-{run}-degree-analysis.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def _nan_to_none(values: np.ndarray) -> List[Optional[float]]:
    result: List[Optional[float]] = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            result.append(None)
        else:
            result.append(float(value))
    return result


def bin_metrics(df: pd.DataFrame, bins: np.ndarray, metrics: Sequence[str]) -> Dict[str, np.ndarray]:
    df = df.copy()
    df["degree_bin"] = pd.cut(df["author_degree"], bins=bins, include_lowest=True)

    result: Dict[str, np.ndarray] = {}
    grouped = df.groupby("degree_bin", observed=False)

    for metric in metrics:
        values = grouped[metric].mean()
        values = values.reindex(df["degree_bin"].cat.categories)
        result[metric] = values.to_numpy(dtype=float)

    return result


def aggregate_runs(
    base_dir: str,
    run_specs: Sequence[Tuple[str, str, int]],
    res_dir: str,
    n_bins: int,
    skip_run_plots: bool,
) -> Dict:
    base_name = os.path.basename(os.path.normpath(base_dir))
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for network, recsys, run in run_specs:
        grouped[(network, recsys)].append(run)

    results: Dict = defaultdict(dict)

    pbar = tqdm(total=len(run_specs), desc=f"Processing runs in {base_dir}")

    for (network, recsys), runs in sorted(grouped.items()):
        run_dfs = []
        used_runs: List[int] = []

        for run in sorted(runs):
            df = load_run_stats(base_dir, network, recsys, run)
            pbar.update(1)
            if df is None or df.empty:
                continue

            if not skip_run_plots:
                out_dir = os.path.join(res_dir, base_name, "recs_vs_degree")
                plot_run(df, base_name, network, recsys, run, out_dir, n_bins=n_bins)

            run_dfs.append(df)
            used_runs.append(run)

        if not run_dfs:
            continue

        all_degrees = np.concatenate([df["author_degree"].to_numpy() for df in run_dfs])
        bins = create_bins(all_degrees, n_bins=n_bins)
        if bins.size < 2:
            continue

        per_run = {metric: [] for metric in METRICS}
        for df in run_dfs:
            binned = bin_metrics(df, bins, METRICS)
            for metric in METRICS:
                per_run[metric].append(binned[metric])

        pooled_df = pd.concat(run_dfs, ignore_index=True)
        global_metrics = bin_metrics(pooled_df, bins, METRICS)

        metric_payload: Dict = {}
        for metric in METRICS:
            arr = np.array(per_run[metric], dtype=float)
            mean = np.empty(arr.shape[1], dtype=float)
            std = np.empty(arr.shape[1], dtype=float)

            for idx in range(arr.shape[1]):
                col = arr[:, idx]
                valid = col[np.isfinite(col)]
                if valid.size == 0:
                    mean[idx] = np.nan
                    std[idx] = np.nan
                else:
                    mean[idx] = float(np.mean(valid))
                    std[idx] = float(np.std(valid))
            global_vals = global_metrics[metric]

            metric_payload[metric] = {
                "per_run": [_nan_to_none(row) for row in arr],
                "mean": _nan_to_none(mean),
                "std": _nan_to_none(std),
                "global": _nan_to_none(global_vals),
            }

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        results[network][recsys] = {
            "runs_used": used_runs,
            "bin_edges": bins.tolist(),
            "bin_centers": bin_centers.tolist(),
            "metrics": metric_payload,
        }

    pbar.close()
    return results


def save_metrics(base_dir: str, res_dir: str, metrics_dict: Dict) -> str:
    base_name = os.path.basename(os.path.normpath(base_dir))
    out_dir = os.path.join(res_dir, base_name, "recs_vs_degree")
    ensure_dir(out_dir)

    out_path = os.path.join(out_dir, "metrics_vs_degree.json")
    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    return out_path


def main() -> None:
    args = parse_args()
    ensure_dir("logs")
    run_plots_enabled = args.run_plots and not args.skip_run_plots

    with open(args.log_file, "w") as log:
        for base_dir in args.base_dirs:
            specs = discover_runs(
                base_dir,
                networks=args.networks,
                recsyss=args.recsyss,
                runs=args.runs,
            )

            if not specs:
                message = f"No runs found in {base_dir}\n"
                print(message.strip())
                log.write(message)
                continue

            header = f"Processing {base_dir} ({len(specs)} runs)"
            print(header)
            log.write(header + "\n")

            results = aggregate_runs(
                base_dir=base_dir,
                run_specs=specs,
                res_dir=args.res_dir,
                n_bins=args.n_bins,
                skip_run_plots=False,
            )
            out_path = save_metrics(base_dir, args.res_dir, results)

            summary = (
                f"Saved metrics-vs-degree for {base_dir}: "
                f"{sum(len(v) for v in results.values())} (network,recsys) pairs at {out_path}\n"
            )
            print(summary.strip())
            log.write(summary)


if __name__ == "__main__":
    main()
