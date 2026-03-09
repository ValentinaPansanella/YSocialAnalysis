import argparse
import json
import os
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_new import discover_runs, get_db_path, pids_to_list
from utils_figures_new import apply_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute recommendation-count distributions per run and aggregate across runs."
    )
    parser.add_argument("--base-dirs", nargs="+", default=["data"])
    parser.add_argument("--res-dir", default="res")
    parser.add_argument("--log-file", default="logs/recs_distribution.log")
    parser.add_argument("--networks", nargs="+", default=None)
    parser.add_argument("--recsyss", nargs="+", default=None)
    parser.add_argument("--runs", nargs="+", type=int, default=None)
    parser.add_argument("--skip-run-plots", action="store_true")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _coerce_post_ids(recommendations: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    # Keep numeric IDs numeric when possible; fall back to string matching.
    post_numeric = pd.to_numeric(posts["post_id"], errors="coerce")
    if post_numeric.notna().all():
        recommendations["post_id"] = pd.to_numeric(recommendations["post_id"], errors="coerce")
        recommendations = recommendations.dropna(subset=["post_id"])
        recommendations["post_id"] = recommendations["post_id"].astype(int)
        posts["post_id"] = post_numeric.astype(int)
    else:
        recommendations["post_id"] = recommendations["post_id"].astype(str)
        posts["post_id"] = posts["post_id"].astype(str)

    return recommendations


def plot_run_distribution(
    frequencies: pd.DataFrame,
    base_name: str,
    network: str,
    recsys: str,
    run: int,
    res_dir: str,
) -> None:
    frequencies = frequencies.copy()
    frequencies["prob"] = frequencies["num_entities"] / frequencies["num_entities"].sum()

    apply_plot_style()

    fig_dir = os.path.join(res_dir, base_name, "recs_distribution")
    ensure_dir(fig_dir)

    fig, axes = plt.subplots(figsize=(12, 4), ncols=2)

    ax = axes[0]
    ax.bar(frequencies["num_recommendations"], frequencies["prob"], color="skyblue")
    ax.set_xlabel("Number of recommendations")
    ax.set_ylabel("Probability")

    ax = axes[1]
    valid = frequencies[(frequencies["num_recommendations"] > 0) & (frequencies["prob"] > 0)]
    if not valid.empty:
        ax.scatter(valid["num_recommendations"], valid["prob"], color="red")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Number of recommendations")
    ax.set_ylabel("Probability")

    plt.suptitle(f"Distribution for {network}-{recsys}-run{run}")
    plt.tight_layout()
    out_path = os.path.join(fig_dir, f"{network}-{recsys}-{run}-num-recs-prob.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def load_run_distribution(base_dir: str, network: str, recsys: str, run: int) -> Optional[pd.DataFrame]:
    db_path = get_db_path(base_dir, network, recsys, run)
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        recommendations = pd.read_sql(
            "SELECT post_ids FROM recommendations",
            conn,
        )
        posts = pd.read_sql(
            "SELECT id AS post_id FROM post",
            conn,
        )

    if recommendations.empty or posts.empty:
        return None

    recommendations["post_id"] = recommendations["post_ids"].apply(pids_to_list)
    recommendations = recommendations.explode("post_id").dropna(subset=["post_id"])

    recommendations = _coerce_post_ids(recommendations, posts)
    valid_posts = set(posts["post_id"].tolist())
    recommendations = recommendations[recommendations["post_id"].isin(valid_posts)]

    if recommendations.empty:
        return None

    post_counts = recommendations["post_id"].value_counts()
    freq = (
        post_counts.value_counts()
        .sort_index()
        .rename_axis("num_recommendations")
        .reset_index(name="num_entities")
    )

    return freq


def compute_global_distribution(all_runs: Sequence[pd.DataFrame]) -> Optional[Dict]:
    all_counts: List[int] = []

    for rc in all_runs:
        repeated = np.repeat(rc["num_recommendations"].to_numpy(), rc["num_entities"].to_numpy())
        all_counts.extend(repeated.tolist())

    if not all_counts:
        return None

    unique, counts = np.unique(all_counts, return_counts=True)
    probs = counts / counts.sum()

    return {"bins": unique.tolist(), "probs": probs.tolist()}


def compute_mean_distribution(all_runs: Sequence[pd.DataFrame]) -> Optional[Dict]:
    if not all_runs:
        return None

    max_count = int(max(rc["num_recommendations"].max() for rc in all_runs))
    common_x = np.arange(1, max_count + 1)

    aligned = []
    for rc in all_runs:
        counts = (
            rc.set_index("num_recommendations")["num_entities"]
            .reindex(common_x, fill_value=0)
            .to_numpy()
        )
        total = counts.sum()
        if total == 0:
            continue
        aligned.append(counts / total)

    if not aligned:
        return None

    arr = np.array(aligned)
    
    
    low = np.nanpercentile(arr, 25, axis=0)
    high = np.nanpercentile(arr, 75, axis=0)
    median = np.nanmedian(arr, axis=0)
    
    return {
        'bins': common_x.tolist(),
        'median': median.tolist(),
        'low': low.tolist(),
        'high': high.tolist()
    }
    # return {
    #     "bins": common_x.tolist(),
    #     "mean": np.mean(arr, axis=0).tolist(),
    #     "std": np.std(arr, axis=0).tolist(),
    # }


def compute_all_distributions(
    base_dir: str,
    runs_specs: Sequence[Tuple[str, str, int]],
    res_dir: str,
    skip_run_plots: bool,
) -> Tuple[Dict, Dict]:
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for network, recsys, run in runs_specs:
        grouped[(network, recsys)].append(run)

    global_metrics: Dict = defaultdict(dict)
    mean_metrics: Dict = defaultdict(dict)

    total_steps = len(runs_specs)
    pbar = tqdm(total=total_steps, desc=f"Processing runs in {base_dir}")

    base_name = os.path.basename(os.path.normpath(base_dir))

    for (network, recsys), runs in sorted(grouped.items()):
        all_runs = []
        used_runs = []

        for run in sorted(runs):
            rc = load_run_distribution(base_dir, network, recsys, run)
            pbar.update(1)
            if rc is None or rc.empty:
                continue

            if not skip_run_plots:
                plot_run_distribution(rc, base_name, network, recsys, run, res_dir)
            all_runs.append(rc)
            used_runs.append(run)

        if not all_runs:
            continue

        global_data = compute_global_distribution(all_runs)
        mean_data = compute_mean_distribution(all_runs)

        if global_data is not None:
            global_data["runs_used"] = used_runs
            global_metrics[network][recsys] = global_data

        if mean_data is not None:
            mean_data["runs_used"] = used_runs
            mean_metrics[network][recsys] = mean_data

    pbar.close()
    return global_metrics, mean_metrics


def save_metrics(base_dir: str, res_dir: str, global_m: Dict, mean_m: Dict) -> None:
    base_name = os.path.basename(os.path.normpath(base_dir))
    out_dir = os.path.join(res_dir, base_name, "recs_distribution")
    ensure_dir(out_dir)

    with open(os.path.join(out_dir, "global_distribution_metrics.json"), "w") as f:
        json.dump(global_m, f, indent=2)

    with open(os.path.join(out_dir, "mean_distribution_metrics.json"), "w") as f:
        json.dump(mean_m, f, indent=2)


def main() -> None:
    args = parse_args()
    ensure_dir("logs")

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

            global_m, mean_m = compute_all_distributions(
                base_dir=base_dir,
                runs_specs=specs,
                res_dir=args.res_dir,
                skip_run_plots=args.skip_run_plots,
            )
            save_metrics(base_dir, args.res_dir, global_m, mean_m)

            summary = (
                f"Saved rec distribution metrics for {base_dir}: "
                f"{sum(len(v) for v in global_m.values())} (network,recsys) pairs\n"
            )
            print(summary.strip())
            log.write(summary)


if __name__ == "__main__":
    main()
