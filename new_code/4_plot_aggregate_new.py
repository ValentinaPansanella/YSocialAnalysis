import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from utils_figures_new import apply_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregate figures from outputs of scripts 2 and 3."
    )
    parser.add_argument("--base-dirs", nargs="+", default=["data"])
    parser.add_argument("--res-dir", default="res")
    parser.add_argument("--log-file", default="logs/plot_aggregate.log")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _to_array(values: List) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def plot_global_distribution(base_name: str, res_dir: str) -> bool:
    path = os.path.join(res_dir, base_name, "recs_distribution", "global_distribution_metrics.json")
    metrics = _load_json(path)
    if not metrics:
        print(f"Skipping {base_name}: global distribution file not found")
        return False

    apply_plot_style()
    plt.figure(figsize=(8, 5))

    plotted = 0
    for network in sorted(metrics.keys()):
        for recsys in sorted(metrics[network].keys()):
            data = metrics[network][recsys]
            if not data:
                continue
            x = np.array(data.get("bins", []), dtype=float)
            y = np.array(data.get("probs", []), dtype=float)

            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if not np.any(valid):
                continue

            plt.plot(x[valid], y[valid], marker="o", markersize=3, label=f"{network}-{recsys}")
            plotted += 1

    if plotted == 0:
        plt.close()
        print(f"Skipping {base_name}: no valid global distribution data")
        return False

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of recommendations")
    plt.ylabel("P(x)")
    plt.legend(fontsize=7, ncol=2)

    out_dir = os.path.join(res_dir, base_name, "recs_distribution")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "sum_over_runs.png")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return True


def plot_mean_distribution(base_name: str, res_dir: str) -> bool:
    path = os.path.join(res_dir, base_name, "recs_distribution", "mean_distribution_metrics.json")
    metrics = _load_json(path)
    if not metrics:
        print(f"Skipping {base_name}: mean distribution file not found")
        return False

    apply_plot_style()
    
    fig, axes = plt.subplots(figsize=(8, 5), ncols=2, sharex=True, sharey=True)

    plotted = 0
    for i, network in enumerate(sorted(metrics.keys())):
        ax = axes[i]
        for recsys in sorted(metrics[network].keys()):
            data = metrics[network][recsys]
            if not data:
                continue

            x = np.array(data.get("bins", []), dtype=float)
            # y = np.array(data.get("mean", []), dtype=float)
            # std = np.array(data.get("std", []), dtype=float)
            y = np.array(data.get("median", []), dtype=float)
            low = np.array(data.get("low", []), dtype=float)
            high = np.array(data.get("high", []), dtype=float)

            valid &= (y - low) > 0
            if not np.any(valid):
                continue

            xx = x[valid]
            yy = y[valid]

            low = yy - low[valid]
            high = yy + low[valid]

            ax.plot(xx, yy, marker="o", markersize=3, label=f"{network}-{recsys}")
            ax.fill_between(xx, low, high, alpha=0.2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Number of recommendations")
            if i == 0:
                ax.set_ylabel("P(x)")
            else:
                ax.set_ylabel("")
            ax.legend(fontsize=7, ncol=1)
            plotted += 1

    if plotted == 0:
        plt.close()
        print(f"Skipping {base_name}: no valid mean distribution data")
        return False

    plt.suptitle("Recommendation Count Distribution Across Runs (Median ± IQR)")
    
    out_dir = os.path.join(res_dir, base_name, "recs_distribution")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "median_over_runs.png")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return True


def _available_metrics(metrics_payload: Dict) -> List[str]:
    metric_names = set()
    for network_data in metrics_payload.values():
        for recsys_data in network_data.values():
            for metric_name in recsys_data.get("metrics", {}).keys():
                metric_names.add(metric_name)
    return sorted(metric_names)


def plot_metrics_vs_degree(base_name: str, res_dir: str) -> int:
    path = os.path.join(res_dir, base_name, "recs_vs_degree", "metrics_vs_degree.json")
    payload = _load_json(path)
    if not payload:
        print(f"Skipping {base_name}: metrics-vs-degree file not found")
        return 0

    metric_names = _available_metrics(payload)
    if not metric_names:
        print(f"Skipping {base_name}: empty metrics-vs-degree payload")
        return 0

    apply_plot_style()

    saved = 0
    for metric in metric_names:
        plt.figure(figsize=(8, 5))
        plotted = 0

        for network in sorted(payload.keys()):
            for recsys in sorted(payload[network].keys()):
                block = payload[network][recsys]
                metric_block = block.get("metrics", {}).get(metric)
                if not metric_block:
                    continue

                x = np.array(block.get("bin_centers", []), dtype=float)
                y = _to_array(metric_block.get("mean", []))
                std = _to_array(metric_block.get("std", []))

                if x.size == 0 or y.size == 0:
                    continue

                n = min(x.size, y.size, std.size)
                x = x[:n]
                y = y[:n]
                std = std[:n]

                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(std) & (x > 0)
                if metric != "delta":
                    valid &= y > 0

                if not np.any(valid):
                    continue

                xx = x[valid]
                yy = y[valid]
                ss = std[valid]

                linestyle = "--" if network == "ER" else "-"
                plt.plot(
                    xx,
                    yy,
                    marker="o",
                    markersize=3,
                    linestyle=linestyle,
                    label=f"{network}-{recsys}",
                )

                low = yy - ss
                high = yy + ss
                if metric != "delta":
                    low = np.maximum(low, 1e-12)
                plt.fill_between(xx, low, high, alpha=0.15)
                plotted += 1

        if plotted == 0:
            plt.close()
            continue

        plt.xscale("log")
        if metric == "delta":
            plt.yscale("symlog", linthresh=1.0)
        else:
            plt.yscale("log")

        plt.xlabel("Author degree")
        plt.ylabel(metric)
        plt.legend(fontsize=7, ncol=2)

        out_dir = os.path.join(res_dir, base_name, "recs_vs_degree")
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"metric_vs_degree_{metric}.png")

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")
        saved += 1

    return saved


def main() -> None:
    args = parse_args()
    ensure_dir("logs")

    with open(args.log_file, "w") as log:
        for base_dir in args.base_dirs:
            base_name = os.path.basename(os.path.normpath(base_dir))
            header = f"Processing aggregate plots for {base_name}"
            print(header)
            log.write(header + "\n")

            count = 0
            if plot_global_distribution(base_name, args.res_dir):
                count += 1
            if plot_mean_distribution(base_name, args.res_dir):
                count += 1
            count += plot_metrics_vs_degree(base_name, args.res_dir)

            summary = f"Created {count} aggregate plots for {base_name}\n"
            print(summary.strip())
            log.write(summary)


if __name__ == "__main__":
    main()
