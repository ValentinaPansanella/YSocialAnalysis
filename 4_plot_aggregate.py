import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

BASE_DIRS = ["data"]  # datasets
RES_DIR = "res"

# ---------------------------------------------------
# UTIL
# ---------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------
# PLOT 1 — GLOBAL DISTRIBUTION
# ---------------------------------------------------

def plot_global_distribution(base_dir):

    path = f"{RES_DIR}/{base_dir}/recs_distribution/global_distribution_metrics.json"

    if not os.path.exists(path):
        print(f"Skipping {base_dir}: file not found")
        return

    with open(path, "r") as f:
        metrics = json.load(f)

    plt.figure(figsize=(6, 4))

    for network in metrics.keys():
        for recsys in metrics[network]:

            d = metrics[network][recsys]

            x = d["bins"]
            y = d["probs"]

            plt.scatter(x, y, label=f"{network}-{recsys}", marker="o")

    plt.xlabel("Num. Recommendations")
    plt.ylabel("P(x)")
    plt.legend()

    out_dir = f"{RES_DIR}/{base_dir}"
    ensure_dir(out_dir)

    out_path = f"{out_dir}/recs_distribution/sum_over_runs.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path}")


# ---------------------------------------------------
# PLOT 2 — MEAN DISTRIBUTION
# ---------------------------------------------------

def plot_mean_distribution(base_dir):

    path = f"{RES_DIR}/{base_dir}/recs_distribution/mean_distribution_metrics.json"

    if not os.path.exists(path):
        print(f"Skipping {base_dir}: file not found")
        return

    with open(path, "r") as f:
        metrics = json.load(f)

    plt.figure(figsize=(6, 4))

    for network in metrics.keys():
        for recsys in metrics[network]:

            d = metrics[network][recsys]

            x = np.array(d["bins"])
            y = np.array(d["mean"])
            std = np.array(d["std"])

            lowy = np.maximum(y - std, 0)
            highy = y + std

            linestyle = "--" if network == "ER" else "-"

            plt.plot(
                x,
                y,
                marker="o",
                markersize=2,
                linestyle=linestyle,
                label=f"{network}-{recsys}",
            )

            plt.fill_between(x, lowy, highy, alpha=0.3)

    plt.xlabel("Num. Recommendations")
    plt.ylabel("P(x)")
    plt.legend()

    plt.text(
        0.95,
        0.95,
        r"$P(x)=\frac{\text{num posts for }x}{\sum \text{num posts}}$",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.6),
    )

    out_dir = f"{RES_DIR}/{base_dir}"
    ensure_dir(out_dir)

    out_path = f"{out_dir}/recs_distribution/mean_over_runs.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path}")


# ---------------------------------------------------
# PLOT 3 — METRICS VS DEGREE
# ---------------------------------------------------

def plot_metrics_vs_degree(base_dir):

    path = f"{RES_DIR}/{base_dir}/recs_vs_degree/metrics_vs_degree.json"

    if not os.path.exists(path):
        print(f"Skipping {base_dir}: file not found")
        return

    with open(path, "r") as f:
        metrics = json.load(f)

    for metric in metrics:

        plt.figure(figsize=(6, 4))

        for network in metrics[metric]:
            for recsys in metrics[metric][network]:

                d = metrics[metric][network][recsys]

                x = np.array(d["degree"])
                y = np.array(d["value"])

                linestyle = "--" if network == "ER" else "-"

                plt.plot(
                    x,
                    y,
                    marker="o",
                    markersize=2,
                    linestyle=linestyle,
                    label=f"{network}-{recsys}",
                )

        plt.xlabel("Node degree")
        plt.ylabel(metric)
        plt.legend()

        out_dir = f"{RES_DIR}/{base_dir}/recs_vs_degree/"
        ensure_dir(out_dir)

        out_path = f"{out_dir}/metric_vs_degree_{metric}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {out_path}")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    for base_dir in BASE_DIRS:

        print(f"\nProcessing {base_dir}")

        plot_global_distribution(base_dir)
        plot_mean_distribution(base_dir)
        plot_metrics_vs_degree(base_dir)


if __name__ == "__main__":
    main()