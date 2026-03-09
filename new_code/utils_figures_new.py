import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
import numpy as np
from collections import Counter
import itertools
from matplotlib.cm import get_cmap
from typing import Optional
from utils_new import *
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

plt.style.use("ggplot")

params = {

    # ---- FONT ----
    "font.family": "sans-serif",
    "font.style": "normal",

    # ---- FIGURE ----
    "figure.facecolor": "white",
    "figure.dpi": 300,

    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "savefig.facecolor": "white", # Background when saving
    "savefig.transparent": False, # <--- CHANGE THIS TO FALSE

    # ---- AXES ----
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1,
    "axes.labelsize": 11,
    "axes.labelcolor": "black",

    "axes.spines.right": False,
    "axes.spines.top": False,

    # ---- GRID (delicate dashed grey) ----
    "axes.grid": True,
    "grid.color": "#D3D3D3",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,

    # ---- TICKS (bigger labels) ----
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.major.size": 3,
    "ytick.major.size": 3,

    # ---- LINES ----
    "lines.linewidth": 1.5,
    "lines.markersize": 6,

    # ---- LEGEND ----
    "legend.fontsize": 8,

    # ---- COLOR CYCLE ----
    "axes.prop_cycle": cycler(
        color=["#D81B60", "#1E88E5", "#FFC107", "#004D40"]
    )
}


def apply_plot_style():
    """
    Applies the custom plot style defined in rcParams.
    Call this function at the beginning of your plotting code to ensure consistent styling.
    """
    plt.rcParams.update(params)


def plot_degree(degrees, network, recsys, run, save = True):
    """
    Plots the degree distribution (Log-Log) for a single run.
    """
    
    apply_plot_style() # Ensure our custom style is applied
    
    if not degrees:
        return

    # Create a specific subfolder for these plots
    save_dir = os.path.join('figs', 'degree_distributions')
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data: Count frequency of each degree
    degrees = list(author_degrees.values())
    deg_counts = Counter(degrees)
    
    # Sort by degree (k)
    x = sorted(deg_counts.keys())
    y = [deg_counts[k] for k in x]
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color='steelblue', alpha=0.7, s=20, edgecolors='none')
    
    # Log-Log scales for Scale-Free/Power-Law visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_title(f"Degree Distribution: {network} - {recsys} - Run {run}")
    ax.set_xlabel("Degree ($k$)")
    ax.set_ylabel("Count ($N_k$)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, f"deg_dist_{network}_{recsys}_{run}.png"))
    else:
        plt.show()
    plt.close()

from typing import Optional

def plot_recommendations_loglog(
    metrics: dict,
    networks: list[str] = ['ER', 'BA'],
    save_path: Optional[str] = None,
    analysis_type: str = 'post'  # 'post' or 'user'
):
    """
    Plot recommendation metrics on a log-log scale.

    Args:
        metrics: Nested dict returned by agg_runs.
        networks: List of network types to include. Default is ['ER','BA'].
        save_path: Optional path to save the plot. If None, the plot is shown.
        analysis_type: Either 'post' or 'user', controls axis labels.
    """
    
    apply_plot_style()
    
    fig, axes = plt.subplots(1, len(networks), figsize=(6*len(networks), 5), sharey=True, sharex=True)
    if len(networks) == 1:
        axes = [axes]  # Ensure iterable
    
    x_label = "# Item Recommendations" if analysis_type == 'post' else "# User Recommendations"

    for ax, network in zip(axes, networks):
        if network not in metrics:
            print(f"Warning: no metrics for network {network}")
            continue

        recsys_list = list(metrics[network].keys())
        
        for recsys in recsys_list:
            data = metrics[network][recsys]
            bins = data['bins']
            mean_dist = data['mean']
            std_dist = data['std']
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Clip lower error for log scale
            yerr_lower = np.minimum(std_dist, mean_dist * 0.9)
            yerr = [yerr_lower, std_dist]
            
            print(len(bin_centers), len(mean_dist))

            mask = mean_dist > 0
            ax.errorbar(bins[mask], mean_dist[mask], yerr=std_dist[mask], fmt='o', capsize=3, label=recsys)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(x_label)
        ax.set_title(f"Network: {network}")
        ax.set_xlim(1, max(bins[mask])+200)
        ax.set_ylim(min(mean_dist[mask]), max(mean_dist[mask])*10)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
    
    axes[0].set_ylabel("Frequency")
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        figname = f'recommendations_frequency_distribution_average_{analysis_type}.png'
        figpath = os.path.join(save_path, figname)
        plt.savefig(figpath)
    else:
        plt.show()
    
    plt.close()
    
    
def plot_recs_vs_degree(
    metrics: dict,
    networks: list[str] = ['ER', 'BA'],
    save_path: str = None,
    scale='log',
    metrics_to_plot: list[str] = ['total_recommendations', 'total_recommendations_normalized','unique_reach', 'delta_recs'],
):
    
    apply_plot_style()

    for metric in metrics_to_plot:
        
        metric_path = os.path.join(save_path, metric) if save_path is not None else None
        if metric_path is not None:
            os.makedirs(metric_path, exist_ok=True)

        fig, axes = plt.subplots(
            1,
            len(networks),
            figsize=(6 * len(networks), 5),
            sharey=True,
        )

        if len(networks) == 1:
            axes = [axes]

        for ax, network in zip(axes, networks):

            if network not in metrics:
                continue

            recsys_list = list(metrics[network].keys())
            
            for recsys in recsys_list:
                if metric not in metrics[network][recsys]:
                    continue

                data = metrics[network][recsys][metric]
                bins = data['bins']
                mean_curve = np.array(data['mean'])
                std_curve = np.array(data['std'])

                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                mask = ~np.isnan(mean_curve)
                x = bin_centers[mask]
                print('x:', x)
                y = mean_curve[mask]
                print('y:', y)
                yerr_std = std_curve[mask]

                if len(x) == 0:
                    continue

                # ----- ERROR BAR FIX -----
                if metric == 'delta_recs':
                    # Allow negative error bars freely
                    yerr = yerr_std 
                else:
                    # Original logic for strictly positive metrics
                    yerr_lower = np.maximum(0, np.minimum(yerr_std, y * 0.9))
                    yerr = [yerr_lower, yerr_std]

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt='o-',
                    capsize=3,
                    label=recsys
                )

            # ----- SCALE FIX -----
            if scale == 'log':
                ax.set_xscale('log')
                if metric == 'delta_recs':
                    # Use 'symlog' (symmetric log) to handle negative numbers but keep log spacing, 
                    # OR just use 'linear'. 'symlog' is usually best if the range is huge (e.g. -1000 to +1000).
                    # linthresh defines the range around 0 that is linear to avoid log(0) issues.
                    ax.set_yscale('symlog', linthresh=10.0) 
                    
                    # Add a prominent 0-line to show where users flip from consumers to broadcasters
                    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
                else:
                    ax.set_yscale('log')

            ax.set_xlabel("Author Degree")
            ax.set_title(f"Network: {network}")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend()

        ylabel_map = {
            'total_recommendations': "Avg Total R",
            'total_recommendations_normalized': r"$\langle R \rangle_{\mathrm{post}}$",
            'unique_reach': "Avg Unique Reach",
            'delta_recs': "Avg ΔR (Reach - Viewed)"
        }
        
        axes[0].set_ylabel(ylabel_map.get(metric, metric))

        plt.tight_layout()

        if save_path is not None:
            figpath = os.path.join(metric_path, f"{metric}_vs_degree_{scale}.png")
            plt.savefig(figpath, dpi=300)
        else:
            plt.show()

        plt.close()


# def plot_reaction_distribution(counts):
#     print('plotting reaction distribution...')
    
#     x = counts.index
#     y = counts.values
    
#     # Plot
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.scatter(x, y, color='royalblue', alpha=0.7, s=30, edgecolors='none')
    
#     # Log-Log scales for Scale-Free/Power-Law visualization
#     ax.set_xscale('log')
#     ax.set_yscale('log')
    
#     ax.set_title(f"Distribution reaction_count: {network} - {recsys} - Run {run}")
#     ax.set_xlabel("Post reactions")
#     ax.set_ylabel("Frequency")
#     ax.grid(True, which="both", ls="--", alpha=0.3)
    
#     plt.tight_layout()
#     if save:
#         plt.savefig(os.path.join(save_dir, f"reaction_count_distrib_{network}_{recsys}_{run}.png"))
#     else:
#         plt.show()
#     plt.close()
#     print('finished plotting reaction distribution...')
    
# def plot_recommendations_distribution(counts):
#     print('plotting recommendations distribution...')
    
#     x = counts.index
#     y = counts.values
    
#     # Plot
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.scatter(x, y, color='royalblue', alpha=0.7, s=30, edgecolors='none')
    
#     # Log-Log scales for Scale-Free/Power-Law visualization
#     ax.set_xscale('log')
#     ax.set_yscale('log')
    
#     ax.set_title(f"Distribution recommendations: {network} - {recsys} - Run {run}")
#     ax.set_xlabel("Post Impressions")
#     ax.set_ylabel("Frequency")
#     ax.grid(True, which="both", ls="--", alpha=0.3)
    
#     plt.tight_layout()
#     if save:
#         plt.savefig(os.path.join(save_dir, f"recommendations_distrib_{network}_{recsys}_{run}.png"))
#     else:
#         plt.show()
#     plt.close()
#     print('finished plotting recommendations distribution...')

