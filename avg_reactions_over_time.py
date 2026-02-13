import os
import glob
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from matplotlib import rcParams
from cycler import cycler


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = "experiments_recsys"
NETWORKS = ['BA', 'ER']
RECSYSS = ['F', 'FP', 'P', 'RC']
RUNS = list(range(10))

# Plot output directory
FIG_DIR = "figs/figs_reactions"
os.makedirs(FIG_DIR, exist_ok=True)

# Plot Styling
plt.style.use("ggplot")

rcParams.update({

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
    ),
})

# =============================================================================
# CORE LOGIC
# =============================================================================

def get_db_path(network, recsys, run):
    # Adjust this path pattern if your folder structure is slightly different
    return os.path.join(BASE_DIR, f"{network}_{recsys}_{run}", "database_server.db")

def compute_quality_time_series(network, recsys, run):
    """
    Computes the average reaction count of posts recommended at each round.
    Returns: dict {round: avg_reaction_count}
    """
    db_path = get_db_path(network, recsys, run)
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # 1. Load Post Quality (Final Reaction Count)
        # We assume 'reaction_count' in the post table reflects the total/final popularity.
        # This tells us if the system is recommending "good" (popular) content at round T.
        df_posts = pd.read_sql("SELECT id, reaction_count FROM post", conn)
        post_quality = df_posts.set_index('id')['reaction_count'].to_dict()
        
        # 2. Stream Recommendations
        cursor = conn.cursor()
        cursor.execute("SELECT round, post_ids FROM recommendations")
        
        # Storage: round -> list of qualities of recommended posts
        round_qualities = defaultdict(list)
        
        while True:
            rows = cursor.fetchmany(10000)
            if not rows:
                break
            
            for row in rows:
                r = row['round']
                pids_str = row['post_ids']
                
                if not pids_str:
                    continue
                
                try:
                    pids = [int(x) for x in pids_str.split('|')]
                except ValueError:
                    continue
                
                # For every recommended post, get its quality (reaction count)
                for pid in pids:
                    q = post_quality.get(pid, 0)
                    round_qualities[r].append(q)

    # 3. Compute Average per Round
    avg_quality_per_round = {}
    for r, qualities in round_qualities.items():
        if qualities:
            avg_quality_per_round[r] = np.mean(qualities)
        else:
            avg_quality_per_round[r] = 0.0
            
    return avg_quality_per_round

# =============================================================================
# AGGREGATION & PLOTTING
# =============================================================================

def aggregate_time_series(list_of_dicts):
    """
    Aggregates a list of runs (dicts) into Mean and Std arrays for plotting.
    """
    if not list_of_dicts:
        return None, None, None
    
    # Find all rounds present across runs
    all_rounds = set()
    for d in list_of_dicts:
        all_rounds.update(d.keys())
    
    if not all_rounds:
        return None, None, None
        
    sorted_rounds = sorted(list(all_rounds))
    
    means = []
    stds = []
    
    for r in sorted_rounds:
        # Collect values for round r from all runs
        vals = [d[r] for d in list_of_dicts if r in d]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(0)
            stds.append(0)
            
    return np.array(sorted_rounds), np.array(means), np.array(stds)

if __name__ == "__main__":
    print("Starting Analysis: Quality of Recommendations over Time...")
    
    # Data Storage: data[network][recsys] = list of runs
    data = defaultdict(lambda: defaultdict(list))
    
    total_ops = len(NETWORKS) * len(RECSYSS) * len(RUNS)
    pbar = tqdm(total=total_ops, desc="Processing Runs")
    
    # 1. Collect Data
    for network in NETWORKS:
        for recsys in RECSYSS:
            for run in RUNS:
                res = compute_quality_time_series(network, recsys, run)
                if res:
                    data[network][recsys].append(res)
                pbar.update(1)
    pbar.close()
    
    # 2. Plotting
    print("Generating Plot...")
    
    fig, axes = plt.subplots(1, len(NETWORKS), figsize=(14, 6))
    if len(NETWORKS) == 1: axes = [axes]
    
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        
        for recsys in RECSYSS:
            runs = data[network][recsys]
            x, y, err = aggregate_time_series(runs)
            
            if x is not None:
                ax.plot(x, y, label=recsys, linewidth=2)
                ax.fill_between(x, y - err, y + err, alpha=0.15)
        
        ax.set_title(f"{network} - Quality of Recommendations")
        ax.set_xlabel("Round (Time)")
        ax.set_ylabel("Avg Reactions of Recommended Posts")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    output_path = os.path.join(FIG_DIR, "avg_reactions_of_recs_over_time.png")
    plt.savefig(output_path, facecolor='white', transparent=False, dpi=300)
    print(f"Saved plot to {output_path}")