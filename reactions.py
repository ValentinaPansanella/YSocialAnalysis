# import os
# import sqlite3
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# from cycler import cycler
# from tqdm import tqdm  # Use notebook version for nice progress bar
# from collections import defaultdict
# import numpy as np


# # --- Configuration ---
# BASE_DIR = "experiments_recsys"
# NETWORKS = ['BA', 'ER']
# RECSYSS = ['F', 'FP', 'P', 'RC']
# RUNS = list(range(10))

# # Folder to save the 80 images
# FIG_DIR = "figS/reactions"
# os.makedirs(FIG_DIR, exist_ok=True)

# # Plot Styling
# plt.style.use("ggplot")

# rcParams.update({

#     # ---- FONT ----
#     "font.family": "sans-serif",
#     "font.style": "normal",

#     # ---- FIGURE ----
#     "figure.facecolor": "white",
#     "figure.dpi": 300,

#     "savefig.bbox": "tight",
#     "savefig.dpi": 300,
#     "savefig.facecolor": "white", # Background when saving
#     "savefig.transparent": False, # <--- CHANGE THIS TO FALSE

#     # ---- AXES ----
#     "axes.facecolor": "white",
#     "axes.edgecolor": "black",
#     "axes.linewidth": 1,
#     "axes.labelsize": 11,
#     "axes.labelcolor": "black",

#     "axes.spines.right": False,
#     "axes.spines.top": False,

#     # ---- GRID (delicate dashed grey) ----
#     "axes.grid": True,
#     "grid.color": "#D3D3D3",
#     "grid.linestyle": "--",
#     "grid.linewidth": 0.6,
#     "grid.alpha": 0.7,

#     # ---- TICKS (bigger labels) ----
#     "xtick.color": "black",
#     "ytick.color": "black",
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "xtick.major.width": 1,
#     "ytick.major.width": 1,
#     "xtick.major.size": 3,
#     "ytick.major.size": 3,

#     # ---- LINES ----
#     "lines.linewidth": 1.5,
#     "lines.markersize": 6,

#     # ---- LEGEND ----
#     "legend.fontsize": 8,

#     # ---- COLOR CYCLE ----
#     "axes.prop_cycle": cycler(
#         color=["#D81B60", "#1E88E5", "#FFC107", "#004D40"]
#     ),
# })



# def get_db_path(network, recsys, run):
#     return os.path.join(BASE_DIR, f"{network}_{recsys}_{run}", "database_server.db")

# def compute_activity_metric(network, recsys, run):
#     """
#     Computes: (Total Reactions in Round R) / (Total Posts Existing in Round R)
#     Returns: dict {round: avg_reactions_per_post}
#     """
#     db_path = get_db_path(network, recsys, run)
#     if not os.path.exists(db_path):
#         return None

#     with sqlite3.connect(db_path) as conn:
#         # 1. Get counts of NEW posts per round
#         # "round" in table 'post' is the creation round
#         df_posts = pd.read_sql("SELECT round FROM post", conn)
        
#         # 2. Get counts of NEW reactions per round
#         # "round" in table 'reactions' is when the reaction happened
#         df_reacts = pd.read_sql("SELECT round FROM reactions", conn)
        
#     if df_posts.empty or df_reacts.empty:
#         return None

#     # Determine max round
#     max_r_p = df_posts['round'].max()
#     max_r_r = df_reacts['round'].max()
#     max_round = max(max_r_p, max_r_r)
    
#     # --- Vectorized Counting ---
    
#     # 1. Count new posts per round
#     # bins must cover 0 to max_round. So bins=max_round+1 range
#     # bincount returns array where index i is count of value i
#     new_posts_per_round = np.bincount(df_posts['round'], minlength=max_round+1)
    
#     # 2. Cumulative posts (Total existing posts at round R)
#     total_posts_at_round = np.cumsum(new_posts_per_round)
    
#     # 3. Count reactions per round
#     reactions_per_round = np.bincount(df_reacts['round'], minlength=max_round+1)
    
#     # 4. Compute Metric (Avoid division by zero)
#     # Metric = Reactions[R] / TotalPosts[R]
    
#     avg_new_reactions = np.zeros_like(reactions_per_round, dtype=float)
    
#     # Mask where posts exist (>0) to avoid div/0
#     mask = total_posts_at_round > 0
#     avg_new_reactions[mask] = reactions_per_round[mask] / total_posts_at_round[mask]
    
#     # Convert to dict for compatibility with aggregation function
#     return {r: val for r, val in enumerate(avg_new_reactions)}

# # =============================================================================
# # AGGREGATION & PLOTTING
# # =============================================================================

# def aggregate_time_series(list_of_dicts):
#     """
#     Aggregates a list of runs (dicts) into Mean and Std arrays.
#     """
#     if not list_of_dicts:
#         return None, None, None
    
#     # Find all rounds present across runs
#     all_rounds = set()
#     for d in list_of_dicts:
#         all_rounds.update(d.keys())
    
#     if not all_rounds:
#         return None, None, None
        
#     sorted_rounds = sorted(list(all_rounds))
    
#     means = []
#     stds = []
    
#     for r in sorted_rounds:
#         # Collect values for round r from all runs
#         # Use .get(r, 0) assuming if round missing, no activity or simulation ended
#         # But usually simulations have fixed length.
#         vals = [d.get(r, 0) for d in list_of_dicts]
        
#         means.append(np.mean(vals))
#         stds.append(np.std(vals))
            
#     return np.array(sorted_rounds), np.array(means), np.array(stds)


# # # --- Main Loop ---
# # total_tasks = len(NETWORKS) * len(RECSYSS) * len(RUNS)
# # pbar = tqdm(total=total_tasks, desc="Generating Plots")

# # for network in NETWORKS:
# #     for recsys in RECSYSS:
# #         for run in RUNS:
# #             # 1. Connect to DB
# #             folder_name = f"{network}_{recsys}_{run}"
# #             db_path = os.path.join(BASE_DIR, folder_name, "database_server.db")
            
# #             if not os.path.exists(db_path):
# #                 pbar.update(1)
# #                 continue

# #             try:
# #                 with sqlite3.connect(db_path) as conn:
# #                     # 2. Get Reaction Counts
# #                     query = "SELECT reaction_count FROM post"
# #                     df = pd.read_sql_query(query, conn)
                
# #                 # 3. Plot
# #                 fig, ax = plt.subplots(figsize=(6, 4))
                
# #                 if not df.empty and df['reaction_count'].max() > 0:
# #                     # Histogram
# #                     ax.hist(df['reaction_count'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
                    
# #                     # Styles
# #                     ax.set_yscale('log') # Use Log scale for better visibility of heavy tails
# #                     ax.set_title(f"Reactions: {network} - {recsys} - Run {run}", fontsize=11)
# #                     ax.set_xlabel("Number of Reactions")
# #                     ax.set_ylabel("Frequency (Log)")
# #                 else:
# #                     ax.text(0.5, 0.5, "No Reactions / Empty", ha='center', va='center')
# #                     ax.set_title(f"{network} - {recsys} - Run {run}")

# #                 # 4. Save and Close
# #                 filename = f"dist_react_{network}_{recsys}_{run}.png"
# #                 plt.tight_layout()
# #                 plt.savefig(os.path.join(FIG_DIR, filename), dpi=150)
# #                 plt.close(fig)  # Important: Closes plot to prevent displaying 80 images inline

# #             except Exception as e:
# #                 print(f"Error in {folder_name}: {e}")

# #             pbar.update(1)


# # print("Starting Analysis: Avg New Reactions per Post per Round...")
    
# # data = defaultdict(lambda: defaultdict(list))

# # total_ops = len(NETWORKS) * len(RECSYSS) * len(RUNS)
# # pbar = tqdm(total=total_ops, desc="Processing Runs")

# # # 1. Collect Data
# # for network in NETWORKS:
# #     for recsys in RECSYSS:
# #         for run in RUNS:
# #             res = compute_activity_metric(network, recsys, run)
# #             if res:
# #                 data[network][recsys].append(res)
# #             pbar.update(1)
# # pbar.close()

# # # 2. Plotting
# # print("Generating Plot...")

# # fig, axes = plt.subplots(1, len(NETWORKS), figsize=(14, 6))
# # if len(NETWORKS) == 1: axes = [axes]

# # for i, network in enumerate(NETWORKS):
# #     ax = axes[i]
    
# #     for recsys in RECSYSS:
# #         runs = data[network][recsys]
# #         x, y, err = aggregate_time_series(runs)
        
# #         if x is not None:
# #             # Plot
# #             ax.plot(x, y, label=recsys, linewidth=2)
# #             ax.fill_between(x, y - err, y + err, alpha=0.15)
    
# #     ax.set_title(f"{network} - Engagement Activity")
# #     ax.set_xscale('log')
# #     ax.set_yscale('log')
# #     ax.set_xlabel("Round (Time)")
# #     ax.set_ylabel("Avg New Reactions / Post")
# #     ax.legend(loc='upper right')
# #     ax.grid(True, linestyle='--', alpha=0.5)
    
# # plt.tight_layout()
# # output_path = os.path.join(FIG_DIR, "avg_new_reactions_per_round.png")
# # plt.savefig(output_path, facecolor='white', transparent=False, dpi=300)

# # print(f"Saved plot to {output_path}")

import os
import glob
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = "experiments_recsys"
NETWORKS = ['BA', 'ER']
RECSYSS = ['F', 'FP', 'P', 'RC']
RUNS = list(range(10))

# Output directory
FIG_DIR = "figs_activity"
os.makedirs(FIG_DIR, exist_ok=True)

# Plot Styling
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'axes.spines.right': False,
    'axes.spines.top': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.transparent': False
})

# =============================================================================
# CORE LOGIC
# =============================================================================

def get_db_path(network, recsys, run):
    return os.path.join(BASE_DIR, f"{network}_{recsys}_{run}", "database_server.db")

def compute_cumulative_activity_metric(network, recsys, run):
    """
    Computes: (Total Accumulated Reactions up to Round R) / (Total Posts Existing in Round R)
    Returns: dict {round: avg_cumulative_reactions_per_post}
    """
    db_path = get_db_path(network, recsys, run)
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        # 1. Get creation rounds of posts
        df_posts = pd.read_sql("SELECT round FROM post", conn)
        
        # 2. Get occurrence rounds of reactions
        df_reacts = pd.read_sql("SELECT round FROM reactions", conn)
        
    if df_posts.empty:
        return None

    # Determine max round
    max_r_p = df_posts['round'].max()
    max_r_r = df_reacts['round'].max() if not df_reacts.empty else 0
    max_round = max(max_r_p, max_r_r)
    
    # --- Vectorized Calculation ---
    
    # 1. Posts Count (Denominator)
    # new_posts_per_round[i] = number of posts created exactly at round i
    new_posts_per_round = np.bincount(df_posts['round'], minlength=max_round+1)
    
    # total_posts_at_round[i] = total posts existing at round i (Cumulative)
    total_posts_at_round = np.cumsum(new_posts_per_round)
    
    # 2. Reactions Count (Numerator)
    if not df_reacts.empty:
        # new_reactions_per_round[i] = reactions happening exactly at round i
        new_reactions_per_round = np.bincount(df_reacts['round'], minlength=max_round+1)
        
        # total_reactions_at_round[i] = total reactions accumulated by the system up to round i
        total_reactions_at_round = np.cumsum(new_reactions_per_round)
    else:
        total_reactions_at_round = np.zeros(max_round+1)
    
    # 3. Compute Metric: Cumulative Reactions / Cumulative Posts
    avg_cumulative_reactions = np.zeros_like(total_posts_at_round, dtype=float)
    
    # Mask where posts exist (>0) to avoid div/0
    mask = total_posts_at_round > 0
    avg_cumulative_reactions[mask] = total_reactions_at_round[mask] / total_posts_at_round[mask]
    
    # Convert to dict for compatibility with aggregation function
    return {r: val for r, val in enumerate(avg_cumulative_reactions)}

# =============================================================================
# AGGREGATION & PLOTTING
# =============================================================================

def aggregate_time_series(list_of_dicts):
    """
    Aggregates a list of runs (dicts) into Mean and Std arrays.
    """
    if not list_of_dicts:
        return None, None, None
    
    all_rounds = set()
    for d in list_of_dicts:
        all_rounds.update(d.keys())
    
    if not all_rounds:
        return None, None, None
        
    sorted_rounds = sorted(list(all_rounds))
    
    means = []
    stds = []
    
    for r in sorted_rounds:
        vals = [d.get(r, 0) for d in list_of_dicts]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
            
    return np.array(sorted_rounds), np.array(means), np.array(stds)

if __name__ == "__main__":
    print("Starting Analysis: Avg Cumulative Reactions per Post...")
    
    data = defaultdict(lambda: defaultdict(list))
    
    total_ops = len(NETWORKS) * len(RECSYSS) * len(RUNS)
    pbar = tqdm(total=total_ops, desc="Processing Runs")
    
    # 1. Collect Data
    for network in NETWORKS:
        for recsys in RECSYSS:
            for run in RUNS:
                res = compute_cumulative_activity_metric(network, recsys, run)
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
        
        ax.set_title(f"{network} - Cumulative Engagement")
        # Log-Log scale helps visualize growth over orders of magnitude
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel("Round (Time)")
        ax.set_ylabel("Avg Cumulative Reactions / Post")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5, which='both')
        
    plt.tight_layout()
    output_path = os.path.join(FIG_DIR, "avg_cumulative_reactions_per_round.png")
    plt.savefig(output_path, facecolor='white', transparent=False, dpi=300)
    print(f"Saved plot to {output_path}")