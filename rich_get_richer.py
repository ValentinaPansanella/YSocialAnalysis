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
FIG_DIR = "figs/figs_rich"
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
# CORE ENGINE
# =============================================================================

def get_db_path(network, recsys, run):
    return os.path.join(BASE_DIR, f"{network}_{recsys}_{run}", "database_server.db")

def compute_preferential_attachment(network, recsys, run):
    """
    Computes P(New Recommendation | Current Reactions).
    Returns: Two dictionaries
        - numerator[k]: Total recs received by posts having k reactions.
        - denominator[k]: Total opportunities (number of posts having k reactions).
    """
    db_path = get_db_path(network, recsys, run)
    if not os.path.exists(db_path):
        return None, None

    with sqlite3.connect(db_path) as conn:
        # 1. Load Post Creation Rounds (Post only exists after this round)
        # We need this to normalize correctly (don't count non-existent posts)
        df_posts = pd.read_sql("SELECT id, round FROM post", conn)
        post_creation = df_posts.set_index('id')['round'].to_dict()
        all_post_ids = set(df_posts['id'].values)
        
        # 2. Load All Reactions (to build the "Wealth" history)
        # We need to know how many reactions post P had at round R.
        # Format: post_id -> list of rounds where reaction occurred
        cursor = conn.cursor()
        cursor.execute("SELECT post_id, round FROM reactions")
        
        # We build a matrix-like structure: reactions_at[post_id] = [r1, r2, r5...]
        post_reaction_events = defaultdict(list)
        for pid, r in cursor.fetchall():
            post_reaction_events[pid].append(r)
            
        # 3. Stream Recommendations (The "New Wealth")
        cursor.execute("SELECT post_ids, round FROM recommendations")
        
        # Storage for results
        # k = cumulative reactions
        numerator = defaultdict(int)   # How many recs given to posts with k reactions
        
        # We need to count occurrences of Recs.
        # Since 'denominator' (how many posts had k reactions) is heavy to compute 
        # inside the loop, we will compute it separately after.
        
        # Optimization: Pre-compute cumulative reactions for all posts at all rounds?
        # Too much memory.
        # Better: Replay the simulation round by round.
        
        # A. Organize Recs by Round
        recs_by_round = defaultdict(list) # round -> list of pids recommended
        while True:
            rows = cursor.fetchmany(10000)
            if not rows: break
            for pids_str, r in rows:
                if not pids_str: continue
                pids = [int(x) for x in pids_str.split('|')]
                recs_by_round[r].extend(pids)
        
    # --- SIMULATION REPLAY ---
    # We step through rounds 0 to MaxRound.
    # We maintain the current reaction count for every post.
    
    max_round = max(recs_by_round.keys()) if recs_by_round else 0
    current_reactions = {pid: 0 for pid in all_post_ids}
    
    # Pre-organize reactions by round for fast update
    # reactions_occurring_at[round] = [pid, pid, pid...]
    reactions_occurring_at = defaultdict(list)
    for pid, rounds in post_reaction_events.items():
        for r in rounds:
            reactions_occurring_at[r].append(pid)
            
    denominator = defaultdict(int) # Total chances: sum of (posts with k reactions) per round
    
    for r in range(max_round + 1):
        # 1. Update denominator (Availability)
        # For every LIVE post, what is its current reaction count?
        # A post is live if post_creation[pid] <= r
        
        # Optimization: Iterate only created posts
        # This part is the bottleneck. Python loops are slow.
        # Let's use vectorized numpy for speed if possible, or just optimize loop.
        
        # Snapshot of current wealth (reactions)
        # We count how many posts have 0 reactions, 1 reaction, etc.
        # Filter for existing posts only
        
        # Fast way: we know 'current_reactions' map.
        # We only care about posts that exist.
        # active_counts = [cnt for pid, cnt in current_reactions.items() if post_creation[pid] <= r]
        # This is still slow (checking 10k items 30 times).
        # But 10k * 30 = 300k ops, which is instant.
        
        active_counts = []
        for pid, created_at in post_creation.items():
            if created_at <= r:
                active_counts.append(current_reactions[pid])
        
        # Add to denominator
        # For each k in active_counts, we increment denominator[k]
        for k in active_counts:
            denominator[k] += 1
            
        # 2. Count Numerator (Recommendations received)
        # recs_at_r is a list of pids recommended in this round
        recs_at_r = recs_by_round.get(r, [])
        for pid in recs_at_r:
            # How many reactions did it have *before* being recommended?
            # (Assuming recs happen based on past state)
            k = current_reactions.get(pid, 0)
            numerator[k] += 1
            
        # 3. Update State (Apply reactions that happened in this round)
        # So they are available for round r+1
        new_reacts = reactions_occurring_at.get(r, [])
        for pid in new_reacts:
            if pid in current_reactions:
                current_reactions[pid] += 1
                
    return dict(numerator), dict(denominator)

def aggregate_runs(data_dicts):
    """
    Aggregates k-v pairs from multiple runs.
    """
    total_num = defaultdict(int)
    total_den = defaultdict(int)
    
    for num, den in data_dicts:
        if num is None: continue
        for k, v in num.items(): total_num[k] += v
        for k, v in den.items(): total_den[k] += v
        
    return total_num, total_den

# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    print("Starting Rich-get-Richer Analysis...")
    
    # Store aggregated data: [network][recsys] -> (numerator, denominator)
    results = defaultdict(dict)
    
    total_ops = len(NETWORKS) * len(RECSYSS) * len(RUNS)
    pbar = tqdm(total=total_ops, desc="Processing")
    
    for network in NETWORKS:
        for recsys in RECSYSS:
            run_data = []
            for run in RUNS:
                num, den = compute_preferential_attachment(network, recsys, run)
                run_data.append((num, den))
                pbar.update(1)
            
            # Aggregate over runs (Sum numerators, Sum denominators)
            results[network][recsys] = aggregate_runs(run_data)
            
    pbar.close()
    
    print("Generating Plots...")
    
    # Plotting
    fig, axes = plt.subplots(1, len(NETWORKS), figsize=(14, 6))
    if len(NETWORKS) == 1: axes = [axes]
    
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        
        for recsys in RECSYSS:
            num, den = results[network][recsys]
            
            # Compute Probability: P(Rec | k) = Num(k) / Den(k)
            # k is number of reactions
            
            x_vals = []
            y_vals = []
            
            # Sort by k
            sorted_k = sorted(den.keys())
            
            for k in sorted_k:
                if den[k] > 50: # Filter noise: requires at least 50 observations of this state
                    prob = num.get(k, 0) / den[k]
                    x_vals.append(k)
                    y_vals.append(prob)
            
            if not x_vals: continue
            
            # Binning for cleaner plots (logarithmic bins)
            # Because high-k tails are noisy
            df = pd.DataFrame({'k': x_vals, 'prob': y_vals})
            # Create bins: 0, 1, 2, ... then exponential
            # Or just plot scatter if dense enough. 
            # Let's use simple binning if many points, or raw if few.
            
            ax.loglog(x_vals, y_vals, 'o', label=recsys, alpha=0.6, markersize=4)
            
            # Fit line for slope estimate (optional visual guide)
            # ax.plot(x_vals, np.poly1d(np.polyfit(np.log1p(x_vals), np.log(y_vals), 1))(np.log1p(x_vals)))

        ax.set_title(f"{network} - Rich-get-Richer Effect")
        ax.set_xlabel("Cumulative Reactions (k)")
        ax.set_ylabel("Avg Recommendations / Round")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rich_get_richer_effect.png"))
    print(f"Saved plot to {FIG_DIR}/rich_get_richer_effect.png")