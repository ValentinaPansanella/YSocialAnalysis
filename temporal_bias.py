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
FIG_DIR = "figs/figs_content_age"
os.makedirs(FIG_DIR, exist_ok=True)

# Plot Styling

# =============================================================================
# CORE LOGIC
# =============================================================================

def get_db_path(network, recsys, run):
    return os.path.join(BASE_DIR, f"{network}_{recsys}_{run}", "database_server.db")

def compute_age_time_series(network, recsys, run):
    """
    Computes the average AGE (Current Round - Creation Round) 
    of posts recommended at each round.
    """
    db_path = get_db_path(network, recsys, run)
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # 1. Load Post Creation Time
        # We need to know WHEN each post was born.
        df_posts = pd.read_sql("SELECT id, round FROM post", conn)
        post_creation_round = df_posts.set_index('id')['round'].to_dict()
        
        # 2. Stream Recommendations
        cursor = conn.cursor()
        cursor.execute("SELECT round, post_ids FROM recommendations")
        
        # Storage: round -> list of ages of recommended posts
        round_ages = defaultdict(list)
        
        while True:
            rows = cursor.fetchmany(10000)
            if not rows:
                break
            
            for row in rows:
                current_round = row['round']
                pids_str = row['post_ids']
                
                if not pids_str:
                    continue
                
                try:
                    pids = [int(x) for x in pids_str.split('|')]
                except ValueError:
                    continue
                
                # For every recommended post, calculate AGE
                for pid in pids:
                    created_at = post_creation_round.get(pid)
                    
                    if created_at is not None:
                        # Age = Current Time - Birth Time
                        age = current_round - created_at
                        round_ages[current_round].append(age)

    # 3. Compute Average Age per Round
    avg_age_per_round = {}
    for r, ages in round_ages.items():
        if ages:
            avg_age_per_round[r] = np.mean(ages)
        else:
            avg_age_per_round[r] = 0.0
            
    return avg_age_per_round

# =============================================================================
# AGGREGATION & PLOTTING
# =============================================================================

def aggregate_time_series(list_of_dicts):
    """
    Aggregates a list of runs (dicts) into Mean and Std arrays.
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
    print("Starting Analysis: Age of Recommendations over Time...")
    
    # Data Storage: data[network][recsys] = list of runs
    data = defaultdict(lambda: defaultdict(list))
    
    total_ops = len(NETWORKS) * len(RECSYSS) * len(RUNS)
    pbar = tqdm(total=total_ops, desc="Processing Runs")
    
    # 1. Collect Data
    for network in NETWORKS:
        for recsys in RECSYSS:
            for run in RUNS:
                res = compute_age_time_series(network, recsys, run)
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
        
        ax.set_title(f"{network} - Content Freshness")
        ax.set_xlabel("Round (Time)")
        ax.set_ylabel("Avg Age of Recommended Posts (Rounds)")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    output_path = os.path.join(FIG_DIR, "avg_age_of_recs_over_time.png")
    plt.savefig(output_path, facecolor='white', transparent=False, dpi=300)
    print(f"Saved plot to {output_path}")