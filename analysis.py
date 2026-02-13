import sys
import os
import glob
import sqlite3
import gc
import pickle
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
from cycler import cycler
from collections import defaultdict, Counter
from tqdm import tqdm
import networkx as nx

matplotlib.use('Agg') 

# --- Configuration & Plotting Styles ---
NETWORKS = ['BA', 'ER']
RECSYSS = ['F', 'FP', 'P', 'RC']
RUNS = list(range(10))

# Binning config for Degree plots (aggregates noisy degree data into bins)
DEGREE_BINS = np.linspace(0, 200, 21) # 20 bins from 0 to 200

# Create directories
DIRS = {
    'data': 'processed_data',
    'figs': 'figs'
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

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
# PART 1: CORE PROCESSING ENGINE (ETL)
# =============================================================================

def get_db_path(network, recsys, run):
    return f"experiments_recsys/{network}_{recsys}_{run}/database_server.db"

def get_cache_path(network, recsys, run):
    return os.path.join(DIRS['data'], f"{network}_{recsys}_{run}.pkl")

# def compute_author_degrees_from_csv(csv_path):
#     """
#     Reads an edge list CSV and computes node degrees using NetworkX.
#     Returns a dict: node -> adjusted degree (degree/2)
#     """
    
#     print('computing user degrees from CSV...')
    
#     try:
#         # Read edge list
#         df_edges = pd.read_csv(csv_path, header=None, names=['u', 'v'])
        
#         # Create undirected graph
#         G = nx.from_pandas_edgelist(df_edges, source='u', target='v')
        
#         # Compute degrees and divide by 2 as per your note
#         author_degrees = {node: deg for node, deg in G.degree()}
#         return author_degrees
#     except Exception as e:
#         print(f"Error computing degrees from {csv_path}: {e}")
#         return {}


def plot_degree(author_degrees, network, recsys, run):
    """
    Plots the degree distribution (Log-Log) for a single run.
    """
    if not author_degrees:
        return

    # Create a specific subfolder for these plots
    save_dir = os.path.join(DIRS['figs'], 'degree_distributions')
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
    plt.savefig(os.path.join(save_dir, f"deg_dist_{network}_{recsys}_{run}.png"))
    plt.close() # Important to close to save memory

def compute_run_metrics(network, recsys, run):
    """
    Computes ALL metrics for a single run in one go to minimize I/O.
    """
    print(f'Start computing metrics for {network}-{recsys}-{run}')

    db_path = get_db_path(network, recsys, run)

    if not os.path.exists(db_path):
        print(f"Warning: DB not found at {db_path}")
        return None 

    metrics = {}

    # --- 1. Load Network Degree ---
    run_folder = os.path.dirname(db_path)
    csv_files = glob.glob(os.path.join(run_folder, "*.csv"))
    author_degrees = {}

    if csv_files:
        try:
            # Load CSV (Node IDs are Integers here)
            df_edges = pd.read_csv(csv_files[0], header=None, names=['source', 'target'])
            deg_counts = pd.concat([df_edges['source'], df_edges['target']]).value_counts()
            author_degrees = deg_counts.to_dict()
        except Exception as e:
            print(f"Error reading network CSV for {network}-{recsys}-{run}: {e}")
            
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # --- 2. Post Metadata (ID, Author, Reactions) ---
        query_posts = "SELECT id, user_id, reaction_count, round FROM post"
        df_posts = pd.read_sql(query_posts, conn)
        df_posts.set_index('id', inplace=True)
        
        # --- CRITICAL FIX: STRING TO INT MAPPING ---
        # 1. Load Map: User ID -> Username (String)
        df_users = pd.read_sql("SELECT id, username FROM user_mgmt", conn)
        user_map = df_users.set_index('id')['username'].to_dict()
        
        # 2. Map Post -> Username (String)
        df_posts['username'] = df_posts['user_id'].map(user_map)
        
        # 3. Convert Username (String) to Node ID (Int)
        #    This makes "42" match 42 in the degree dictionary
        def safe_to_int(x):
            try:
                return int(float(x)) # Handles "1.0" or "1" strings
            except (ValueError, TypeError):
                return -1 # Fallback for non-numeric usernames
                
        df_posts['node_id'] = df_posts['username'].apply(safe_to_int)
        
        # 4. Map Node ID -> Degree (using the CSV loaded degrees)
        df_posts['degree'] = df_posts['node_id'].map(author_degrees).fillna(0).astype(int)

        # --- 3. Recommendations Analysis ---
        cursor.execute("SELECT round, post_ids, user_id FROM recommendations")
    
        # Initialization
        recs_per_round = defaultdict(int)
        post_rec_counts = Counter() 
        post_rec_rounds = defaultdict(list) 
        post_unique_reach = defaultdict(set) 
        post_impressions = Counter()
        unique_recs_per_round = defaultdict(set) 

        total_recs_count = 0
    
        while True:
            rows = cursor.fetchmany(50000) # Large batch for speed
            if not rows:
                break
                
            for row in rows:
                r = row['round']
                viewer_id = row['user_id']
                pids_str = row['post_ids']
                
                if not pids_str:
                    continue
                    
                try:
                    pids = [int(x) for x in pids_str.split("|")]
                except ValueError:
                    continue 
                
                count = len(pids)
                total_recs_count += count
                
                # Aggregations
                recs_per_round[r] += count
                unique_recs_per_round[r].update(pids)
                
                for pid in pids:
                    post_rec_counts[pid] += 1
                    post_rec_rounds[pid].append(r)
                    post_impressions[pid] += 1
                    post_unique_reach[pid].add(viewer_id)

    # --- METRICS CALCULATION ---

    # Convert sets to counts
    post_reach_counts = {pid: len(viewers) for pid, viewers in post_unique_reach.items()}
    
    # Metric A
    metrics['recs_per_round'] = dict(recs_per_round)
    
    # Metric B
    metrics['rec_count_hist'] = dict(Counter(post_rec_counts.values()))
    
    # Metric C: Streaks
    streak_counts = Counter()
    gap_counts = Counter()
    consecutive_pairs = 0
    total_pairs = 0
    
    for pid, rounds in post_rec_rounds.items():
        rounds.sort()
        current_streak = 1
        for i in range(1, len(rounds)):
            if rounds[i] == rounds[i-1] + 1:
                current_streak += 1
                consecutive_pairs += 1
            else:
                streak_counts[current_streak] += 1
                current_streak = 1
                gap = rounds[i] - rounds[i-1] - 1
                gap_counts[gap] += 1
            total_pairs += 1
        streak_counts[current_streak] += 1 
   
    metrics['streak_hist'] = dict(streak_counts)
    metrics['gap_hist'] = dict(gap_counts)
    metrics['consec_prob'] = consecutive_pairs / total_pairs if total_pairs > 0 else 0
    
    # Metric D: Fraction of Total Posts Recommended
    total_posts_per_round = df_posts['round'].value_counts().to_dict()
    frac_rec_per_round = {}
    
    for r, total_p in total_posts_per_round.items():
        if total_p > 0:
            frac_rec_per_round[r] = len(unique_recs_per_round.get(r, [])) / total_p
        else:
            frac_rec_per_round[r] = 0.0
            
    metrics['frac_rec_per_round'] = frac_rec_per_round

    # Metric E: Reactions
    df_posts['rec_count'] = df_posts.index.map(post_rec_counts).fillna(0).astype(int)
    reactions_by_rec_count = df_posts.groupby('rec_count')['reaction_count'].mean().to_dict()
    metrics['reactions_by_rec_count'] = reactions_by_rec_count
    
    metrics['avg_reactions_all'] = df_posts['reaction_count'].mean()
    metrics['avg_reactions_rec'] = df_posts[df_posts['rec_count'] > 0]['reaction_count'].mean()
    
    # Metric F: Recommendations vs Author Degree
    avg_rec_per_degree = df_posts.groupby('degree')['rec_count'].mean().to_dict()
    metrics['rec_per_degree'] = avg_rec_per_degree
    
    # Metric G: Unique Reach vs Author Degree
    df_posts['unique_reach'] = df_posts.index.map(post_reach_counts).fillna(0).astype(int)
    reach_by_degree = df_posts.groupby('degree')['unique_reach'].mean().to_dict()
    metrics['reach_per_degree'] = reach_by_degree

    print(f'End computing metrics for {network}-{recsys}-{run}')
    
    return metrics

def run_etl_pipeline():
    print('run_etl_pipeline: Starting ETL pipeline for all experiments...')
    """Iterates through all experiments, computes or loads cache."""
    total_tasks = len(NETWORKS) * len(RECSYSS) * len(RUNS)
    pbar = tqdm(total=total_tasks, desc="Processing Experiments", unit="run")
    
    # Structure: aggregated_data[network][recsys] = list of run_metrics (dicts)
    aggregated_data = defaultdict(lambda: defaultdict(list))

    for network in NETWORKS:
        for recsys in RECSYSS:
            for run in RUNS:
                print(f"Processing {network}-{recsys}-{run}...")
                cache_file = get_cache_path(network, recsys, run)
                
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        run_metrics = pickle.load(f)
                else:
                    run_metrics = compute_run_metrics(network, recsys, run)
                    if run_metrics:
                        with open(cache_file, 'wb') as f:
                            print('saving metrics to cache...')
                            pickle.dump(run_metrics, f)
                    # aggressive garbage collection after heavy processing
                    gc.collect() 
                
                if run_metrics:
                    aggregated_data[network][recsys].append(run_metrics)
                
                pbar.update(1)
    
    pbar.close()
    return aggregated_data

# =============================================================================
# PART 2: PLOTTING & AGGREGATION
# =============================================================================

def bin_xy_data(x_vals, y_vals, bins):
    """
    Aggregates (X, Y) scatter data into bins to compute smooth means.
    Returns:
        bin_centers: center value of each bin
        bin_means: mean of y-values in each bin
    """
    
    # Create a DataFrame from x and y values
    df = pd.DataFrame({'x': x_vals, 'y': y_vals})
    
    # Assign each x-value to a bin
    df['bin'] = pd.cut(df['x'], bins)
    
    # Group by bin and compute the mean of y-values in each bin
    grouped = df.groupby('bin', observed=False)['y'].mean()
    
    # Compute the bin centers (middle point of each bin interval)
    bin_centers = [b.mid for b in grouped.index]
    
    return np.array(bin_centers), grouped.values


def aggregate_runs_binned(runs_dicts, bins=DEGREE_BINS):
    """
    Flattens multiple runs of {x: y} dicts and bins them.
    Used for plotting smoothed 'Reach vs Degree' curves.
    """
    print("aggregate_runs_binned: Starting aggregation over runs...")  # Debug
    
    all_x = []  # Will store all x-values across runs
    all_y = []  # Will store all y-values across runs
    
    for idx, d in enumerate(runs_dicts):
        if not d: 
            continue
        # Extend all_x and all_y with keys/values of this run
        all_x.extend(list(d.keys()))
        all_y.extend(list(d.values()))
    
    if not all_x: 
        return np.array([]), np.array([])
    
    # Bin the flattened data using bin_xy_data
    return bin_xy_data(all_x, all_y, bins)


def aggregate_xy(run_dicts, key_x_strategy='union'):
    """
    Averages X-Y data across multiple runs (dicts {x: y}).
    key_x_strategy:
        'union' -> include all x-values seen in any run
        'intersection' -> include only x-values present in all runs
    Returns:
        sorted_keys: array of x-values
        means: array of mean y-values across runs
        stds: array of std deviations across runs
    """
    
    if not run_dicts:
        return None, None, None

    # Collect all unique x-values across runs
    all_keys = set()
    for idx, d in enumerate(run_dicts):
        all_keys.update(d.keys())
    
    sorted_keys = sorted(list(all_keys))
    
    means = []  # mean y-value per x
    stds = []   # standard deviation per x
    
    # Compute mean and std for each x-value
    for k in sorted_keys:
        vals = [d.get(k, 0) for d in run_dicts]  # fill missing with 0
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    
    return np.array(sorted_keys), np.array(means), np.array(stds)

def plot_gap_distribution(data):
    """
    Plots the distribution of 'Gaps' (time in rounds between two recommendations 
    of the same post).
    """
    print("Generating Gap Distribution Plots...")
    
    fig, axes = plt.subplots(2, sharey=True, figsize=(12, 8))
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        for recsys in RECSYSS:
            # Extract gap histograms
            runs_data = [r['gap_hist'] for r in data[network][recsys]]
            x, y, err = aggregate_xy(runs_data)
            
            # Normalize to probability density
            # y contains counts, so we divide by total count per run approx
            # A safer way with aggregate_xy output is to normalize the means:
            total = np.sum(y)
            if total > 0:
                y_norm = y / total
                # Error propagation is complex, simpler to just scale error similarly
                err_norm = err / total
                
                # Plot
                ax.plot(x, y_norm, 'o-', label=recsys, markersize=4, alpha=0.7)
                ax.fill_between(x, y_norm - err_norm, y_norm + err_norm, alpha=0.2)
            
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(f"Network: {network}")
        ax.set_ylabel("Probability P(Gap)")
        ax.legend()
    
    axes[-1].set_xlabel("Gap Size (Rounds between Recs)")
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], '8_gap_distribution.png'))
    plt.close()

def plot_quality_bias(data):
    """
    Plots the Average Reactions of Recommended posts vs Global Average.
    """
    print("Generating Quality Bias Plots...")
    
    fig, axes = plt.subplots(1, len(NETWORKS), figsize=(12, 6), squeeze=False)
    axes = axes.flatten()
    
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        
        # Prepare data for bar chart
        means_rec = []
        stds_rec = []
        baseline_all = [] # To compute global average of "All Posts"
        
        for recsys in RECSYSS:
            # Get avg reactions for RECOMMENDED posts
            rec_vals = [r['avg_reactions_rec'] for r in data[network][recsys]]
            means_rec.append(np.mean(rec_vals))
            stds_rec.append(np.std(rec_vals))
            
            # Get avg reactions for ALL posts (Baseline)
            all_vals = [r['avg_reactions_all'] for r in data[network][recsys]]
            baseline_all.extend(all_vals)
            
        # Global baseline for this network
        global_avg = np.mean(baseline_all)
        global_std = np.std(baseline_all)
        
        # Plot Bars
        x_pos = np.arange(len(RECSYSS))
        ax.bar(x_pos, means_rec, yerr=stds_rec, capsize=5, alpha=0.7, 
               color=[plt.rcParams['axes.prop_cycle'].by_key()['color'][k%4] for k in range(len(RECSYSS))])
        
        # --- FIX: Pass explicit lists to axhline and fill_between ---
        left = x_pos[0] - 0.5
        right = x_pos[-1] + 0.5
        ax.axhline(global_avg, color='black', linestyle='--', linewidth=2, label="Avg of All Posts (Pool)")
        
        # Pass arrays of coordinates, not scalars
        ax.fill_between([left, right], 
                        [global_avg - global_std, global_avg - global_std], 
                        [global_avg + global_std, global_avg + global_std], 
                        color='black', alpha=0.1)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(RECSYSS)
        ax.set_title(f"Network: {network}")
        ax.set_ylabel("Avg Reactions")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], '9_quality_bias.png'))
    plt.close()


def plot_comparative_diff(data):
    """
    Plots the DIFFERENCE in Unique Reach vs Degree between strategies.
    Baseline is usually 'RC' (Reverse Chrono).
    """
    print("Generating Comparative Difference Plots...")
    
    BASELINE = 'RC'
    if BASELINE not in RECSYSS:
        print(f"Skipping diff plots: Baseline {BASELINE} not in data.")
        return

    fig, axes = plt.subplots(1, len(NETWORKS), figsize=(14, 6), squeeze=False)
    axes = axes.flatten()

    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        
        # 1. Get Baseline Data (RC)
        baseline_runs = [r['reach_per_degree'] for r in data[network][BASELINE]]
        # We need the raw binned means for the baseline
        # Note: To diff correctly, we aggregate ALL runs to get a stable baseline curve
        base_x, base_y = aggregate_runs_binned(baseline_runs)
        
        # 2. Compare others to Baseline
        for recsys in RECSYSS:
            if recsys == BASELINE: continue
            
            target_runs = [r['reach_per_degree'] for r in data[network][recsys]]
            target_x, target_y = aggregate_runs_binned(target_runs)
            
            # Calculate Diff (Target - Baseline)
            # Ensure shapes match (binning guarantees this if bins are fixed)
            # Handle NaNs created by empty bins
            mask = ~np.isnan(base_y) & ~np.isnan(target_y)
            
            diff = target_y[mask] - base_y[mask]
            centers = base_x[mask]
            
            ax.plot(centers, diff, 'o-', label=f"{recsys} - {BASELINE}", lw=2)

        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f"Network: {network}\nReach Difference vs Degree")
        ax.set_xlabel("Author Degree")
        ax.set_ylabel(fr"$\Delta$ Avg Unique Reach (vs {BASELINE})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], 'diff_reach_vs_degree.png'), facecolor='white', dpi=600)
    plt.close()

def plot_absolute_reach(data):
    """Plots absolute Unique Reach vs Degree."""
    print("Generating Absolute Reach Plots...")
    
    fig, axes = plt.subplots(1, len(NETWORKS), figsize=(14, 6), squeeze=False)
    axes = axes.flatten()

    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        for recsys in RECSYSS:
            runs_data = []
            for r in data[network][recsys]:
                runs_data.append(r['reach_per_degree'])                
            x, y = aggregate_runs_binned(runs_data)
            
            # Filter NaNs
            mask = ~np.isnan(y)
            ax.plot(x[mask], y[mask], 'o-', label=recsys, alpha=0.8)

        ax.set_yscale('log')
        ax.set_title(f"Network: {network}")
        ax.set_ylabel("Avg Unique Reach (Users)")
        ax.set_xlabel("Author Degree")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], 'absolute_reach_vs_degree.png'), facecolor='white', dpi=600)
    plt.close()

def plot_all(data):
    print("Generating Plots...")
    
    # --- Plot 1: Average Recommendations per Round ---
    fig, axes = plt.subplots(2, sharey=True, figsize=(12, 8))
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        for recsys in RECSYSS:
            runs_data = [r['recs_per_round'] for r in data[network][recsys]]
            x, y, err = aggregate_xy(runs_data)
            
            # Cumulative
            y_cum = np.cumsum(y)
            # Std of cumulative is complex, simply accumulating std is an approximation (upper bound)
            # For simplicity in visualization, we plot cumsum of means
            
            ax.plot(x, y_cum, label=recsys)
            # ax.fill_between(x, y_cum - err, y_cum + err, alpha=0.2) # Std on cumulative is tricky
        
        ax.set_title(f"Network: {network}")
        ax.set_ylabel("Cumulative Recommendations")
        ax.legend()
    
    axes[-1].set_xlabel("Round")
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], '1_recs_per_round_cumulative.png'), facecolor='white', dpi=600)
    plt.close()

    # --- Plot 2: Histogram of Recommendations per Post ---
    fig, axes = plt.subplots(2, sharey=True, figsize=(12, 8))
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        for recsys in RECSYSS:
            runs_data = [r['rec_count_hist'] for r in data[network][recsys]]
            x, y, err = aggregate_xy(runs_data)
            
            # Filter out x=0 if it exists and dominates
            mask = x > 0
            
            line = ax.plot(x[mask], y[mask], 'o-', label=recsys, markersize=4)[0]
            ax.fill_between(x[mask], y[mask]-err[mask], y[mask]+err[mask], color=line.get_color(), alpha=0.2)
            
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(f"Network: {network}")
        ax.set_ylabel("Number of Posts (Log)")
        ax.legend()
    
    axes[-1].set_xlabel("Number of Recommendations Received (k)")
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], '2_rec_count_distribution.png'), facecolor='white', dpi=600)
    plt.close()

    # --- Plot 3: Consecutive Recommendation Probability ---
    # Bar chart
    rows = []
    for network in NETWORKS:
        for recsys in RECSYSS:
            probs = [r['consec_prob'] for r in data[network][recsys]]
            rows.append({
                'Network': network,
                'RecSys': recsys,
                'Prob': np.mean(probs),
                'Std': np.std(probs)
            })
    df_consec = pd.DataFrame(rows)
    
    # Pivot for plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(NETWORKS))
    width = 0.2
    
    for i, recsys in enumerate(RECSYSS):
        subset = df_consec[df_consec['RecSys'] == recsys]
        # Align bars
        offset = (i - len(RECSYSS)/2) * width + width/2
        ax.bar(x + offset, subset['Prob'], width, yerr=subset['Std'], label=recsys, capsize=5)
        
    ax.set_xticks(x)
    ax.set_xticklabels(NETWORKS)
    ax.set_ylabel("Probability of Consecutive Rec.")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], '3_consecutive_prob.png'), facecolor='white', dpi=600)
    plt.close()

    # --- Plot 4: Streak Length Distribution ---
    fig, axes = plt.subplots(2, figsize=(12, 8))
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        for j, recsys in enumerate(RECSYSS):
            runs_data = [r['streak_hist'] for r in data[network][recsys]]
            x, y, err = aggregate_xy(runs_data)
            
            # Normalize to percentage
            total = np.sum(y)
            y_pct = (y / total) * 100
            err_pct = (err / total) * 100
            
            # Jitter x slightly
            ax.errorbar(x + (j*0.1), y_pct, yerr=err_pct, fmt='o-', label=recsys, alpha=0.7)
            
        ax.set_title(f"Network: {network}")
        ax.set_xlabel("Consecutive Streak Length")
        ax.set_ylabel("% of Streaks")
        ax.set_yscale('log')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], '4_streak_distribution.png'))
    plt.close()

    # --- Plot 5: Fraction of Posts Recommended per Round ---
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        for recsys in RECSYSS:
            runs_data = [r['frac_rec_per_round'] for r in data[network][recsys]]
            x, y, err = aggregate_xy(runs_data)
            
            ax.plot(x, y, label=recsys)
            ax.fill_between(x, y-err, y+err, alpha=0.2)
            
        ax.set_title(f"Network: {network}")
        ax.set_ylabel("Fraction of Posts Recommended")
        ax.legend()
        
    axes[-1].set_xlabel("Round")
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], '5_fraction_recommended.png'), facecolor='white', dpi=600)
    plt.close()

    # --- Plot 6: Reactions vs Number of Recommendations ---
    fig, axes = plt.subplots(2, figsize=(12, 8))
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        for recsys in RECSYSS:
            runs_data = [r['reactions_by_rec_count'] for r in data[network][recsys]]
            x, y, err = aggregate_xy(runs_data)
            
            # Sort by x
            idx = np.argsort(x)
            x, y, err = x[idx], y[idx], err[idx]
            
            ax.plot(x, y, label=recsys)
            ax.fill_between(x, y-err, y+err, alpha=0.2)
            
        ax.set_title(f"Network: {network}")
        ax.set_ylabel("Avg Reactions")
        ax.set_xlabel("Number of Recommendations")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], '6_reactions_vs_recs.png'), facecolor='white', dpi=600)
    plt.close()
    
    # --- Plot 7: Recommendations vs Author Degree ---
    fig, axes = plt.subplots(2, figsize=(12, 8))
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        for recsys in RECSYSS:
            runs_data = [r['rec_per_degree'] for r in data[network][recsys]]
            x, y, err = aggregate_xy(runs_data)
            
            if x is not None:
                # Sort by degree
                idx = np.argsort(x)
                x, y, err = x[idx], y[idx], err[idx]
                
                ax.plot(x, y, 'o-', label=recsys, markersize=4, alpha=0.7)
                ax.fill_between(x, y-err, y+err, alpha=0.2)
            
        ax.set_title(f"Network: {network}")
        ax.set_ylabel("Avg Recommendations per Post")
        ax.set_xlabel("Author Degree")
        ax.set_yscale('log')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['figs'], '7_recs_vs_degree.png'), facecolor='white', dpi=600)
    plt.close()
    
    plot_absolute_reach(data)
    plot_comparative_diff(data)
    plot_gap_distribution(data)
    plot_quality_bias(data)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    old_stdout = sys.stdout

    log_file = open("analysis.log","w")

    sys.stdout = log_file

    print("Starting Analysis Pipeline...")
    
    # 1. Compute or Load Data
    data = run_etl_pipeline()
    
    # 2. Generate Figures
    plot_all(data)
    
    print(f"Done! Processed data saved to '{DIRS['data']}' and figures to '{DIRS['figs']}'.")
    
    sys.stdout = old_stdout

    log_file.close()
    