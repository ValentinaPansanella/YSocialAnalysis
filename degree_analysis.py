import os
import glob
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm
from matplotlib import rcParams
from cycler import cycler

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = "experiments_recsys"  # The folder containing the experiment subfolders
NETWORKS = ['BA', 'ER']
RECSYSS = ['F', 'FP', 'P', 'RC']
RUNS = list(range(10))

# Binning configuration for Degrees
# We use bins to average out the noise of exact degree values.
DEGREE_BINS = np.linspace(0, 200, 21) # 0, 10, 20... 200

# Plot output directory
FIG_DIR = "figs/figs_degree"
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
# HELPER FUNCTIONS
# =============================================================================

def get_experiment_paths(network, recsys, run):
    """Constructs file paths based on naming convention."""
    folder_name = f"{network}_{recsys}_{run}"
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    db_path = os.path.join(folder_path, "database_server.db")
    
    # Find CSV (name might vary slightly, so we glob)
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    csv_path = csv_files[0] if csv_files else None
    
    return db_path, csv_path

def compute_author_degrees(csv_path):
    """
    Reads the edge list CSV and computes degree per username.
    Assumes undirected graph where edges appear twice (A,B) and (B,A).
    """
    if not csv_path:
        return {}
    
    try:
        # Load CSV. Column 0 is Source, Column 1 is Target.
        # Since edges appear twice (both directions), counting occurrences 
        # in Column 0 gives the degree.
        df = pd.read_csv(csv_path, header=None, names=['source', 'target'])
        degree_counts = df['source'].value_counts().to_dict()
        return degree_counts
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return {}

def process_run(network, recsys, run):
    """
    Main ETL function for a single run.
    Returns a DataFrame aggregated by Degree Bin with metrics.
    """
    db_path, csv_path = get_experiment_paths(network, recsys, run)
    
    if not os.path.exists(db_path) or not csv_path:
        return None

    # 1. Get Degrees (Username -> Degree)
    username_degrees = compute_author_degrees(csv_path)

    metrics_by_degree = defaultdict(lambda: {'count': 0, 'total_recs': 0, 'total_reach': 0})

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # 2. Map Usernames to User IDs
        # We need this because CSV uses usernames, but DB tables use integer user_ids
        df_users = pd.read_sql("SELECT id, username FROM user_mgmt", conn)
        # Create map: user_id -> degree
        # We map user_id -> username -> degree
        id_to_degree = {}
        for _, row in df_users.iterrows():
            u_id = row['id']
            u_name = row['username']
            # Default to 0 if not in CSV (isolated node)
            deg = username_degrees.get(u_name, 0)
            id_to_degree[u_id] = deg

        # 3. Load Posts (Author mapping)
        # We need to know who wrote which post
        query_post = "SELECT id, user_id FROM post"
        df_posts = pd.read_sql(query_post, conn)
        post_authors = df_posts.set_index('id')['user_id'].to_dict()

        # 4. Process Recommendations
        # We stream this to be memory efficient
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, post_ids FROM recommendations")
        
        # Track metrics per post
        post_stats = defaultdict(lambda: {'recs': 0, 'viewers': set()})
        
        while True:
            rows = cursor.fetchmany(10000)
            if not rows:
                break
            
            for row in rows:
                viewer_id = row['user_id']
                post_ids_str = row['post_ids']
                
                if not post_ids_str:
                    continue
                
                try:
                    pids = [int(x) for x in post_ids_str.split('|')]
                except ValueError:
                    continue
                
                for pid in pids:
                    post_stats[pid]['recs'] += 1
                    post_stats[pid]['viewers'].add(viewer_id)

    # 5. Aggregate metrics by Author Degree
    # We aggregate into a list first: [(degree, recs, reach), ...]
    data_points = []

    for pid, stats in post_stats.items():
        author_id = post_authors.get(pid)
        if author_id is None: 
            continue # Should not happen unless integrity error
        
        degree = id_to_degree.get(author_id, 0)
        recs = stats['recs']
        reach = len(stats['viewers'])
        
        data_points.append({'degree': degree, 'recs': recs, 'reach': reach})

    # Convert to DataFrame for easy binning
    if not data_points:
        return None

    df = pd.DataFrame(data_points)
    
    # Bin the degrees
    df['bin'] = pd.cut(df['degree'], bins=DEGREE_BINS, labels=DEGREE_BINS[:-1])
    
    # Group by Bin and compute Means
    # We want: Average Recs per Post for authors in this bin
    #          Average Reach per Post for authors in this bin
    grouped = df.groupby('bin', observed=False)[['recs', 'reach']].mean()
    
    return grouped

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def collect_data():
    """Iterates over all configs and collects averaged data."""
    # Storage structure: data[network][recsys] = list of DataFrames (one per run)
    raw_data = defaultdict(lambda: defaultdict(list))
    
    total_ops = len(NETWORKS) * len(RECSYSS) * len(RUNS)
    pbar = tqdm(total=total_ops, desc="Processing Runs")

    for network in NETWORKS:
        for recsys in RECSYSS:
            for run in RUNS:
                df_run = process_run(network, recsys, run)
                if df_run is not None:
                    raw_data[network][recsys].append(df_run)
                pbar.update(1)
    
    pbar.close()
    return raw_data

def aggregate_over_runs(raw_data):
    """
    Computes Mean and Std over the runs.
    Returns: agg_data[network][recsys] = DataFrame with columns [recs_mean, recs_std, reach_mean, reach_std]
    """
    agg_data = defaultdict(dict)
    
    for network in NETWORKS:
        for recsys in RECSYSS:
            dfs = raw_data[network][recsys]
            if not dfs:
                continue
            
            # Concatenate all runs into one DF, preserving index (the bins)
            combined = pd.concat(dfs)
            
            # Group by the index (Degree Bin) and compute mean/std across runs
            stats = combined.groupby(combined.index)[['recs', 'reach']].agg(['mean', 'std'])
            
            # Flatten columns
            stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
            agg_data[network][recsys] = stats
            
    return agg_data

# =============================================================================
# PLOTTING
# =============================================================================

def plot_metric(agg_data, metric_prefix, title_suffix, filename):
    """Generic plotter for Recs and Reach."""
    
    fig, axes = plt.subplots(1, len(NETWORKS), figsize=(14, 6))
    if len(NETWORKS) == 1: axes = [axes]
    
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        
        for recsys in RECSYSS:
            if recsys not in agg_data[network]: continue
            
            df = agg_data[network][recsys]
            x = df.index.astype(float) # Bins
            y = df[f'{metric_prefix}_mean']
            err = df[f'{metric_prefix}_std']
            
            # Plot
            ax.plot(x, y, 'o-', label=recsys, markersize=4)
            ax.fill_between(x, y - err, y + err, alpha=0.2)
        
        ax.set_title(f"{network} - {title_suffix}")
        ax.set_xlabel("Author Degree (Bin)")
        ax.set_ylabel(title_suffix)
        if metric_prefix == 'recs':
            ax.set_yscale('log') # Recs usually vary wildly
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename))
    print(f"Saved {filename}")
    plt.close()

def plot_delta_reach(agg_data):
    """
    Plots (Method Reach - RC Reach).
    Baseline is RC.
    """
    fig, axes = plt.subplots(1, len(NETWORKS), figsize=(14, 6))
    if len(NETWORKS) == 1: axes = [axes]
    
    for i, network in enumerate(NETWORKS):
        ax = axes[i]
        
        # Get Baseline (RC)
        if 'RC' not in agg_data[network]:
            print(f"Skipping Delta plot for {network}: No RC data found.")
            continue
            
        rc_df = agg_data[network]['RC']
        rc_mean = rc_df['reach_mean']
        
        for recsys in RECSYSS:
            if recsys == 'RC': continue # Skip baseline vs baseline
            if recsys not in agg_data[network]: continue
            
            df = agg_data[network][recsys]
            
            # Align indices just in case (though bins are fixed)
            # We assume bins match perfectly due to fixed range
            x = df.index.astype(float)
            
            # Calculate Difference
            # Note: We propagate errors simply by summing variances (sqrt(std1^2 + std2^2))
            diff_mean = df['reach_mean'] - rc_mean
            
            # Standard Error of the difference
            diff_std = np.sqrt(df['reach_std']**2 + rc_df['reach_std']**2)
            
            ax.plot(x, diff_mean, 'o-', label=f"{recsys} - RC", markersize=4)
            ax.fill_between(x, diff_mean - diff_std, diff_mean + diff_std, alpha=0.15)
            
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f"{network} - Delta Unique Reach (vs RC)")
        ax.set_xlabel("Author Degree (Bin)")
        ax.set_ylabel("$\Delta$ Reach (Users)")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "delta_reach_vs_degree.png"))
    print("Saved delta_reach_vs_degree.png")
    plt.close()

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("1. Collecting and processing data...")
    raw_results = collect_data()
    
    print("2. Aggregating over runs...")
    averaged_results = aggregate_over_runs(raw_results)
    
    print("3. Generating Plots...")
    # Plot b) Amount of recommendations vs Degree
    plot_metric(averaged_results, 'recs', 'Avg Recommendations per Post', 'recs_vs_degree.png')
    
    # Plot c) Unique Reach vs Degree
    plot_metric(averaged_results, 'reach', 'Avg Unique Reach per Post', 'reach_vs_degree.png')
    
    # Plot d) Delta Reach vs Degree (Difference with RC)
    plot_delta_reach(averaged_results)
    
    print("Done.")