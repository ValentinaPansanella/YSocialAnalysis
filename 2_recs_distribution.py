import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import powerlaw

from utils import get_db_path, pids_to_list, post_id_mapping
from utils_figures import apply_plot_style

BASE_DIRS = ['old_data', 'data']
NETWORKS = ['BA', 'ER']
RECSYSS = ['F', 'RC', 'P', 'FP', 'hybrid']
RUNS = list(range(10))

LOG_FILE = "logs/recs_distribution.log"
os.makedirs("logs", exist_ok=True)

# ==========================================================
# PLOT DISTRIBUTION FROM A SINGLE RUN
# ==========================================================
def plot(frequencies, base_dir, network, recsys, run, entity='posts'):
    """
    Plots the distribution of the number of recommendations.
    """

    # Normalize counts to probabilities
    frequencies['prob'] = frequencies['num_entities'] / frequencies['num_entities'].sum()
    
    apply_plot_style()
    
    fig_dir = os.path.join('res', base_dir, 'recs_distribution')
    os.makedirs(fig_dir, exist_ok=True)
    
    # ----- Bar plot -----
    fig, axes = plt.subplots(figsize=(12, 4), ncols=2)
    ax = axes[0]
    ax.bar(frequencies['num_recommendations'], frequencies['prob'], color='skyblue')
    ax.set_xlabel('Number of recommendations')
    ax.set_ylabel('Probability')
    ax.set_xlim(frequencies['num_recommendations'].min() -1 , frequencies['num_recommendations'].max()+1)
    ax.set_ylim(frequencies['prob'][frequencies['prob']>0].min(), frequencies['prob'].max()*1.1)
    # Annotate formula
    ax.text(
        0.95, 0.95, r'$P(x) = \frac{\text{num\_posts for x}}{\sum \text{num\_posts}}$',
        transform=ax.transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.6)
    )
    
    # ----- Scatter log-log plot -----
    ax = axes[1]
    ax.scatter(frequencies['num_recommendations'], frequencies['prob'], color='red')
    ax.set_xlabel('Number of recommendations')
    ax.set_ylabel('Probability')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(frequencies['num_recommendations'].min()*0.8, frequencies['num_recommendations'].max()*1.1)
    ax.set_ylim(frequencies['prob'][frequencies['prob']>0].min()*0.8, frequencies['prob'].max()*1.1)

    # Avoid cutting data
    plt.suptitle(f'Distribution for {network}-{recsys}-run{run}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{network}-{recsys}-{run}-num-recs-prob.png'), dpi=150)
    plt.close()


# ==========================================================
# LOAD DISTRIBUTION FROM A SINGLE RUN
# ==========================================================

def load_run_distribution(base_dir, network, recsys, run):
    """
    Returns a DataFrame:
        num_recommendations | num_entities
    """
    db_path = get_db_path(base_dir, network, recsys, run)
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        recommendations = pd.read_sql("SELECT post_ids, round AS rec_round FROM recommendations", conn)
        posts = pd.read_sql("SELECT id AS post_id, round AS creation_round FROM post", conn)
        
    print(posts.groupby('creation_round').size())
    print(recommendations.groupby('rec_round').size())


    # explode recommended posts
    recommendations['post_id'] = recommendations['post_ids'].apply(pids_to_list)
    recommendations = recommendations.explode('post_id').dropna(subset=['post_id'])

    mapping = post_id_mapping(base_dir, network, recsys, run)
    recommendations['post_id'] = recommendations['post_id'].map(mapping)
    recommendations = recommendations.dropna(subset=['post_id'])
    posts['post_id'] = posts['post_id'].map(mapping)
    posts = posts.dropna(subset=['post_id'])
    
    rec_df = recommendations.merge(posts, on='post_id', how='left')

    # count recommendations per post
    post_counts = recommendations['post_id'].value_counts()

    # frequency distribution
    freq = post_counts.groupby(post_counts).size().reset_index(name='num_entities')
    freq.columns = ['num_recommendations', 'num_entities']
    
    plot(freq, base_dir, network, recsys, run)

    return freq.sort_values('num_recommendations').reset_index(drop=True)


# ==========================================================
# GLOBAL DISTRIBUTION (all runs pooled)
# ==========================================================

def compute_global_distribution(all_runs):
    all_counts = []

    for rc in all_runs:
        repeated = np.repeat(rc['num_recommendations'], rc['num_entities'])
        all_counts.extend(repeated)

    if not all_counts:
        return None

    unique, counts = np.unique(all_counts, return_counts=True)
    probs = counts / counts.sum()

    return {
        'bins': unique.tolist(),
        'probs': probs.tolist()
    }


# ==========================================================
# MEAN DISTRIBUTION (average over runs)
# ==========================================================

def compute_mean_distribution(all_runs):
    if not all_runs:
        return None

    max_count = max(rc['num_recommendations'].max() for rc in all_runs)
    common_x = np.arange(1, max_count + 1)

    aligned = []
    for rc in all_runs:
        counts = (
            rc.set_index('num_recommendations')['num_entities']
              .reindex(common_x, fill_value=0)
        )
        density = counts / counts.sum()
        aligned.append(density.values)

    df_aligned = pd.DataFrame(aligned, columns=common_x)

    return {
        'bins': common_x.tolist(),
        'mean': df_aligned.mean().values.tolist(),
        'std': df_aligned.std().values.tolist()
    }


# ==========================================================
# POWER-LAW FIT (per run)
# ==========================================================

def compute_powerlaw_metrics(all_runs):
    alphas, kmins = [], []

    for rc in all_runs:
        repeated = np.repeat(rc['num_recommendations'], rc['num_entities'])
        repeated = repeated[repeated > 0]

        if len(repeated) < 10:
            continue

        fit = powerlaw.Fit(repeated, discrete=True, verbose=False)
        alphas.append(fit.power_law.alpha)
        kmins.append(fit.power_law.xmin)

    if not alphas:
        return None

    return {
        'alphas': alphas,
        'kmins': kmins,
        'mean_alpha': float(np.mean(alphas)),
        'std_alpha': float(np.std(alphas))
    }


# ==========================================================
# MAIN AGGREGATION PIPELINE
# ==========================================================

def compute_all_distributions(base_dir):

    global_metrics = defaultdict(dict)
    mean_metrics = defaultdict(dict)
    pl_metrics = defaultdict(dict)

    total_steps = len(NETWORKS) * len(RECSYSS) * len(RUNS)
    pbar = tqdm(total=total_steps, desc="Processing runs")

    for network in NETWORKS:
        for recsys in RECSYSS:

            all_runs = []

            # collect distributions for each run
            for run in RUNS:
                rc = load_run_distribution(base_dir, network, recsys, run)
                if rc is not None and not rc.empty:
                    all_runs.append(rc)
                pbar.update(1)

            # compute metrics
            global_metrics[network][recsys] = compute_global_distribution(all_runs)
            mean_metrics[network][recsys] = compute_mean_distribution(all_runs)
            #pl_metrics[network][recsys] = compute_powerlaw_metrics(all_runs)

    pbar.close()

    return global_metrics, mean_metrics


# ==========================================================
# SAVE METRICS
# ==========================================================

def save_metrics(base_dir, global_m, mean_m):

    res_dir = f"res/{base_dir}/recs_distribution/"
    os.makedirs(res_dir, exist_ok=True)

    with open(os.path.join(res_dir, "global_distribution_metrics.json"), 'w') as f:
        json.dump(global_m, f)

    with open(os.path.join(res_dir, "mean_distribution_metrics.json"), 'w') as f:
        json.dump(mean_m, f)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    old_stdout = sys.stdout
    log_file = open(LOG_FILE, "w")
    sys.stdout = log_file

    print("Starting Analysis Pipeline...")

    for base_dir in BASE_DIRS:
        print(f"Processing {base_dir}...")
        global_m, mean_m = compute_all_distributions(base_dir)
        save_metrics(base_dir, global_m, mean_m)

    print("Completed Analysis Pipeline.")

    sys.stdout = old_stdout
    log_file.close()