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

from utils import *
from utils_figures import *

matplotlib.use('Agg') 


# ============================================
# RUN FUNCTIONS
# ============================================

def run_post_analysis(network, recsys, run):
    """
    Returns a DataFrame with the distribution of how often each post is recommended.
    Columns: num_recommendations, num_entities
    """
    db_path = get_db_path(BASE_DIR, network, recsys, run)
    if not os.path.exists(db_path):
        print(f"Warning: DB not found at {db_path}")
        return None

    with sqlite3.connect(db_path) as conn:
        df_recommendations = pd.read_sql("SELECT round, post_ids, user_id FROM recommendations", conn)
        df_recommendations['post_id'] = df_recommendations['post_ids'].apply(pids_to_list)
        df_recommendations = df_recommendations.explode('post_id')

        # Count recommendations per post
        post_counts = df_recommendations['post_id'].value_counts().rename("num_recommendations")

        # Frequency distribution: how many posts got recommended x times
        freq_counts = post_counts.groupby(post_counts).size().reset_index(name='num_entities')
        
        res = freq_counts.sort_values('num_recommendations').reset_index(drop=True)
        
        print(f"Frequency counts (post) for {network}-{recsys}-{run}:")
        print(res)

        return res

def run_user_analysis(network, recsys, run):
    """
    Returns a DataFrame with the distribution of how often each author is recommended.
    Columns: num_recommendations, num_entities
    """
    db_path = get_db_path(BASE_DIR, network, recsys, run)
    if not os.path.exists(db_path):
        print(f"Warning: DB not found at {db_path}")
        return None

    with sqlite3.connect(db_path) as conn:
        df_rec = pd.read_sql("SELECT round, post_ids, user_id FROM recommendations", conn)
        df_rec['post_id'] = df_rec['post_ids'].apply(pids_to_list)
        df_rec = df_rec.explode('post_id')

        df_posts = pd.read_sql("SELECT id AS post_id, user_id AS author_id FROM post", conn)
        df_rec = df_rec.merge(df_posts, on='post_id', how='left')

        # Count recommendations per author
        author_counts = df_rec['author_id'].value_counts().rename("num_recommendations")

        # Frequency distribution: how many authors got recommended x times
        freq_counts = author_counts.groupby(author_counts).size().reset_index(name='num_entities')
        
        res = freq_counts.sort_values('num_recommendations').reset_index(drop=True)
        
        print(f"Frequency counts (user) for {network}-{recsys}-{run}:")
        print(res)

        return res

# ============================================
# AGGREGATION FUNCTIONS
# ============================================
def agg_runs(run_func, n_bins=20):
    metrics = defaultdict(dict)
    
    total_steps = len(NETWORKS) * len(RECSYSS) * len(RUNS)
    pbar = tqdm(total=total_steps, desc=f'Processing runs ({run_func.__name__})')

    for network in NETWORKS:
        for recsys in RECSYSS:
            all_runs = []
            for run in RUNS:
                rc = run_func(network, recsys, run)
                pbar.update(1)
                if rc is not None:
                    all_runs.append(rc)

            if not all_runs:
                continue

            # max number of recommendations across all runs
            max_count = max(rc['num_recommendations'].max() for rc in all_runs)

            # Log-spaced bins
            bins = np.logspace(0, np.log10(max_count), num=n_bins, dtype=int)
            bins = np.unique(bins)

            aligned_runs = []
            for rc in all_runs:
                data_points = np.repeat(rc['num_recommendations'].values, rc['num_entities'].values)
                hist, _ = np.histogram(data_points, bins=bins, density=True)
                aligned_runs.append(hist)

            df_aligned = pd.DataFrame(aligned_runs)
            metrics[network][recsys] = {
                'mean': df_aligned.mean(axis=0),
                'std': df_aligned.std(axis=0),
                'bins': bins
            }
            print(f"Aggregated {len(all_runs)} runs for {network}-{recsys}")
            print(metrics[network][recsys])

    pbar.close()
    return metrics


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    old_stdout = sys.stdout
    log_file = open("res/recs_distribution.log","w")
    sys.stdout = log_file

    print("Starting Analysis Pipeline...")

    post_metrics = agg_runs(run_post_analysis, n_bins=20)
    user_metrics = agg_runs(run_user_analysis, n_bins=20)
    
    plot_recommendations_loglog(post_metrics, networks=['ER','BA'], save_path='figs/recommendations/posts/', analysis_type='post')
    plot_recommendations_loglog(user_metrics, networks=['ER','BA'], save_path='figs/recommendations/users/', analysis_type='user')

    print("Analysis Completed.")

    sys.stdout = old_stdout
    log_file.close()