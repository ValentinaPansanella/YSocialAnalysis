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


def run_analysis(
    base_dir, 
    network, 
    recsys, 
    run, 
    normalize_per_post: bool = False
):
    """
    Returns a dataframe with:
        author_id
        total_recommendations
        author_degree
        unique_reach
        delta_recs

    Args:
        normalize_per_post: if True, divide total_recommendations by number of posts by the author
    """

    print('Computing author degrees for:', network, recsys, run)
    degrees = compute_author_degrees(base_dir, network, recsys, run, by='id')
    all_users = list(degrees.keys())
    
    db_path = get_db_path(base_dir, network, recsys, run)
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return None

    with sqlite3.connect(db_path) as conn:
        df_rec = pd.read_sql(
            "SELECT round AS recommended_at, post_ids, user_id FROM recommendations",
            conn
        )
        df_rec['post_id'] = df_rec['post_ids'].apply(pids_to_list)
        df_rec = df_rec.explode('post_id')
        df_rec = df_rec.rename(columns={'user_id': 'viewer_id'}).drop(columns=['post_ids'])

        df_posts = pd.read_sql(
            "SELECT id AS post_id, user_id AS author_id FROM post",
            conn
        )
        df_rec = df_rec.merge(df_posts, on="post_id", how="left")

        # Count total posts per author (needed for normalization)
        if normalize_per_post:
            author_post_counts = df_posts.groupby('author_id')['post_id'].count()

    # Map author degree
    df_rec['author_degree'] = df_rec['author_id'].map(degrees)

    # 3. Aggregate at author level (your existing code)
    # author_stats = (
    #     df_rec.groupby('author_id')
    #     .agg(
    #         total_recommendations=('post_id', 'count'),
    #         author_degree=('author_degree', 'first'),
    #         unique_reach=('viewer_id', 'nunique')
    #     )
    #     .reset_index()
    # )

    # 4. Add delta to author stats
    author_stats = pd.DataFrame({'author_id': all_users})
    author_stats['author_degree'] = author_stats['author_id'].map(degrees)

    # 2. Filter out self-views for reach/viewed math
    df_rec_clean = df_rec[df_rec['viewer_id'] != df_rec['author_id']]

    # 3. Compute metrics
    recs_count = df_rec.groupby('author_id')['post_id'].count()
    reach_map = df_rec_clean.groupby('author_id')['viewer_id'].nunique()
    viewed_map = df_rec_clean.groupby('viewer_id')['author_id'].nunique()

    # 4. Map and aggressively fill NaN with 0
    author_stats['total_recommendations'] = author_stats['author_id'].map(recs_count).fillna(0)
    author_stats['unique_reach'] = author_stats['author_id'].map(reach_map).fillna(0)
    author_stats['authors_viewed'] = author_stats['author_id'].map(viewed_map).fillna(0)

    # 5. Compute Delta
    author_stats['delta_recs'] = author_stats['unique_reach'] - author_stats['authors_viewed']

    # THE ULTIMATE TEST:
    print(f"Total Network Delta (Must be 0): {author_stats['delta_recs'].sum()}")

    # 5. Your existing normalization code
    if normalize_per_post:
        # Divide total_recommendations by number of posts
        author_stats['total_recommendations_normalized'] = (
            author_stats['total_recommendations'] / author_stats['author_id'].map(author_post_counts)
        )
    
    print(f"Run {network}-{recsys}-{run}: {len(author_stats)} authors with recommendations")
    print(author_stats.sort_values('author_degree'))
    return author_stats

# ==================================================
# 2️⃣ AGGREGATE OVER RUNS
# ==================================================

def aggregate_runs(base_dir:str, 
                   networks: list[str] = ['ER', 'BA'], 
                   recsyss: list[str] = ['F', 'RC'],
                   runs: list[int] = list(range(10)), 
                   n_bins:int=20,
                   metrics_to_compute: list[str] = ['total_recommendations', 'total_recommendations_normalized','unique_reach', 'delta_recs'],
): 
    """
    Aggregates runs and computes binned mean/STD for:
        - total_recommendations
        - unique_reach
    Returns a nested dict:
        metrics[network][recsys] = {
            'total_recommendations': {'mean': ..., 'std': ..., 'bins': ...},
            'unique_reach': {'mean': ..., 'std': ..., 'bins': ...}
        }
    """
    
    metrics = defaultdict(dict)
    total_steps = len(networks) * len(recsyss) * len(runs)
    pbar = tqdm(total=total_steps, desc="Processing runs")

    for network in networks:
        for recsys in recsyss:

            all_runs = []

            for run in runs:
                print('Processing:', network, recsys, run)
                df = run_analysis(base_dir, network, recsys, run, normalize_per_post=True)
                pbar.update(1)
                if df is not None:
                    all_runs.append(df)

            if not all_runs:
                continue

            # ----------------------------------------
            # Define fixed log bins across runs
            # ----------------------------------------
            all_degrees = np.concatenate([df['author_degree'].dropna().values for df in all_runs])
            bins = np.logspace(0, np.log10(max(all_degrees)), num=n_bins, dtype=int)
            bins = np.unique(bins)

            # ----------------------------------------
            # Compute per-run binned means for all metrics
            # ----------------------------------------
            aligned = {metric: [] for metric in metrics_to_compute}

            for df in all_runs:
                df = df.copy()
                df['degree_bin'] = pd.cut(df['author_degree'], bins=bins, include_lowest=True)

                for metric_name in metrics_to_compute:
                    if metric_name in df.columns:
                        bin_means = df.groupby('degree_bin')[metric_name].mean()
                        bin_means = bin_means.reindex(pd.IntervalIndex.from_breaks(bins), fill_value=np.nan)
                        aligned[metric_name].append(bin_means.values)

            # Convert to DataFrame for aggregation and store in metrics dict
            metrics[network][recsys] = {}
            for metric_name, values_list in aligned.items():
                df_metric = pd.DataFrame(values_list) if values_list else pd.DataFrame()
                metrics[network][recsys][metric_name] = {
                    'mean': df_metric.mean(axis=0) if not df_metric.empty else np.nan,
                    'std': df_metric.std(axis=0) if not df_metric.empty else np.nan,
                    'bins': bins
                }

            print(f"Aggregated {len(all_runs)} runs for {network}-{recsys}")
            print(metrics[network][recsys])

    pbar.close()
    return metrics
    
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    old_stdout = sys.stdout
    log_file = open("res/recs_vs_degree.log", "w")
    sys.stdout = log_file

    print("Starting Analysis Pipeline...")

    # -------------------------------
    # Aggregate metrics across runs
    # -------------------------------
    metrics = aggregate_runs(BASE_DIR, NETWORKS, RECSYSS, RUNS, n_bins=20, metrics_to_compute=['total_recommendations', 'total_recommendations_normalized','unique_reach', 'delta_recs'])

    # -------------------------------
    # Plot total recommendations
    # ------------------------------
    
    plot_recs_vs_degree(
        metrics,
        networks=['ER','BA'],
        save_path='figs/recommendations_vs_degree/',
        scale='log',
        metrics_to_plot=['total_recommendations', 'unique_reach', 'total_recommendations_normalized', 'delta_recs']
    )

    print("Ending Analysis Pipeline...")

    sys.stdout = old_stdout
    log_file.close()
    
