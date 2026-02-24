import os
import glob 
import pandas as pd
import numpy as np
import sqlite3

BASE_DIR = "experiments_recsys"
NETWORKS = ['BA', 'ER']
RECSYSS = ['F', 'RC']
RUNS = list(range(10))

def get_db_path(base_dir, network, recsys, run):
    return os.path.join(base_dir, f"{network}_{recsys}_{run}", "database_server.db")

def get_cache_path(base_dir, network, recsys, run):
    cache_dir = os.path.join(base_dir, 'cache_data')
    os.makedir(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{network}_{recsys}_{run}.pkl")

def compute_author_degrees(base_dir, network, recsys, run, by='id'):
    """
    Compute node degrees from network CSV and map to either user IDs or usernames.

    Parameters
    ----------
    base_dir : str
        Base folder containing experiment data
    network : str
        Network type (e.g., 'ER', 'BA')
    recsys : str
        Recommendation system name (e.g., 'RC')
    run : int
        Run number
    by : str
        'id' to return dict keyed by user ID, 'username' to return dict keyed by username

    Returns
    -------
    dict
        Mapping of user_id or username -> degree
    """

    db_path = get_db_path(base_dir, network, recsys, run)
    run_folder = os.path.dirname(db_path)
    csv_files = glob.glob(os.path.join(run_folder, "*.csv"))

    if not csv_files:
        print(f"No edge CSV found for {network}-{recsys}-{run}")
        return {}

    # --- read edges and compute degrees ---
    try:
        df_edges = pd.read_csv(csv_files[0], header=None, names=['source', 'target'])
    except Exception as e:
        print(f"Error reading edge CSV: {e}")
        return {}

    deg_counts = pd.concat([df_edges['source'], df_edges['target']]).value_counts()
    degrees = {k: int(v / 2) for k, v in deg_counts.to_dict().items()}

    # --- map to user_id or username ---
    if by in ['id', 'username']:
        try:
            with sqlite3.connect(db_path) as conn:
                df_users = pd.read_sql("SELECT username, id FROM user_mgmt", conn)
                username_to_id = dict(zip(df_users["username"], df_users["id"]))
                id_to_username = dict(zip(df_users["id"], df_users["username"]))

                if by == 'id':
                    # map degrees to user_id
                    degrees_by_id = {
                        username_to_id[user]: deg
                        for user, deg in degrees.items()
                        if user in username_to_id
                    }
                    return degrees_by_id
                else:
                    # keep degrees keyed by username
                    return {user: deg for user, deg in degrees.items()}

        except Exception as e:
            print(f"Error reading user table from DB: {e}")
            return {}
    else:
        print("Parameter `by` must be 'id' or 'username'")
        return {}

def pids_to_list(pids_str):
    try:
        pids = [int(x) for x in pids_str.split("|")]
        return pids 
    except ValueError:
        pass 

def compute_delta_recs(df_rec: pd.DataFrame, all_users: list) -> pd.Series:
    """
    Compute delta_recs for each user:
        delta_recs = unique_reach - num_authors_viewed
        
    Args:
        df_rec: DataFrame with columns 'author_id', 'viewer_id'
        all_users: A complete list or array of ALL user IDs in the network.
                   (This ensures users with 0 reach are counted!)
                   
    Returns a pandas Series indexed by user_id for ALL users.
    """
    
    # 1. Unique reach: number of distinct viewers per author
    reach = df_rec.groupby('author_id')['viewer_id'].nunique()
    
    # 2. Number of authors viewed by each viewer (excluding self-views)
    mask_not_self = df_rec['viewer_id'] != df_rec['author_id']
    viewed = df_rec[mask_not_self].groupby('viewer_id')['author_id'].nunique()
    
    # 3. CRITICAL: Reindex both against the COMPLETE list of all users
    # This forces users with 0 reach or 0 views to actually have a 0.0 value
    reach_aligned = reach.reindex(all_users, fill_value=0)
    viewed_aligned = viewed.reindex(all_users, fill_value=0)
    
    # 4. Compute delta
    delta = reach_aligned - viewed_aligned
    
    # Optional: rename the index for clarity
    delta.index.name = 'user_id'
    
    return delta

def gini_coefficient(values):
    """Compute Gini coefficient of array of values"""
    if len(values) == 0: return 0.0
    array = np.array(values, dtype=float)
    if np.amin(array) < 0:
        array -= np.amin(array) # Values cannot be negative
    array += 0.0000001 # Values cannot be 0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))




    