import csv
import glob
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RUN_PATTERN = re.compile(r"^(?P<network>[^_]+)_(?P<recsys>[^_]+)_(?P<run>\d+)$")


def _normalize_base_dir(base_dir: str) -> str:
    return os.path.normpath(str(base_dir))


def _is_old_layout(base_dir: str) -> bool:
    base = _normalize_base_dir(base_dir)
    return os.path.basename(base) == "old_data"


def get_db_path(base_dir: str, network: str, recsys: str, run: int) -> str:
    base = _normalize_base_dir(base_dir)
    if _is_old_layout(base):
        return os.path.join(base, f"{network}_{recsys}_{run}", "database_server.db")
    return os.path.join(base, f"{network}_{recsys}_{run}.db")


def get_csv_path(base_dir: str, network: str, recsys: str, run: int) -> Optional[str]:
    base = _normalize_base_dir(base_dir)
    if _is_old_layout(base):
        run_dir = os.path.join(base, f"{network}_{recsys}_{run}")
        csv_files = glob.glob(os.path.join(run_dir, "*.csv"))
        return csv_files[0] if csv_files else None

    csv_path = os.path.join(base, f"{network}_{recsys}_{run}.csv")
    return csv_path if os.path.exists(csv_path) else None


def get_cache_path(base_dir: str, network: str, recsys: str, run: int) -> str:
    cache_dir = os.path.join(base_dir, "cache_data")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{network}_{recsys}_{run}.pkl")


def _parse_run_token(token: str) -> Optional[Tuple[str, str, int]]:
    match = RUN_PATTERN.match(token)
    if not match:
        return None
    return (
        match.group("network"),
        match.group("recsys"),
        int(match.group("run")),
    )


def discover_runs(
    base_dir: str,
    networks: Optional[Sequence[str]] = None,
    recsyss: Optional[Sequence[str]] = None,
    runs: Optional[Sequence[int]] = None,
) -> List[Tuple[str, str, int]]:
    base = _normalize_base_dir(base_dir)

    if not os.path.isdir(base):
        return []

    discovered: List[Tuple[str, str, int]] = []

    if _is_old_layout(base):
        for entry in os.listdir(base):
            parsed = _parse_run_token(entry)
            if parsed is None:
                continue
            network, recsys, run = parsed
            db_path = os.path.join(base, entry, "database_server.db")
            if os.path.exists(db_path):
                discovered.append((network, recsys, run))
    else:
        for db_file in Path(base).glob("*.db"):
            parsed = _parse_run_token(db_file.stem)
            if parsed is None:
                continue
            discovered.append(parsed)

    if networks is not None:
        network_set = set(networks)
        discovered = [spec for spec in discovered if spec[0] in network_set]

    if recsyss is not None:
        recsys_set = set(recsyss)
        discovered = [spec for spec in discovered if spec[1] in recsys_set]

    if runs is not None:
        run_set = {int(r) for r in runs}
        discovered = [spec for spec in discovered if spec[2] in run_set]

    discovered = sorted(set(discovered), key=lambda x: (x[0], x[1], x[2]))
    return discovered


def compute_author_degrees(base_dir: str, network: str, recsys: str, run: int, by: str = "id") -> Dict:
    db_path = get_db_path(base_dir, network, recsys, run)
    csv_path = get_csv_path(base_dir, network, recsys, run)

    if csv_path is None or not os.path.exists(csv_path):
        return {}

    try:
        deg_counts = Counter()
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                source = row[0].strip()
                target = row[1].strip()
                if source:
                    deg_counts[source] += 1
                if target:
                    deg_counts[target] += 1
    except Exception:
        return {}

    degrees_by_username = {str(node): int(count) for node, count in deg_counts.items()}

    if by == "username":
        return degrees_by_username

    if by != "id":
        raise ValueError("Parameter `by` must be 'id' or 'username'")

    if not os.path.exists(db_path):
        return {}

    try:
        with sqlite3.connect(db_path) as conn:
            df_users = pd.read_sql("SELECT id, username FROM user_mgmt", conn)
    except Exception:
        return {}

    username_to_id = {
        str(username): user_id
        for user_id, username in zip(df_users["id"], df_users["username"])
    }

    return {
        username_to_id[user]: deg
        for user, deg in degrees_by_username.items()
        if user in username_to_id
    }


def pids_to_list(pids_raw):
    if pids_raw is None or (isinstance(pids_raw, float) and np.isnan(pids_raw)):
        return []

    if isinstance(pids_raw, (list, tuple, np.ndarray)):
        return list(pids_raw)

    values = []
    for token in str(pids_raw).split("|"):
        token = token.strip()
        if not token:
            continue

        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            values.append(int(token))
        else:
            values.append(token)

    return values


def post_id_mapping(base_dir: str, network: str, recsys: str, run: int) -> Dict:
    db_path = get_db_path(base_dir, network, recsys, run)
    if not os.path.exists(db_path):
        return {}

    with sqlite3.connect(db_path) as conn:
        df_posts = pd.read_sql("SELECT id AS post_id FROM post", conn)

    post_ids_unique = sorted(df_posts["post_id"].dropna().unique())
    return {pid: idx for idx, pid in enumerate(post_ids_unique)}


def compute_delta_recs(df_rec: pd.DataFrame, all_users: Sequence) -> pd.Series:
    reach = df_rec.groupby("author_id")["viewer_id"].nunique()

    mask_not_self = df_rec["viewer_id"] != df_rec["author_id"]
    viewed = df_rec[mask_not_self].groupby("viewer_id")["author_id"].nunique()

    reach_aligned = reach.reindex(all_users, fill_value=0)
    viewed_aligned = viewed.reindex(all_users, fill_value=0)

    delta = reach_aligned - viewed_aligned
    delta.index.name = "user_id"
    return delta


def gini_coefficient(values) -> float:
    if len(values) == 0:
        return 0.0

    array = np.array(values, dtype=float)
    if np.amin(array) < 0:
        array -= np.amin(array)

    array += 1e-7
    array = np.sort(array)

    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float(np.sum((2 * index - n - 1) * array) / (n * np.sum(array)))
