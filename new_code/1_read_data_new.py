import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tqdm


DEFAULT_TABLES_TO_KEEP = ["recommendations", "post", "reaction", "user_mgmt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and normalize simulation outputs from nested ZIP archives."
    )
    parser.add_argument("--input-dir", default=".", help="Directory containing top-level ZIP archives")
    parser.add_argument("--input-zips", nargs="*", default=None, help="Specific ZIP files to process")
    parser.add_argument("--output-dir", default="data", help="Output directory for extracted .db/.csv/.json")
    parser.add_argument("--log-file", default="dataset_info.log")
    parser.add_argument("--min-free-space-gb", type=float, default=5.0)
    parser.add_argument("--tables-to-keep", nargs="+", default=DEFAULT_TABLES_TO_KEEP)
    parser.add_argument("--skip-inspect", action="store_true")
    return parser.parse_args()


def check_disk_space(path: str) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def parse_zip_filename(zip_name: str) -> Optional[Tuple[str, str, int]]:
    stem = Path(zip_name).stem

    # Expected: NETWORK_RECSYS or NETWORK_RECSYS_RUN.
    match = re.match(r"^(?P<network>[^_]+)_(?P<recsys>[^_]+?)(?:_(?P<run>\d+))?$", stem)
    if not match:
        return None

    network = match.group("network")
    recsys = match.group("recsys")
    run = int(match.group("run") or 0)
    return network, recsys, run


def keep_only_tables(db_path: Path, tables_to_keep: Sequence[str]) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0] for row in cur.fetchall()]

    for table in all_tables:
        if table not in tables_to_keep:
            cur.execute(f"DROP TABLE IF EXISTS {table}")

    conn.commit()
    conn.close()


def _is_valid_agents_payload(data) -> bool:
    if not isinstance(data, dict):
        return False
    agents = data.get("agents")
    if not isinstance(agents, list) or len(agents) == 0:
        return False
    first = agents[0]
    if not isinstance(first, dict):
        return False
    return "id" in first


def _load_json_if_valid(raw: bytes) -> Optional[dict]:
    try:
        payload = json.loads(raw)
    except Exception:
        return None

    if _is_valid_agents_payload(payload):
        return payload
    return None


def _replace_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))


def _sample(values: Sequence[str], n: int = 5) -> List[str]:
    values = list(values)
    return values[:n]


def check_users_consistency(db_path: Path, json_path: Path) -> Tuple[int, int, List[str], List[str]]:
    with open(json_path, "r") as f:
        payload = json.load(f)

    agents = payload.get("agents", [])
    json_users = {str(agent.get("id")) for agent in agents if isinstance(agent, dict) and "id" in agent}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM user_mgmt")
    db_users = {str(row[0]) for row in cur.fetchall()}
    conn.close()

    missing_in_json = sorted(db_users - json_users)
    missing_in_db = sorted(json_users - db_users)

    return len(missing_in_json), len(missing_in_db), _sample(missing_in_json), _sample(missing_in_db)


def process_zip(
    zip_path: Path,
    output_dir: Path,
    log_file,
    tables_to_keep: Sequence[str],
    min_free_space_gb: float,
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], int], int, List[Tuple[str, Optional[str]]]]:
    db_sizes = defaultdict(int)
    run_counts = defaultdict(int)
    corrupted_files: List[Tuple[str, Optional[str]]] = []
    total_db_bytes = 0

    run_info = parse_zip_filename(zip_path.name)
    log_file.write(f"\n=== Processing ZIP: {zip_path.name} ===\n")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_done = False
            json_candidates: List[Tuple[int, bytes, str]] = []
            db_written = False

            for member in zf.infolist():
                member_name = member.filename

                if not (
                    member_name.endswith("database_server.db")
                    or member_name.endswith(".csv")
                    or member_name.endswith(".json")
                    or member_name.endswith(".zip")
                ):
                    continue

                if member_name.endswith(".zip"):
                    free_space = check_disk_space("/tmp")
                    if free_space < min_free_space_gb:
                        log_file.write(f"⚠ Skipping nested zip {member_name} (low disk space)\n")
                        continue

                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            nested_path = Path(zf.extract(member_name, path=tmpdir))
                            nested_stats = process_zip(
                                nested_path,
                                output_dir,
                                log_file,
                                tables_to_keep,
                                min_free_space_gb,
                            )
                            nested_db_sizes, nested_run_counts, nested_total, nested_corrupted = nested_stats
                            for key, value in nested_db_sizes.items():
                                db_sizes[key] += value
                            for key, value in nested_run_counts.items():
                                run_counts[key] += value
                            total_db_bytes += nested_total
                            corrupted_files.extend(nested_corrupted)
                    except zipfile.BadZipFile:
                        log_file.write(f"❌ CORRUPTED NESTED ZIP: {member_name}\n")
                        corrupted_files.append((str(zip_path), member_name))

                    continue

                if run_info is None:
                    continue

                network, recsys, run = run_info
                stem = f"{network}_{recsys}_{run}"

                if member_name.endswith("database_server.db"):
                    extracted_path = Path(zf.extract(member_name, path=output_dir))
                    final_db_path = output_dir / f"{stem}.db"
                    _replace_file(extracted_path, final_db_path)

                    keep_only_tables(final_db_path, tables_to_keep)

                    size_bytes = final_db_path.stat().st_size
                    db_sizes[(network, recsys)] += size_bytes
                    run_counts[(network, recsys)] += 1
                    total_db_bytes += size_bytes
                    db_written = True

                    log_file.write(f"✅ Extracted DB: {final_db_path.name} ({size_bytes / (1024 ** 2):.2f} MB)\n")
                    continue

                if member_name.endswith(".csv") and not csv_done:
                    extracted_csv = Path(zf.extract(member_name, path=output_dir))
                    final_csv = output_dir / f"{stem}.csv"
                    _replace_file(extracted_csv, final_csv)
                    csv_done = True
                    log_file.write(f"📄 Extracted CSV: {final_csv.name}\n")
                    continue

                if member_name.endswith(".json"):
                    raw = zf.read(member_name)
                    payload = _load_json_if_valid(raw)
                    if payload is None:
                        continue

                    # Prefer the canonical simulator file if present.
                    score = 2 if Path(member_name).name == "1k_agents.json" else 1
                    json_candidates.append((score, raw, member_name))

            if run_info is not None and db_written and not csv_done:
                network, recsys, run = run_info
                log_file.write(f"⚠ No CSV found for {network}_{recsys}_{run}.db\n")

            if run_info is not None and db_written:
                network, recsys, run = run_info
                stem = f"{network}_{recsys}_{run}"

                if json_candidates:
                    json_candidates.sort(key=lambda x: x[0], reverse=True)
                    _, best_raw, best_name = json_candidates[0]
                    json_path = output_dir / f"{stem}.json"
                    with open(json_path, "wb") as jf:
                        jf.write(best_raw)
                    log_file.write(f"📄 Extracted JSON: {json_path.name} (from {best_name})\n")
                else:
                    log_file.write(f"⚠ No valid agents JSON found for {stem}.db\n")

    except zipfile.BadZipFile:
        log_file.write(f"❌ CORRUPTED ZIP: {zip_path}\n")
        corrupted_files.append((str(zip_path), None))

    return db_sizes, run_counts, total_db_bytes, corrupted_files


def inspect_db(db_path: Path, output_dir: Path, log_file) -> None:
    log_file.write(f"\n=== DB: {db_path.name} ===\n")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    if not tables:
        log_file.write("⚠ No tables found\n")
        conn.close()
        return

    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        num_rows = cur.fetchone()[0]

        cur.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cur.fetchall()]

        log_file.write(f"Table: {table} | Columns ({len(columns)}): {columns} | Rows: {num_rows}\n")

        if table == "user_mgmt":
            cur.execute("SELECT COUNT(DISTINCT id) FROM user_mgmt")
            num_users = cur.fetchone()[0]
            log_file.write(f"Unique users: {num_users}\n")

    conn.close()

    json_path = output_dir / f"{db_path.stem}.json"
    if not json_path.exists():
        log_file.write(f"⚠ No corresponding JSON for {db_path.name}\n")
        return

    try:
        missing_json_count, missing_db_count, sample_json, sample_db = check_users_consistency(db_path, json_path)
        log_file.write(f"Users in DB not in JSON: {missing_json_count}; sample={sample_json}\n")
        log_file.write(f"Users in JSON not in DB: {missing_db_count}; sample={sample_db}\n")
    except Exception as exc:
        log_file.write(f"⚠ Consistency check failed for {db_path.name}: {exc}\n")


def _resolve_input_zips(input_dir: str, input_zips: Optional[Sequence[str]]) -> List[Path]:
    if input_zips:
        return sorted(Path(p) for p in input_zips)

    return sorted(Path(input_dir).glob("*.zip"))


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_files = _resolve_input_zips(args.input_dir, args.input_zips)
    print(f"Found {len(zip_files)} ZIP files to process.")

    global_db_sizes = defaultdict(int)
    global_run_counts = defaultdict(int)
    total_db_bytes_global = 0
    corrupted_files_global: List[Tuple[str, Optional[str]]] = []

    with open(args.log_file, "a") as log_file:
        log_file.write(f"Starting Full Extraction and Logging Pipeline at {datetime.now().isoformat()}\n")

        for zip_path in zip_files:
            stats = process_zip(
                zip_path=zip_path,
                output_dir=output_dir,
                log_file=log_file,
                tables_to_keep=args.tables_to_keep,
                min_free_space_gb=args.min_free_space_gb,
            )
            db_sizes, run_counts, total_db_bytes, corrupted_files = stats

            total_db_bytes_global += total_db_bytes
            corrupted_files_global.extend(corrupted_files)

            for key, value in db_sizes.items():
                global_db_sizes[key] += value
            for key, value in run_counts.items():
                global_run_counts[key] += value

        log_file.write("\n=== DB SIZE + OCCURRENCES BY (NETWORK, RECSYS) ===\n")
        for key in sorted(global_db_sizes.keys()):
            total_bytes = global_db_sizes[key]
            count = global_run_counts[key]
            network, recsys = key
            total_gb = total_bytes / (1024 ** 3)
            avg_mb = (total_bytes / count) / (1024 ** 2) if count > 0 else 0
            log_file.write(
                f"{network} | {recsys} | Runs: {count} | "
                f"Total DB: {total_gb:.2f} GB | Avg per run: {avg_mb:.2f} MB\n"
            )

        log_file.write(f"\nTOTAL .db SIZE (ALL): {total_db_bytes_global / (1024 ** 3):.2f} GB\n")

        log_file.write("\n=== CORRUPTED FILES ===\n")
        if corrupted_files_global:
            for bulk, nested in corrupted_files_global:
                log_file.write(f"Bulk: {bulk} | Nested: {nested}\n")
        else:
            log_file.write("None detected.\n")

        if not args.skip_inspect:
            log_file.write("\n=== TABLE INSPECTION ===\n")
            db_files = sorted(output_dir.glob("*.db"))
            if not db_files:
                log_file.write("No DB files found.\n")
            for db_path in tqdm.tqdm(db_files, desc="Inspecting DBs", file=sys.stdout):
                inspect_db(db_path, output_dir, log_file)

        log_file.write("\nPipeline Completed.\n")

    print(f"Done. All output logged in {args.log_file}")


if __name__ == "__main__":
    main()
