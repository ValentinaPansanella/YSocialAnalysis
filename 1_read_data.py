import os
import json
import zipfile
import sqlite3
from pathlib import Path
import sys
import tempfile
import shutil
from collections import defaultdict
import tqdm
from datetime import datetime

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

PATH = Path("/home/pansanella/ysim/")
OUTPUT_DIR = Path("/home/pansanella/mydata/YSocial/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "logs/dataset_info.log"

TABLES_TO_KEEP = [
    "recommendations",
    "post",
    "reaction",
    "user_mgmt"
]

MIN_FREE_SPACE_GB = 5

# ---------------------------------------------------
# UTILS
# ---------------------------------------------------

def check_disk_space(path):
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)


def parse_zip_filename(zip_name):
    """
    Extract (network, recsys, run) from zip filename
    """

    name = Path(zip_name).stem
    parts = name.split("_")

    if len(parts) >= 3:
        network, recsys, run = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        network, recsys, run = parts[0], parts[1], "0"
    else:
        network, recsys, run = name, "UNKNOWN", "0"

    return network, recsys, run


def keep_only_tables(db_path, tables_to_keep):

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0] for row in cur.fetchall()]

    for t in all_tables:
        if t not in tables_to_keep:
            cur.execute(f"DROP TABLE IF EXISTS {t}")

    conn.commit()
    conn.close()


# ---------------------------------------------------
# USER CONSISTENCY CHECK
# ---------------------------------------------------

def check_users_consistency(db_path, json_path, log_file):

    try:
        with open(json_path) as f:
            data = json.load(f)

        agents = data["agents"]

        json_users = set(str(a["id"]) for a in agents)

    except Exception as e:
        log_file.write(f"⚠ JSON error {json_path.name}: {e}\n")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id FROM user_mgmt;")
    db_users = set(str(r[0]) for r in cur.fetchall())

    conn.close()

    missing_json = db_users - json_users
    missing_db = json_users - db_users

    print(f"Users in DB not in JSON: {len(missing_json)}")
    print(f"Users in JSON not in DB: {len(missing_db)}")

    if missing_json:
        log_file.write(f"⚠ Users in DB not in JSON ({json_path.name}): {len(missing_json)}\n")

    if missing_db:
        log_file.write(f"⚠ Users in JSON not in DB ({db_path.name}): {len(missing_db)}\n")


# ---------------------------------------------------
# ZIP PROCESSING
# ---------------------------------------------------

def process_zip(zip_path, log_file):

    zip_path = Path(zip_path)

    db_sizes = defaultdict(int)
    run_counts = defaultdict(int)
    corrupted_files = []
    total_db_bytes = 0

    log_file.write(f"\n=== Processing ZIP: {zip_path.name} ===\n")

    try:

        with zipfile.ZipFile(zip_path, "r") as zf:

            members = [m.filename for m in zf.infolist()]

            db_member = None
            csv_member = None
            json_member = None
            nested_zips = []

            for m in members:

                if m.endswith("database_server.db"):
                    db_member = m

                elif m.endswith(".csv"):
                    csv_member = m

                elif m == "1k_agents.json":
                    json_member = m

                elif m.endswith(".zip"):
                    nested_zips.append(m)

            # ---------------------------------------------------
            # HANDLE NESTED ZIPs
            # ---------------------------------------------------

            for member in nested_zips:

                if check_disk_space("/tmp") < MIN_FREE_SPACE_GB:
                    log_file.write(f"⚠ Skipping nested zip {member} (low disk space)\n")
                    continue

                try:

                    with tempfile.TemporaryDirectory() as tmpdir:

                        nested_path = zf.extract(member, tmpdir)

                        nested_db_sizes, nested_run_counts, nested_bytes, nested_corrupted = process_zip(
                            nested_path, log_file
                        )

                        for k, v in nested_db_sizes.items():
                            db_sizes[k] += v

                        for k, v in nested_run_counts.items():
                            run_counts[k] += v

                        total_db_bytes += nested_bytes
                        corrupted_files.extend(nested_corrupted)

                except zipfile.BadZipFile:

                    log_file.write(f"❌ CORRUPTED NESTED ZIP: {member}\n")
                    corrupted_files.append((zip_path, member))

            # ---------------------------------------------------
            # EXTRACT MAIN FILES
            # ---------------------------------------------------

            network, recsys, run = parse_zip_filename(zip_path)
            base = f"{network}_{recsys}_{run}"

            # DB
            if db_member:

                extracted = zf.extract(db_member, OUTPUT_DIR)
                new_db_path = OUTPUT_DIR / f"{base}.db"

                if new_db_path.exists():
                    new_db_path.unlink()

                os.rename(extracted, new_db_path)

                keep_only_tables(new_db_path, TABLES_TO_KEEP)

                size_bytes = new_db_path.stat().st_size

                db_sizes[(network, recsys)] += size_bytes
                run_counts[(network, recsys)] += 1
                total_db_bytes += size_bytes

                log_file.write(
                    f"✅ Extracted DB: {new_db_path.name} ({size_bytes/(1024**2):.2f} MB)\n"
                )

            # CSV
            if csv_member:

                extracted = zf.extract(csv_member, OUTPUT_DIR)
                new_csv_path = OUTPUT_DIR / f"{base}.csv"

                if new_csv_path.exists():
                    new_csv_path.unlink()

                os.rename(extracted, new_csv_path)

                log_file.write(f"📄 Extracted CSV: {new_csv_path.name}\n")

            # JSON
            if json_member:

                extracted = zf.extract(json_member, OUTPUT_DIR)
                new_json_path = OUTPUT_DIR / f"{base}.json"

                if new_json_path.exists():
                    new_json_path.unlink()

                os.rename(extracted, new_json_path)

                log_file.write(f"📄 Extracted JSON: {new_json_path.name}\n")

    except zipfile.BadZipFile:

        log_file.write(f"❌ CORRUPTED ZIP: {zip_path}\n")
        corrupted_files.append((zip_path, None))

    return db_sizes, run_counts, total_db_bytes, corrupted_files


# ---------------------------------------------------
# DB INSPECTION
# ---------------------------------------------------

def inspect_db(db_path, log_file):

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
        rows = cur.fetchone()[0]

        cur.execute(f"PRAGMA table_info({table})")
        columns = [c[1] for c in cur.fetchall()]

        log_file.write(
            f"Table: {table} | Columns ({len(columns)}): {columns} | Rows: {rows}\n"
        )

        if table == "user_mgmt":

            cur.execute("SELECT COUNT(DISTINCT id) FROM user_mgmt")
            users = cur.fetchone()[0]

            log_file.write(f"Unique users: {users}\n")

    json_path = OUTPUT_DIR / f"{db_path.stem}.json"

    if json_path.exists():
        check_users_consistency(db_path, json_path, log_file)
    else:
        log_file.write(f"⚠ Missing JSON for {db_path.name}\n")

    conn.close()


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    zip_files = sorted(PATH.glob("*.zip"))

    print(f"Found {len(zip_files)} ZIP files to process.")

    global_db_sizes = defaultdict(int)
    global_run_counts = defaultdict(int)
    total_db_bytes_global = 0
    corrupted_files_global = []

    with open(LOG_FILE, "a") as log_file:

        original_stdout = sys.stdout
        sys.stdout = log_file

        try:

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"Pipeline start: {now}\n")

            for zip_path in zip_files:

                db_sizes, run_counts, total_db_bytes, corrupted = process_zip(zip_path, log_file)

                total_db_bytes_global += total_db_bytes
                corrupted_files_global.extend(corrupted)

                for k, v in db_sizes.items():
                    global_db_sizes[k] += v

                for k, v in run_counts.items():
                    global_run_counts[k] += v

            log_file.write("\n=== DB SIZE SUMMARY ===\n")

            for key in sorted(global_db_sizes):

                network, recsys = key

                total_bytes = global_db_sizes[key]
                runs = global_run_counts[key]

                total_gb = total_bytes / (1024**3)
                avg_mb = (total_bytes / runs) / (1024**2)

                log_file.write(
                    f"{network} | {recsys} | Runs: {runs} | Total: {total_gb:.2f} GB | Avg: {avg_mb:.2f} MB\n"
                )

            log_file.write(f"\nTOTAL DB SIZE: {total_db_bytes_global/(1024**3):.2f} GB\n")

            log_file.write("\n=== CORRUPTED FILES ===\n")

            if corrupted_files_global:
                for c in corrupted_files_global:
                    log_file.write(f"{c}\n")
            else:
                log_file.write("None\n")

            log_file.write("\n=== DB INSPECTION ===\n")

            db_files = list(OUTPUT_DIR.glob("*.db"))

            for db in tqdm.tqdm(db_files, desc="Inspecting DBs", file=original_stdout):
                inspect_db(db, log_file)

        finally:
            sys.stdout = original_stdout

    print(f"Done. Log written to {LOG_FILE}")


if __name__ == "__main__":
    main()