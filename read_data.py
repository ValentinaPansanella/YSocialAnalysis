import os
import zipfile
import sqlite3
from pathlib import Path
import sys
import tempfile
import shutil
from collections import defaultdict

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
PATH = "/home/pansanella/ysim/"
OUTPUT_DIR = Path("/home/pansanella/mydata/YSocial/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "dataset_info.log"
MIN_FREE_SPACE_GB = 5
TABLES_TO_KEEP = ["recommendations", "post", "reaction", "user_mgmt"]

# ---------------------------------------------------
# UTILS
# ---------------------------------------------------
def check_disk_space(path):
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

def parse_zip_filename(zip_name):
    """Returns (network, recsys, run)"""
    name = os.path.basename(zip_name).replace(".zip", "")
    parts = name.split("_")
    
    if len(parts) == 2:
        return parts[0], parts[1], None
    elif len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        return "UNKNOWN", "UNKNOWN", None

def keep_only_tables(db_path, tables_to_keep):
    """Remove all tables not in tables_to_keep"""
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
# PROCESS SINGLE ZIP
# ---------------------------------------------------
def process_zip(zip_path, log_file):
    db_sizes = defaultdict(int)
    run_counts = defaultdict(int)
    corrupted_files = []
    total_db_bytes = 0

    log_file.write(f"\n========================================\n")
    log_file.write(f"TOP ZIP: {zip_path}\n")
    log_file.write("========================================\n")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():

                # Skip non-db files
                if not member.endswith("database_server.db") and not member.endswith(".zip"):
                    continue

                # Nested ZIPs (optional)
                if member.endswith(".zip"):
                    free_space = check_disk_space("/tmp")
                    if free_space < MIN_FREE_SPACE_GB:
                        log_file.write(f"⚠ Skipping nested zip {member} (low disk space)\n")
                        continue

                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            nested_path = zf.extract(member, path=tmpdir)
                            u, c, db, corrupted = process_zip(nested_path, log_file)
                            total_db_bytes += db
                            corrupted_files.extend(corrupted)
                    except zipfile.BadZipFile:
                        log_file.write(f"❌ CORRUPTED NESTED ZIP: {member}\n")
                        corrupted_files.append((zip_path, member))
                    continue

                # database_server.db file
                try:
                    zf.extract(member, OUTPUT_DIR)
                    extracted_path = OUTPUT_DIR / Path(member).name

                    network, recsys, run = parse_zip_filename(zip_path)

                    # Build new filename
                    if run is None:
                        new_name = f"{network}_{recsys}.db"
                    else:
                        new_name = f"{network}_{recsys}_{run}.db"

                    new_path = OUTPUT_DIR / new_name
                    os.rename(extracted_path, new_path)

                    # Filter tables
                    keep_only_tables(new_path, TABLES_TO_KEEP)

                    size_bytes = new_path.stat().st_size
                    db_sizes[(network, recsys)] += size_bytes
                    run_counts[(network, recsys)] += 1
                    total_db_bytes += size_bytes

                    log_file.write(f"✅ Extracted DB: {new_name} ({size_bytes / (1024**2):.2f} MB)\n")

                except zipfile.BadZipFile:
                    log_file.write(f"⚠ Skipped corrupted DB: {member}\n")
                    corrupted_files.append((zip_path, member))

    except zipfile.BadZipFile:
        log_file.write(f"❌ CORRUPTED ZIP: {zip_path}\n")

    return db_sizes, run_counts, total_db_bytes, corrupted_files

# ---------------------------------------------------
# TABLE INSPECTION
# ---------------------------------------------------
def inspect_db(db_path):
    """Print info about each table in a DB."""
    print(f"\n=== DB: {db_path.name} ===")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # List all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    if not tables:
        print("  ⚠ No tables found in this DB.")
        conn.close()
        return

    for table in tables:
        # Get number of rows
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        num_rows = cur.fetchone()[0]

        # Get column names
        cur.execute(f"PRAGMA table_info({table});")
        columns = [row[1] for row in cur.fetchall()]
        num_cols = len(columns)

        print(f"  Table: {table}")
        print(f"    Columns ({num_cols}): {columns}")
        print(f"    Rows: {num_rows}")
        
        if table == "user_mgmt":
            cur.execute(f"SELECT COUNT(DISTINCT id) FROM {table};")
            num_users = cur.fetchone()[0]
            print(f"    Unique users: {num_users}")

    conn.close()

# ---------------------------------------------------
# MAIN SEQUENTIAL PIPELINE
# ---------------------------------------------------
if __name__ == "__main__":
    zip_files = [Path(PATH) / f for f in os.listdir(PATH) if f.startswith("experiments_bulk_") and f.endswith(".zip")]
    print(f"Found {len(zip_files)} ZIP files.")

    global_db_sizes = defaultdict(int)
    global_run_counts = defaultdict(int)
    total_db_bytes_global = 0
    corrupted_files_global = []

    with open(LOG_FILE, "w") as log_file:
        sys.stdout = log_file

        log_file.write("Starting Full Extraction and Logging Pipeline...\n\n")

        for zip_path in zip_files:
            db_sizes, run_counts, total_db_bytes, corrupted_files = process_zip(zip_path, log_file)
            total_db_bytes_global += total_db_bytes
            corrupted_files_global.extend(corrupted_files)

            # Merge aggregates
            for k, v in db_sizes.items():
                global_db_sizes[k] += v
            for k, v in run_counts.items():
                global_run_counts[k] += v

        # SUMMARY
        log_file.write("\n========================================\n")
        log_file.write("DB SIZE + OCCURRENCES BY (NETWORK, RECSYS)\n")
        log_file.write("========================================\n")
        for key in sorted(global_db_sizes.keys()):
            total_bytes = global_db_sizes[key]
            count = global_run_counts.get(key, 0)
            total_gb = total_bytes / (1024**3)
            avg_mb = (total_bytes / count) / (1024**2) if count > 0 else 0
            network, recsys = key
            log_file.write(f"{network} | {recsys}\n")
            log_file.write(f"  Runs found: {count}\n")
            log_file.write(f"  Total DB:   {total_gb:.2f} GB\n")
            log_file.write(f"  Avg per run:{avg_mb:.2f} MB\n\n")

        log_file.write("\n========================================\n")
        log_file.write(f"TOTAL .db SIZE (ALL): {total_db_bytes_global / (1024**3):.2f} GB\n")
        log_file.write("========================================\n")

        log_file.write("\n========================================\n")
        log_file.write("CORRUPTED FILES DETECTED\n")
        log_file.write("========================================\n")
        if corrupted_files_global:
            for bulk, nested in corrupted_files_global:
                log_file.write(f"Bulk: {bulk}\n")
                log_file.write(f"  Corrupted nested zip: {nested}\n")
        else:
            log_file.write("None detected.\n")
        
        log_file.write("\n========================================\n")
        log_file.write("TABLES INSPECTION\n")
        log_file.write("========================================\n")
        db_files = list(OUTPUT_DIR.glob("*.db"))
        if not db_files:
            log_file.write(f"No DB files found in {OUTPUT_DIR}\n")
        else:
            log_file.write(f"Found {len(db_files)} DB files in {OUTPUT_DIR}\n")
        
        import shutil   
        for db_path in db_files:
            parts = db_path.stem.split("_")
            # Determine new name
            if len(parts) == 2:
                new_path = OUTPUT_DIR / f"{parts[0]}_{parts[1]}_0.db"
                # Move/rename safely, overwriting if new_path already exists
                if db_path != new_path:
                    if new_path.exists():
                        # Optional: remove the old copy to avoid duplicates
                        new_path.unlink()
                    shutil.move(db_path, new_path)
                db_to_inspect = new_path
            else:
                # Already has a run number or unknown, keep as-is
                db_to_inspect = db_path

            # Inspect DB using the correct path
            inspect_db(db_to_inspect)

        log_file.write("\nPipeline Completed.\n")
        sys.stdout = sys.__stdout__

    print("Done. All output logged in", LOG_FILE)