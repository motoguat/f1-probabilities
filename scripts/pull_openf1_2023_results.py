import time
from typing import Any, Dict, List

import duckdb
import pandas as pd
import requests

BASE = "https://api.openf1.org/v1"
DB_PATH = "data/f1.duckdb"


def fetch(endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{BASE}/{endpoint}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> None:
    con = duckdb.connect(DB_PATH)

    # Pull session keys we already stored
    session_keys = [r[0] for r in con.execute(
        "SELECT session_key FROM sessions_2023 ORDER BY date_start;"
    ).fetchall()]

    print(f"Found {len(session_keys)} race sessions in DuckDB.")
    if not session_keys:
        raise SystemExit("No session_keys found. Run the sessions script first.")

    all_rows: List[Dict[str, Any]] = []

    for i, sk in enumerate(session_keys, start=1):
        print(f"[{i}/{len(session_keys)}] Fetching results for session_key={sk} ...")
        rows = fetch("session_result", {"session_key": sk})
        all_rows.extend(rows)
        time.sleep(0.2)  # be polite

    print(f"Total result rows fetched: {len(all_rows)}")
    if not all_rows:
        raise SystemExit("No results returned from OpenF1.")

    df = pd.DataFrame(all_rows)

    # Keep a clean, stable subset (you can expand later)
    keep = [
        "meeting_key",
        "session_key",
        "driver_number",
        "driver_code",
        "full_name",
        "team_name",
        "position",
        "grid_position",
        "status",
        "points",
        "time",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Normalize types where possible
    if "position" in df.columns:
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
    if "grid_position" in df.columns:
        df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce")
    if "points" in df.columns:
        df["points"] = pd.to_numeric(df["points"], errors="coerce")

    # Replace table to keep reruns simple
    con.execute("DROP TABLE IF EXISTS results_2023;")
    con.register("df_results", df)
    con.execute("CREATE TABLE results_2023 AS SELECT * FROM df_results;")
    con.unregister("df_results")

        # Quick validation
    row_count = con.execute("SELECT COUNT(*) FROM results_2023;").fetchone()[0]
    distinct_races = con.execute("SELECT COUNT(DISTINCT session_key) FROM results_2023;").fetchone()[0]

    print("Row count:", row_count)
    print("Distinct races:", distinct_races)

    sample = con.execute(
        """
        SELECT session_key, driver_number, position, points
        FROM results_2023
        WHERE position <= 3
        ORDER BY session_key, position
        LIMIT 9;
        """
    ).fetchall()

    print("Sample podium rows:")
    for row in sample:
        print("  ", row)

    con.close()
    print("Done.")

if __name__ == "__main__":
    main()

