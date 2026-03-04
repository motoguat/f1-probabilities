import json
import time
from typing import Any, Dict, List

import duckdb
import requests

BASE = "https://api.openf1.org/v1"
DB_PATH = "data/f1.duckdb"


def fetch(endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{BASE}/{endpoint}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> None:
    print("Fetching 2023 Race sessions from OpenF1...")
    sessions = fetch("sessions", {"year": 2023, "session_name": "Race"})

    print(f"Fetched {len(sessions)} sessions.")
    if not sessions:
        raise SystemExit("No sessions returned. Check API availability.")

    # Write raw snapshot for debugging / case study artifact
    with open("data/openf1_sessions_2023.json", "w") as f:
        json.dump(sessions, f, indent=2)

    print(f"Saving to DuckDB: {DB_PATH}")
    con = duckdb.connect(DB_PATH)

    # Create table from JSON objects
    con.execute("CREATE TABLE IF NOT EXISTS sessions_2023 AS SELECT * FROM read_json_auto('data/openf1_sessions_2023.json');")

    # Basic checks
    row_count = con.execute("SELECT COUNT(*) FROM sessions_2023;").fetchone()[0]
    sample = con.execute(
        """
        SELECT meeting_key, session_key, country_name, circuit_short_name, date_start
        FROM sessions_2023
        ORDER BY date_start
        LIMIT 5;
        """
    ).fetchall()

    print("Row count:", row_count)
    print("Sample rows:")
    for s in sample:
        print("  ", s)

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
