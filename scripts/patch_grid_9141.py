import duckdb
import pandas as pd
import requests

DB_PATH = "data/f1.duckdb"
SESSION_KEY = 9141

def fetch_starting_grid(session_key: int) -> pd.DataFrame:
    url = "https://api.openf1.org/v1/position"
    params = {"session_key": session_key}

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError(f"No position data returned for session_key={session_key}")

    df = pd.DataFrame(data)

    # sort to earliest record per driver
    sort_cols = ["driver_number"]
    if "lap_number" in df.columns:
        sort_cols.append("lap_number")
    if "date" in df.columns:
        sort_cols.append("date")

    df = df.sort_values(sort_cols, ascending=True)
    first = df.groupby("driver_number", as_index=False).first()

    out = first[["driver_number", "position"]].copy()
    out = out.rename(columns={"position": "grid_position"})
    out["session_key"] = session_key

    out["grid_position"] = pd.to_numeric(out["grid_position"], errors="coerce")
    out = out.dropna(subset=["grid_position"])

    return out

def main():
    grid = fetch_starting_grid(SESSION_KEY)

    con = duckdb.connect(DB_PATH)

    before = con.execute("""
        SELECT COUNT(*) FROM race_results_enriched_2023
        WHERE session_key=? AND grid_position IS NULL
    """, [SESSION_KEY]).fetchone()[0]
    print("Before null grid rows:", before)

    con.register("grid_df", grid)

    con.execute("""
        UPDATE race_results_enriched_2023 AS r
        SET grid_position = g.grid_position
        FROM grid_df AS g
        WHERE r.session_key = g.session_key
          AND r.driver_number = g.driver_number
          AND r.session_key = ?
          AND r.grid_position IS NULL
    """, [SESSION_KEY])

    con.unregister("grid_df")

    after = con.execute("""
        SELECT COUNT(*) FROM race_results_enriched_2023
        WHERE session_key=? AND grid_position IS NULL
    """, [SESSION_KEY]).fetchone()[0]
    print("After null grid rows:", after)

    sample = con.execute("""
        SELECT driver_number, full_name, team_name, grid_position, finish_position, dnf_flag
        FROM race_results_enriched_2023
        WHERE session_key=?
        ORDER BY grid_position
    """, [SESSION_KEY]).fetchall()

    print("Sample (ordered by grid):")
    for row in sample[:25]:
        print(row)

    con.close()

if __name__ == "__main__":
    main()
