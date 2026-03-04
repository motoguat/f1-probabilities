import time
import duckdb
import pandas as pd
import requests

BASE = "https://api.openf1.org/v1"
DB_PATH = "data/f1.duckdb"

def fetch(endpoint, params):
    r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    con = duckdb.connect(DB_PATH)

    races = con.execute("""
        SELECT meeting_key, session_key, date_start
        FROM sessions_2023
        ORDER BY date_start;
    """).df()

    all_dfs = []

    for i, row in races.iterrows():
        meeting_key = int(row["meeting_key"])
        race_session_key = int(row["session_key"])

        print(f"[{i+1}/{len(races)}] meeting_key={meeting_key} race_session_key={race_session_key}")

        # 1) Race results
        race_results = fetch("session_result", {"session_key": race_session_key})
        df_race = pd.DataFrame(race_results)
        if df_race.empty:
            print("  - No race results, skipping")
            continue

        # 2) Driver names/teams (use race session)
        drivers = fetch("drivers", {"session_key": race_session_key})
        df_drv = pd.DataFrame(drivers)

        # 3) Find qualifying session within same meeting
        qual_sessions = fetch("sessions", {"meeting_key": meeting_key, "session_name": "Qualifying"})
        df_qual_sess = pd.DataFrame(qual_sessions)

        if df_qual_sess.empty or "session_key" not in df_qual_sess.columns:
            print("  - No qualifying session found for meeting, grid_position will be NULL")
            df_qual_res = pd.DataFrame()
        else:
            qual_session_key = int(df_qual_sess.sort_values("date_start").iloc[0]["session_key"])
            qual_results = fetch("session_result", {"session_key": qual_session_key})
            df_qual_res = pd.DataFrame(qual_results)

        # Normalize join keys
        for _df in (df_race, df_drv, df_qual_res):
            if not _df.empty and "driver_number" in _df.columns:
                _df["driver_number"] = pd.to_numeric(_df["driver_number"], errors="coerce").astype("Int64")

        # Merge in driver name + team
        if not df_drv.empty and {"driver_number", "full_name", "team_name"}.issubset(df_drv.columns):
            df_race = df_race.merge(
                df_drv[["driver_number", "full_name", "team_name"]].drop_duplicates(subset=["driver_number"]),
                on="driver_number",
                how="left"
            )

        # Merge in qualifying position as grid_position
        if not df_qual_res.empty and {"driver_number", "position"}.issubset(df_qual_res.columns):
            df_grid = df_qual_res[["driver_number", "position"]].copy()
            df_grid = df_grid.rename(columns={"position": "grid_position"})
            df_grid["grid_position"] = pd.to_numeric(df_grid["grid_position"], errors="coerce")
            df_grid = df_grid.dropna(subset=["driver_number"]).drop_duplicates(subset=["driver_number"])
            df_race = df_race.merge(df_grid, on="driver_number", how="left")
        else:
            df_race["grid_position"] = pd.NA

        # Normalize numeric fields
        df_race["finish_position"] = pd.to_numeric(df_race.get("position"), errors="coerce")
        df_race["points"] = pd.to_numeric(df_race.get("points"), errors="coerce")
        df_race["grid_position"] = pd.to_numeric(df_race.get("grid_position"), errors="coerce")

        df_race["dnf_flag"] = df_race["finish_position"].isna().astype(int)
        df_race["meeting_key"] = meeting_key
        df_race["session_key"] = race_session_key

        keep = [
            "meeting_key",
            "session_key",
            "driver_number",
            "full_name",
            "team_name",
            "grid_position",
            "finish_position",
            "points",
            "dnf_flag",
        ]
        df_race = df_race[[c for c in keep if c in df_race.columns]]

        all_dfs.append(df_race)
        time.sleep(0.2)

    final_df = pd.concat(all_dfs, ignore_index=True)

    con.execute("DROP TABLE IF EXISTS race_results_enriched_2023;")
    con.register("final_df", final_df)
    con.execute("CREATE TABLE race_results_enriched_2023 AS SELECT * FROM final_df;")
    con.unregister("final_df")

    print("\nEnriched table created.")
    print("Counts:", con.execute(
        "SELECT COUNT(*), COUNT(grid_position), COUNT(finish_position) FROM race_results_enriched_2023"
    ).fetchall())

    print("Sample podium rows:")
    print(con.execute("""
        SELECT session_key, full_name, team_name, grid_position, finish_position, points
        FROM race_results_enriched_2023
        WHERE finish_position <= 3
        ORDER BY session_key, finish_position
        LIMIT 9;
    """).fetchall())

    con.close()
    print("Done.")

if __name__ == "__main__":
    main()
