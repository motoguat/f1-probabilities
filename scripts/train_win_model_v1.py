import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

DB_PATH = "data/f1.duckdb"
TARGET = "win" #options: "podium", "win", "top10"

def main():
    con = duckdb.connect(DB_PATH)

    df = con.execute("""
        SELECT
            meeting_key,
            session_key,
            driver_number,
            full_name,
            team_name,
            grid_position,
            finish_position
        FROM race_results_enriched_2023
        WHERE grid_position IS NOT NULL
          AND finish_position IS NOT NULL
    """).df()

    # Full prediction set (grid only; finish can be NULL)
    df_full = con.execute("""
        SELECT
            meeting_key,
            session_key,
            driver_number,
            full_name,
            team_name,
            grid_position,
            finish_position
        FROM race_results_enriched_2023
        WHERE grid_position IS NOT NULL
    """).df()

    con.close()

    # Target
    if TARGET == "podium":
        df["y"] = (df["finish_position"] <= 3).astype(int)
    elif TARGET == "win":
        df["y"] = (df["finish_position"] == 1).astype(int)
    elif TARGET == "top10":
        df["y"] = (df["finish_position"] <= 10).astype(int)
    else:
        raise ValueError(f"Invalid TARGET: {TARGET}")	

    # Rolling avg finish over last 3 races (shifted to avoid lookahead)
    df = df.sort_values(["driver_number", "meeting_key"]).copy()
    df["roll_finish_3"] = (
        df.groupby("driver_number")["finish_position"]
          .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )
    df["roll_finish_3"] = df["roll_finish_3"].fillna(10)

    # Rolling avg finish for df_full (prediction set)
    df_full = df_full.sort_values(["driver_number", "meeting_key"]).copy()

    # Use finish_position for history, but fill NaN so rolling works
    finish_for_roll = df_full["finish_position"].fillna(20)

    df_full["roll_finish_3"] = (
        finish_for_roll.groupby(df_full["driver_number"])
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    df_full["roll_finish_3"] = df_full["roll_finish_3"].fillna(10)

    # Features
    X = df[["grid_position", "roll_finish_3"]].copy()
    y = df["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base = HistGradientBoostingClassifier(random_state=42)
    base.fit(X_train, y_train)

    # Calibrate probabilities (key for a “probability engine”)
    calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calib.fit(X_train, y_train)

    p_test = calib.predict_proba(X_test)[:, 1]

    # Features for full-table predictions (df_full)
    X_full = df_full[["grid_position", "roll_finish_3"]].copy()
    p_all = calib.predict_proba(X_full)[:, 1]    

    # --- Save FULL predictions to DuckDB (all rows) ---
    table_name = "predictions_win_2023_full"

    preds = df_full[[
        "meeting_key", "session_key", "driver_number",
        "full_name", "team_name", "grid_position", "roll_finish_3", "finish_position"
    ]].copy()

    preds["p_win"] = p_all
    preds["y_true_win"] = np.where(
        preds["finish_position"].notna(),
        (preds["finish_position"] == 1).astype(int),
        np.nan
    )

    preds = preds.sort_values(["session_key", "p_win"], ascending=[True, False])

    con2 = duckdb.connect(DB_PATH)
    con2.execute(f"DROP TABLE IF EXISTS {table_name};")
    con2.register("preds_df", preds)
    con2.execute(f"CREATE TABLE {table_name} AS SELECT * FROM preds_df;")
    con2.unregister("preds_df")
    con2.close()

    print(f"\nSaved {len(preds)} rows to DuckDB table: {table_name}")
    # Show a few example predictions
    df_test = df.loc[X_test.index, ["full_name", "team_name", "grid_position", "roll_finish_3"]].copy()
    df_test["p_win"] = p_test
    df_test = df_test.sort_values(["grid_position", "roll_finish_3"]).head(15)

    print("\nExample predictions (best grid positions):")
    for _, r in df_test.iterrows():
        print(
            f"  P(win)={r['p_win']:.2f} | grid={int(r['grid_position'])} "
            f"| roll_finish_3={r['roll_finish_3']:.1f} | {r['full_name']} ({r['team_name']})"
        )

if __name__ == "__main__":
    main()
