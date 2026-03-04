import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

DB_PATH = "data/f1.duckdb"
TARGET = "podium" #options: "podium", "win", "top10"

def main():
    con = duckdb.connect(DB_PATH)

    # FULL dataset (score on this)
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

    # TRAIN dataset (train on this)
    df_train = df_full[df_full["finish_position"].notna()].copy()

    # Target
    if TARGET == "podium":
        df_train["y"] = (df_train["finish_position"] <= 3).astype(int)
    elif TARGET == "win":
        df_train["y"] = (df_train["finish_position"] == 1).astype(int)
    elif TARGET == "top10":
        df_train["y"] = (df_train["finish_position"] <= 10).astype(int)
    else:
        raise ValueError(f"Invalid TARGET: {TARGET}")	

    # Rolling avg finish over last 3 races (shifted to avoid lookahead)
    df_train = df_train.sort_values(["driver_number", "meeting_key"]).copy()
    df_train["roll_finish_3"] = (
        df_train.groupby("driver_number")["finish_position"]
          .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )
    df_train["roll_finish_3"] = df_train["roll_finish_3"].fillna(10)

    # Rolling feature for FULL scoring set:
    # Use last known roll_finish_3 from training data per driver.
    last_roll = (
        df_train.sort_values(["driver_number", "meeting_key"])
                .groupby("driver_number")["roll_finish_3"]
                .last()
                .reset_index()
                .rename(columns={"roll_finish_3": "roll_finish_3_last"})
    )

    df_full = df_full.copy()
    df_full = df_full.merge(last_roll, on="driver_number", how="left")
    df_full["roll_finish_3"] = df_full["roll_finish_3_last"].fillna(10)
    df_full = df_full.drop(columns=["roll_finish_3_last"])

    # Features
    X = df_train[["grid_position", "roll_finish_3"]].copy()
    y = df_train["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base = HistGradientBoostingClassifier(random_state=42)
    base.fit(X_train, y_train)

    # Calibrate probabilities (key for a “probability engine”)
    calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calib.fit(X_train, y_train)

    p_test = calib.predict_proba(X_test)[:, 1]

    # Score ALL rows (FULL)
    X_full = df_full[["grid_position", "roll_finish_3"]].copy()
    p_all = calib.predict_proba(X_full)[:, 1]

    # --- Save FULL predictions to DuckDB (all rows) ---
    table_name = f"predictions_{TARGET}_2023_full"

    # Map target → probability column name
    prob_col = {
        "podium": "p_podium",
        "win": "p_win",
        "top10": "p_top10"
    }[TARGET]

    prob_col = {"podium": "p_podium", "win": "p_win", "top10": "p_top10"}[TARGET]
    table_name = f"predictions_{TARGET}_2023_full"

    preds = df_full[[
        "meeting_key","session_key","driver_number","full_name","team_name",
        "grid_position","roll_finish_3","finish_position"
    ]].copy()

    preds[prob_col] = p_all

    # y_true only where finish_position exists
    if TARGET == "podium":
        preds["y_true"] = np.where(preds["finish_position"].notna(),
                                   (preds["finish_position"] <= 3).astype(int),
                                   np.nan)
    elif TARGET == "win":
        preds["y_true"] = np.where(preds["finish_position"].notna(),
                                   (preds["finish_position"] == 1).astype(int),
                                   np.nan)
    elif TARGET == "top10":
        preds["y_true"] = np.where(preds["finish_position"].notna(),
                                   (preds["finish_position"] <= 10).astype(int),
                                   np.nan)
    else:
        raise ValueError(f"Invalid TARGET: {TARGET}")

    # sort first (recommended)
    preds = preds.sort_values(["session_key", prob_col], ascending=[True, False])

    # save
    table_name = f"predictions_{TARGET}_2023_full"
    con2 = duckdb.connect(DB_PATH)
    con2.execute(f"DROP TABLE IF EXISTS {table_name};")
    con2.register("preds_df", preds)
    con2.execute(f"CREATE TABLE {table_name} AS SELECT * FROM preds_df;")
    con2.unregister("preds_df")
    con2.close()
    
    print(f"\nSaved {len(preds)} rows to DuckDB table: {table_name}")

    print("AUC:", round(roc_auc_score(y_test, p_test), 4))
    print("Brier:", round(brier_score_loss(y_test, p_test), 4))

    # Show a few example predictions
    df_test = df_train.loc[X_test.index, ["full_name", "team_name", "grid_position", "roll_finish_3"]].copy()
    df_test["p_podium"] = p_test
    df_test = df_test.sort_values(["grid_position", "roll_finish_3"]).head(15)

    print("\nExample predictions (best grid positions):")
    for _, r in df_test.iterrows():
        print(
            f"  P(podium)={r['p_podium']:.2f} | grid={int(r['grid_position'])} "
            f"| roll_finish_3={r['roll_finish_3']:.1f} | {r['full_name']} ({r['team_name']})"
        )

if __name__ == "__main__":
    main()
