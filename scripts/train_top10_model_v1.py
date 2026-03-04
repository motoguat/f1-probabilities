import duckdb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

DB_PATH = "data/f1.duckdb"

def add_roll_finish_3(df: pd.DataFrame, fill_finish_for_roll: float) -> pd.DataFrame:
    """
    Adds roll_finish_3 = shifted rolling mean of last 3 finishes per driver.
    If finish_position is missing, use fill_finish_for_roll for history so rolling works.
    """
    df = df.sort_values(["driver_number", "meeting_key"]).copy()

    finish_for_roll = df["finish_position"].fillna(fill_finish_for_roll)
    df["roll_finish_3"] = (
        finish_for_roll.groupby(df["driver_number"])
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    df["roll_finish_3"] = df["roll_finish_3"].fillna(10)
    return df


def main():
    con = duckdb.connect(DB_PATH)

    # TRAINING SET: needs finish_position to create y and rolling feature cleanly
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

    # FULL PREDICTION SET: allow missing finish_position (e.g., DNFs/missing), but require grid
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

    # Target for Top 10
    df["y"] = (df["finish_position"] <= 10).astype(int)

    # Rolling feature
    df = add_roll_finish_3(df, fill_finish_for_roll=20)
    df_full = add_roll_finish_3(df_full, fill_finish_for_roll=20)

    # Features
    X = df[["grid_position", "roll_finish_3"]].copy()
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base = HistGradientBoostingClassifier(random_state=42)
    base.fit(X_train, y_train)

    calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calib.fit(X_train, y_train)

    p_test = calib.predict_proba(X_test)[:, 1]

    # Predict for ALL rows (df_full)
    X_full = df_full[["grid_position", "roll_finish_3"]].copy()
    p_all = calib.predict_proba(X_full)[:, 1]

    # --- Save FULL predictions to DuckDB ---
    table_name = "predictions_top10_2023_full"

    preds = df_full[[
        "meeting_key", "session_key", "driver_number",
        "full_name", "team_name", "grid_position",
        "roll_finish_3", "finish_position"
    ]].copy()

    preds["p_top10"] = p_all

    # y_true only where finish_position exists
    preds["y_true"] = np.where(
        preds["finish_position"].notna(),
        (preds["finish_position"] <= 10).astype(int),
        np.nan
    )

    preds = preds.sort_values(["session_key", "p_top10"], ascending=[True, False])

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
    df_test = df.loc[X_test.index, ["full_name", "team_name", "grid_position", "roll_finish_3"]].copy()
    df_test["p_top10"] = p_test
    df_test = df_test.sort_values(["grid_position", "roll_finish_3"]).head(15)

    print("\nExample predictions (best grid positions):")
    for _, r in df_test.iterrows():
        print(
            f"  P(top10)={r['p_top10']:.2f} | grid={int(r['grid_position'])} "
            f"| roll_finish_3={r['roll_finish_3']:.1f} | {r['full_name']} ({r['team_name']})"
        )


if __name__ == "__main__":
    main()
