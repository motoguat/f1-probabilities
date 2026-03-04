import duckdb
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

DB_PATH = "data/f1.duckdb"

DEFAULT_ROLL = 10.0          # fallback roll_finish_3 if no history
FILL_FINISH_FOR_ROLL = 20.0  # used only when computing rolling history and finish_position is missing


def compute_roll_finish_3(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes roll_finish_3 in a leak-free way (shifted within driver history).
    Assumes history_df contains ONLY races strictly before the race being predicted.
    """
    h = history_df.sort_values(["driver_number", "date_start", "meeting_key", "session_key"]).copy()

    finish_for_roll = h["finish_position"].fillna(FILL_FINISH_FOR_ROLL)

    h["roll_finish_3"] = (
        finish_for_roll.groupby(h["driver_number"])
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    h["roll_finish_3"] = h["roll_finish_3"].fillna(DEFAULT_ROLL)
    return h


def fit_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fits a probability model. If the training set is degenerate (only one class),
    return None and we will predict the base rate.
    """
    if y_train.nunique() < 2:
        return None

    base = HistGradientBoostingClassifier(random_state=42)
    base.fit(X_train, y_train)

    # Calibrate if we have enough data; otherwise return base model
    # isotonic with cv can fail very early season if not enough samples per class
    try:
        calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calib.fit(X_train, y_train)
        return calib
    except Exception:
        return base


def predict_proba(model, X: pd.DataFrame, base_rate: float) -> np.ndarray:
    if model is None:
        return np.full(len(X), base_rate, dtype=float)
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    # Pull 2023 races + results, with date_start for strict ordering
    df = con.execute("""
        SELECT
            s.meeting_key,
            s.session_key,
            s.date_start,
            rr.driver_number,
            rr.full_name,
            rr.team_name,
            rr.grid_position,
            rr.finish_position
        FROM race_results_enriched_2023 rr
        JOIN sessions_2023 s
          ON rr.session_key = s.session_key
        WHERE rr.grid_position IS NOT NULL
        ORDER BY s.date_start, s.meeting_key, s.session_key, rr.driver_number
    """).df()
    con.close()

    # Target for WIN only
    df["y_win"] = np.where(df["finish_position"].notna(), (df["finish_position"] == 1).astype(int), np.nan)

    # Identify races in chronological order
    races = (
        df[["meeting_key", "session_key", "date_start"]]
        .drop_duplicates()
        .sort_values(["date_start", "meeting_key", "session_key"])
        .reset_index(drop=True)
    )

    oos_rows = []
    used_races = 0

    for i in range(len(races)):
        mk = int(races.loc[i, "meeting_key"])
        sk = int(races.loc[i, "session_key"])
        dt = races.loc[i, "date_start"]

        # Train on strictly earlier races (date_start < current date_start)
        train_hist = df[df["date_start"] < dt].copy()

        # Predict current race rows
        race_df = df[(df["meeting_key"] == mk) & (df["session_key"] == sk)].copy()
        if race_df.empty:
            continue

        # Need at least *some* history to create useful roll; otherwise we still run with default
        if not train_hist.empty:
            train_hist = compute_roll_finish_3(train_hist)
            # For roll feature at prediction time, we need each driver's last computed roll from history
            last_roll = (
                train_hist.sort_values(["driver_number", "date_start", "meeting_key", "session_key"])
                .groupby("driver_number", as_index=False)
                .tail(1)[["driver_number", "roll_finish_3"]]
            )
            race_df = race_df.merge(last_roll, on="driver_number", how="left")
            race_df["roll_finish_3"] = race_df["roll_finish_3"].fillna(DEFAULT_ROLL)
        else:
            race_df["roll_finish_3"] = DEFAULT_ROLL

        # Build training set (only rows with known y)
        if train_hist.empty:
            # no training data yet → use base-rate guess (tiny)
            base_rate = 1.0 / max(len(race_df), 1)
            model = None
        else:
            train_labeled = train_hist[train_hist["y_win"].notna()].copy()
            if train_labeled.empty:
                base_rate = 1.0 / max(len(race_df), 1)
                model = None
            else:
                X_train = train_labeled[["grid_position", "roll_finish_3"]].copy()
                y_train = train_labeled["y_win"].astype(int)
                base_rate = float(y_train.mean())
                model = fit_model(X_train, y_train)

        X_race = race_df[["grid_position", "roll_finish_3"]].copy()
        p_win = predict_proba(model, X_race, base_rate)

        race_df["p_win"] = p_win
        race_df["y_true"] = race_df["y_win"]

        oos_rows.append(
            race_df[[
                "meeting_key","session_key","date_start","driver_number","full_name","team_name",
                "grid_position","roll_finish_3","finish_position","p_win","y_true"
            ]]
        )
        used_races += 1

    if not oos_rows:
        raise RuntimeError("No out-of-sample rows produced. Check joins / data availability.")

    df_oos = pd.concat(oos_rows, ignore_index=True)

    # Overall metrics (only rows with known y_true)
    scored = df_oos[df_oos["y_true"].notna()].copy()
    scored["y_true"] = scored["y_true"].astype(int)

    metrics = []
    if len(scored) > 0 and scored["y_true"].nunique() == 2:
        auc = roc_auc_score(scored["y_true"], scored["p_win"])
    else:
        auc = np.nan

    brier = brier_score_loss(scored["y_true"], scored["p_win"]) if len(scored) > 0 else np.nan
    ll = log_loss(scored["y_true"], scored["p_win"]) if len(scored) > 0 and scored["y_true"].nunique() == 2 else np.nan

    metrics.append({
        "season": 2023,
        "market": "win",
        "races_predicted": used_races,
        "rows_scored": int(len(scored)),
        "auc": float(auc) if auc == auc else None,
        "brier": float(brier) if brier == brier else None,
        "logloss": float(ll) if ll == ll else None,
    })
    df_metrics = pd.DataFrame(metrics)

    # Write to DuckDB
    conw = duckdb.connect(DB_PATH)

    conw.execute("DROP TABLE IF EXISTS oos_predictions_win_2023;")
    conw.register("oos_df", df_oos)
    conw.execute("CREATE TABLE oos_predictions_win_2023 AS SELECT * FROM oos_df;")
    conw.unregister("oos_df")

    conw.execute("DROP TABLE IF EXISTS oos_metrics_win_2023;")
    conw.register("m_df", df_metrics)
    conw.execute("CREATE TABLE oos_metrics_win_2023 AS SELECT * FROM m_df;")
    conw.unregister("m_df")

    conw.close()

    print(f"Saved oos_predictions_win_2023 rows: {len(df_oos)}")
    print("Saved oos_metrics_win_2023:")
    print(df_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
