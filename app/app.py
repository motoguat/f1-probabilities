import duckdb
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DB_PATH = str(REPO_ROOT / "data" / "f1.duckdb")

TARGETS = {
    "Podium (Top 3)": {"table": "v_predictions_podium_2023_full", "prob_col": "p_podium"},
    "Win": {"table": "v_predictions_win_2023_full", "prob_col": "p_win"},
    "Top 10": {"table": "v_predictions_top10_2023_full", "prob_col": "p_top10"},
}

# ----------------------------
# Helpers
# ----------------------------
def kelly_fraction(p: float, odds_decimal: float) -> float:
    b = max(odds_decimal - 1.0, 1e-9)
    f = (b * p - (1.0 - p)) / b
    return float(max(0.0, f))

def settle_from_finish(finish_position: float, market_name: str) -> int:
    if pd.isna(finish_position):
        return 0
    fp = int(finish_position)
    if market_name == "Win":
        return 1 if fp == 1 else 0
    if market_name == "Podium":
        return 1 if fp <= 3 else 0
    if market_name == "Top 10":
        return 1 if fp <= 10 else 0
    return 0

@st.cache_data
def load_sessions():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT
            meeting_key,
            session_key,
            country_name,
            circuit_short_name,
            date_start
        FROM sessions_2023
        ORDER BY date_start
    """).df()
    con.close()

    df["label"] = (
        df["country_name"].astype(str)
        + " — "
        + df["circuit_short_name"].astype(str)
        + " ("
        + df["date_start"].astype(str).str[:10]
        + ")"
    )
    return df

@st.cache_data
def load_predictions(table: str, prob_col: str):
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(f"""
        SELECT
            meeting_key,
            session_key,
            driver_number,
            full_name,
            team_name,
            grid_position,
            roll_finish_3,
            finish_position,
            {prob_col} AS prob
        FROM {table}
    """).df()
    con.close()
    return df

@st.cache_data
def load_compare_all():
    con_all = duckdb.connect(DB_PATH, read_only=True)
    df = con_all.execute("""
        WITH win AS (
          SELECT
            meeting_key,
            session_key,
            driver_number,
            full_name,
            team_name,
            grid_position,
            roll_finish_3,
            finish_position,
            p_win,
            y_true
          FROM v_predictions_win_2023_full
        ),
        pod AS (
          SELECT
            session_key,
            driver_number,
            p_podium,
            y_true
          FROM v_predictions_podium_2023_full
        ),
        t10 AS (
          SELECT
            session_key,
            driver_number,
            p_top10,
            y_true
          FROM v_predictions_top10_2023_full
        )
        SELECT
          w.meeting_key,
          w.session_key,
          w.driver_number,
          w.full_name,
          w.team_name,
          w.grid_position,
          w.roll_finish_3,
          w.finish_position,
          w.p_win,
          p.p_podium,
          t.p_top10,
          w.y_true AS y_true_win,
          p.y_true AS y_true_podium,
          t.y_true AS y_true_top10
        FROM win w
        LEFT JOIN pod p USING (session_key, driver_number)
        LEFT JOIN t10 t USING (session_key, driver_number)
    """).df()
    con_all.close()
    return df

# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="F1 Probabilities", layout="wide")
st.title("F1 Probabilities Viewer (2023)")

sessions = load_sessions()

tab_race, tab_season, tab_diag = st.tabs(["Race", "Season", "Diagnostics"])

# =========================================================
# RACE TAB
# =========================================================
with tab_race:
    target_label = st.selectbox("Target", list(TARGETS.keys()))
    table = TARGETS[target_label]["table"]
    prob_col = TARGETS[target_label]["prob_col"]

    preds_all = load_predictions(table, prob_col)

    available_session_keys = sorted(preds_all["session_key"].unique().tolist())
    sessions_avail = sessions[sessions["session_key"].isin(available_session_keys)].copy()

    if sessions_avail.empty:
        st.error(f"No races available in {table}.")
        st.stop()

    race_label = st.selectbox("Race", sessions_avail["label"].tolist())
    selected = sessions_avail[sessions_avail["label"] == race_label].iloc[0]
    session_key = int(selected["session_key"])

    st.subheader("Race comparison: Win vs Podium vs Top 10")

    compare_sql = f"""
    WITH base AS (
      SELECT
        meeting_key,
        session_key,
        driver_number,
        full_name,
        team_name,
        grid_position,
        roll_finish_3,
        finish_position,
        p_podium
      FROM v_predictions_podium_2023_full
      WHERE session_key = {session_key}
    ),
    w AS (
      SELECT session_key, driver_number, p_win
      FROM v_predictions_win_2023_full
      WHERE session_key = {session_key}
    ),
    t AS (
      SELECT session_key, driver_number, p_top10
      FROM v_predictions_top10_2023_full
      WHERE session_key = {session_key}
    )
    SELECT
      base.full_name,
      base.team_name,
      base.driver_number,
      base.grid_position,
      base.roll_finish_3,
      base.finish_position,
      w.p_win,
      base.p_podium,
      t.p_top10
    FROM base
    LEFT JOIN w ON base.session_key = w.session_key AND base.driver_number = w.driver_number
    LEFT JOIN t ON base.session_key = t.session_key AND base.driver_number = t.driver_number
    ORDER BY w.p_win DESC NULLS LAST, base.p_podium DESC NULLS LAST, t.p_top10 DESC NULLS LAST;
    """

    con_compare = duckdb.connect(DB_PATH, read_only=True)
    df_compare = con_compare.execute(compare_sql).df()
    con_compare.close()

    st.caption(f"session_key: {session_key}  |  rows: {len(df_compare)}")

    sort_metric = st.selectbox("Sort by", ["p_win", "p_podium", "p_top10"], index=0)
    df_compare = df_compare.sort_values(sort_metric, ascending=False)

    for col in ["p_win", "p_podium", "p_top10"]:
        if col in df_compare.columns:
            df_compare[col] = df_compare[col].round(3)

    st.dataframe(df_compare, use_container_width=True)

    # Rank + chart view (kept inside Race tab)
    df_race = preds_all[preds_all["session_key"] == session_key].copy()
    df_race = df_race.sort_values("prob", ascending=False)

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Ranked drivers")
        out = df_race[[
            "full_name", "team_name", "driver_number",
            "grid_position", "roll_finish_3", "finish_position", "prob"
        ]].copy()
        out["prob"] = out["prob"].round(4)
        st.dataframe(out, use_container_width=True)

    with right:
        st.subheader("Top N")
        top_n_chart = st.slider("Top N", 1, 20, 10, 1, key="chart_top_n")
        chart_df = df_race.head(top_n_chart)[["full_name", "prob"]].set_index("full_name")
        st.bar_chart(chart_df)

# =========================================================
# SEASON TAB
# =========================================================
with tab_season:
    st.header("Bankroll simulation (season)")

    df_compare_all = load_compare_all()

    # ---- Controls ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start_bankroll = st.number_input("Starting bankroll ($)", min_value=10.0, value=100.0, step=10.0)
    with c2:
        market = st.selectbox("Market", ["Win", "Podium", "Top 10"], index=1)
    with c3:
        kelly_mult = st.slider("Kelly multiplier", 0.0, 1.0, 0.25, 0.05)
    with c4:
        max_bet_frac = st.slider("Max bet (% bankroll)", 0.0, 0.25, 0.05, 0.01)

    c5, c6, c7 = st.columns(3)
    with c5:
        min_edge = st.slider("Min edge (p_model - p_book)", 0.0, 0.20, 0.01, 0.005, format="%.3f")
    with c6:
        min_ev = st.slider("Min EV per $1", 0.0, 0.50, 0.00, 0.01)
    with c7:
        top_n = st.slider("Max bets per race", 0, 10, 3, 1)

    # Market pricing assumptions (season tab)
    discount = st.slider(
        "Book odds discount (lower = worse payout / more vig)",
        min_value=0.80, max_value=1.00, value=0.92, step=0.01,
        key="season_discount"
    )
    sigma = st.slider(
        "Book mispricing noise (σ)",
        min_value=0.0, max_value=0.20, value=0.08, step=0.01,
        key="season_sigma"
    )

    seed = st.number_input("Random seed", min_value=0, value=7, step=1)
    run_sim = st.button("Run bankroll simulation")

    if run_sim:
        np.random.seed(int(seed))
        prob_col = {"Win": "p_win", "Podium": "p_podium", "Top 10": "p_top10"}[market]

        season = df_compare_all.copy()
        season = season.dropna(subset=[prob_col, "finish_position"]).copy()

        season["p_model"] = season[prob_col].astype(float).clip(lower=1e-6, upper=1 - 1e-6)
        season["fair_odds"] = 1.0 / season["p_model"]

        if float(sigma) > 0:
            noise = np.exp(np.random.normal(0.0, float(sigma), len(season)))
            season["book_odds"] = season["fair_odds"] * float(discount) * noise
        else:
            season["book_odds"] = season["fair_odds"] * float(discount)

        season["p_book"] = (1.0 / season["book_odds"]).clip(lower=1e-6, upper=1 - 1e-6)
        season["edge"] = season["p_model"] - season["p_book"]
        season["ev_per_$1"] = season["p_model"] * season["book_odds"] - 1.0

        race_keys = season[["meeting_key", "session_key"]].drop_duplicates().sort_values(["meeting_key", "session_key"])

        bankroll = float(start_bankroll)
        equity_rows = []
        bet_rows = []

        for _, rk in race_keys.iterrows():
            sk = int(rk["session_key"])
            mk = int(rk["meeting_key"])

            race = season[season["session_key"] == sk].copy()
            if race.empty:
                equity_rows.append({"meeting_key": mk, "session_key": sk, "bankroll": bankroll, "race_profit": 0.0, "bets": 0})
                continue

            race = race[(race["edge"] >= float(min_edge)) & (race["ev_per_$1"] >= float(min_ev))].copy()
            race = race.sort_values("ev_per_$1", ascending=False)
            race = race.head(int(top_n)) if int(top_n) > 0 else race.iloc[0:0]

            race_profit = 0.0
            bets_made = 0

            for _, r in race.iterrows():
                p = float(r["p_model"])
                o = float(r["book_odds"])

                kf = kelly_fraction(p, o)
                stake_frac = min(float(max_bet_frac), float(kelly_mult) * kf)
                stake = bankroll * stake_frac
                if stake <= 0:
                    continue

                hit = settle_from_finish(r["finish_position"], market)
                profit = (stake * (o - 1.0)) if hit == 1 else (-stake)

                bankroll += profit
                race_profit += profit
                bets_made += 1

                bet_rows.append({
                    "meeting_key": mk,
                    "session_key": sk,
                    "market": market,
                    "driver": r["full_name"],
                    "team": r["team_name"],
                    "grid": r["grid_position"],
                    "finish": r["finish_position"],
                    "p_model": round(p, 4),
                    "p_book": round(float(r["p_book"]), 4),
                    "edge": round(float(r["edge"]), 4),
                    "book_odds": round(o, 3),
                    "ev_per_$1": round(float(r["ev_per_$1"]), 4),
                    "kelly_frac": round(kf, 4),
                    "stake": round(stake, 2),
                    "hit": hit,
                    "profit": round(profit, 2),
                    "bankroll_after": round(bankroll, 2),
                })

            equity_rows.append({
                "meeting_key": mk,
                "session_key": sk,
                "bankroll": round(bankroll, 2),
                "race_profit": round(race_profit, 2),
                "bets": bets_made
            })

        df_equity = pd.DataFrame(equity_rows)
        df_ledger = pd.DataFrame(bet_rows)

        st.subheader("Equity curve")
        if not df_equity.empty:
            st.line_chart(df_equity.set_index("meeting_key")["bankroll"])
        else:
            st.info("No equity rows produced.")

        st.subheader("Bet ledger")
        st.dataframe(df_ledger, use_container_width=True)

# =========================================================
# DIAGNOSTICS TAB
# =========================================================
with tab_diag:
    st.header("Diagnostics — Win Model (OOS 2023)")

    con = duckdb.connect(DB_PATH, read_only=True)
    oos = con.execute("""
        SELECT p_win, y_true
        FROM oos_predictions_win_2023
        WHERE y_true IS NOT NULL
          AND p_win IS NOT NULL
    """).df()
    con.close()

    if oos.empty:
        st.error("No scored OOS rows found.")
        st.stop()

    # Clean
    oos["p_win"] = oos["p_win"].clip(1e-6, 1 - 1e-6)
    oos["y_true"] = (oos["y_true"] >= 0.5).astype(int)

    y = oos["y_true"].values
    p = oos["p_win"].values

    # Metrics
    from sklearn.metrics import roc_auc_score, brier_score_loss

    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)

    c1, c2, c3 = st.columns(3)
    c1.metric("Scored Rows", f"{len(oos):,}")
    c2.metric("AUC (OOS)", f"{auc:.4f}")
    c3.metric("Brier (OOS)", f"{brier:.4f}")

    # ---------------------------------
    # Calibration Table
    # ---------------------------------
    st.subheader("Calibration")

    n_bins = st.slider("Number of bins", 5, 20, 10)

    oos["bin"] = pd.qcut(oos["p_win"], q=n_bins, duplicates="drop")

    cal = (
        oos.groupby("bin", observed=False)
        .agg(
            count=("p_win", "size"),
            avg_pred=("p_win", "mean"),
            actual_rate=("y_true", "mean"),
        )
        .reset_index()
    )

    st.dataframe(cal, use_container_width=True)

    # ---------------------------------
    # Calibration Curve
    # ---------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(cal["avg_pred"], cal["actual_rate"], marker="o")
    ax.set_xlabel("Average predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration Curve (OOS)")
    st.pyplot(fig)

    # ---------------------------------
    # Probability Histogram
    # ---------------------------------
    st.subheader("Probability Distribution")

    fig2, ax2 = plt.subplots()
    ax2.hist(oos["p_win"], bins=25)
    ax2.set_xlabel("Predicted P(win)")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    # ---------------------------------
    # Feature Influence Proxy (v0.1)
    # ---------------------------------
    st.subheader("Feature Influence (Proxy)")

    st.caption("Correlation between features and predicted probability (not causal importance).")

    con = duckdb.connect(DB_PATH, read_only=True)
    feat_df = con.execute("""
        SELECT grid_position, roll_finish_3, p_win
        FROM oos_predictions_win_2023
        WHERE y_true IS NOT NULL
    """).df()
    con.close()

    corr = feat_df[["grid_position", "roll_finish_3"]].corrwith(feat_df["p_win"])
    imp = pd.DataFrame({
        "feature": corr.index,
        "corr_with_p_win": corr.values
    }).sort_values("corr_with_p_win")

    st.dataframe(imp, use_container_width=True)
