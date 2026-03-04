import duckdb
import pandas as pd
import streamlit as st
import numpy as np

DB_PATH = "data/f1.duckdb"

TARGETS = {
    "Podium (Top 3)": {"table": "v_predictions_podium_2023_full", "prob_col": "p_podium"},
    "Win": {"table": "v_predictions_win_2023_full", "prob_col": "p_win"},
    "Top 10": {"table": "v_predictions_top10_2023_full", "prob_col": "p_top10"},
}

def kelly_fraction(p: float, odds_decimal: float) -> float:
    """
    Kelly for decimal odds:
      b = odds - 1
      f* = (b*p - (1-p)) / b
    """
    b = max(odds_decimal - 1.0, 1e-9)
    f = (b * p - (1.0 - p)) / b
    return float(max(0.0, f))


def compute_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0

st.set_page_config(page_title="F1 Probabilities", layout="wide")
st.title("F1 Probabilities Viewer (2023)")

tab_race, tab_season, tab_diag = st.tabs(["Race", "Season", "Diagnostics"])
    st.header("Model Diagnostics (Win)")
    st.write("TODO diagnostics here")with tab_diag:
    st.header("Model Diagnostics (Win)")
    st.write("Diagnostics tab is wired ✅")

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

sessions = load_sessions()

with tab_race:
    st.write("TODO: move reace UI here")

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

    con_compare = duckdb.connect(DB_PATH)
    df_compare = con_compare.execute(compare_sql).df()
    con_compare.close()

    con_all = duckdb.connect(DB_PATH)
    df_compare_all = con_all.execute("""
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

    st.caption(f"session_key: {session_key}  |  rows: {len(df_compare)}")
    sort_metric = st.selectbox(
        "Sort by",
        ["p_win", "p_podium", "p_top10"],
        index=0
    )

    df_compare = df_compare.sort_values(sort_metric, ascending=False)

    for col in ["p_win", "p_podium", "p_top10"]:
        if col in df_compare.columns:
            df_compare[col] = df_compare[col].round(3)

    # =========================
    # Market pricing assumptions (shared)
    # =========================
    st.header("Market pricing assumptions")

    discount = st.slider(
        "Book odds discount (lower = worse payout / more vig)",
        min_value=0.80,
        max_value=1.00,
        value=0.92,
        step=0.01,
        key="shared_discount",
    )

    sigma = st.slider(
        "Book mispricing noise (σ)",
        min_value=0.0,
        max_value=0.20,
        value=0.08,
        step=0.01,
        key="shared_sigma",
    )

    metric_for_bets = st.selectbox(
        "Use which probability for betting math?",
        ["p_win", "p_podium", "p_top10"],
        index=0,
    )

    # Avoid divide-by-zero
    p = df_compare[metric_for_bets].clip(lower=1e-6, upper=1 - 1e-6)

    df_bets = df_compare.copy()
    df_bets["p_model"] = p

    df_bets = df_bets[df_bets["p_model"] > 0.01].copy()

    # Decimal odds

    # --- Simulated "book" odds (independent from model) ---
    noise = sigma

    rng = np.random.default_rng(42)

    # Fair odds from your model
    df_bets["fair_odds"] = 1.0 / df_bets["p_model"]

    # Book probability = model prob + noise, then convert to odds
    p_book = (df_bets["p_model"] + rng.normal(0, noise, size=len(df_bets))).clip(1e-3, 1 - 1e-3)

    # Apply your vig/discount as worse payout (lower odds)
    df_bets["book_odds"] = (1.0 / p_book) * discount
    df_bets["implied_p_book"] = 1.0 / df_bets["book_odds"]

    # DEBUG
    st.caption(f"DEBUG: discount={discount}")

    # Edges / EV
    df_bets["edge_p"] = (df_bets["p_model"] - df_bets["implied_p_book"])
    df_bets["ev_per_$1"] = (df_bets["p_model"] * df_bets["book_odds"] - 1.0)

    df_bets["positive_ev"] = df_bets["ev_per_$1"] > 0

    # Kelly fraction (optional, but useful): f* = (b*p - q)/b where b = odds-1
    b = (df_bets["book_odds"] - 1.0).clip(lower=1e-6)
    q = 1.0 - df_bets["p_model"]
    df_bets["kelly_frac"] = ((b * df_bets["p_model"] - q) / b).clip(lower=0.0, upper=1.0)

    df_bets["positive_ev"] = df_bets["ev_per_$1"] > 0

    # Make it readable
    show_cols = [
        "full_name", "team_name", "grid_position", "finish_position",
        metric_for_bets, "fair_odds", "book_odds", "implied_p_book",
        "edge_p", "ev_per_$1", "kelly_frac"
    ]

    for col in ["fair_odds", "book_odds"]:
        df_bets[col] = df_bets[col].round(2)
    for col in ["p_model", "implied_p_book", "edge_p", "ev_per_$1", "kelly_frac"]:
        df_bets[col] = df_bets[col].round(3)

    # Build display dataframe
    df_display = df_bets[show_cols].copy()

    # Sort by EV (best first)
    df_display = df_display.sort_values("ev_per_$1", ascending=False)

    st.subheader("Positive EV Opportunities")

    positive_df = df_display[df_display["ev_per_$1"] > 0]

    def highlight_ev(row):
        if row["ev_per_$1"] > 0:
            return ["background-color: rgba(0, 200, 0, 0.2)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        positive_df.style.apply(highlight_ev, axis=1),
        width="stretch"
    )

    st.subheader("All Simulated Bets")

    st.dataframe(
        df_display,
        width="stretch"
    )

    st.dataframe(df_compare, width="stretch")

with tab_season:
    st.write("TODO: move season UI here")

with tab_diag:
    st.write("TODO: diagnostics here")

# =========================
# Market pricing assumptions (shared)
# =========================
# =========================
# Bankroll simulation (season)
# =========================
st.header("Bankroll simulation (season)")

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

seed = st.number_input("Random seed", min_value=0, value=7, step=1)
run_sim = st.button("Run bankroll simulation")

# ---- Helpers ----
def kelly_fraction(p: float, odds_decimal: float) -> float:
    b = max(odds_decimal - 1.0, 1e-9)
    f = (b * p - (1.0 - p)) / b
    return float(max(0.0, f))

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min())

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

if run_sim:
    np.random.seed(int(seed))

    prob_col = {"Win": "p_win", "Podium": "p_podium", "Top 10": "p_top10"}[market]

    season = df_compare_all.copy()
    season = season.dropna(subset=[prob_col, "finish_position"]).copy()

    # Book pricing (same idea as your single-race betting section)
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

    # Chronological order: meeting_key is fine for 2023
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

        # Select bets
        race = race[(race["edge"] >= float(min_edge)) & (race["ev_per_$1"] >= float(min_ev))].copy()
        race = race.sort_values("ev_per_$1", ascending=False)
        if int(top_n) > 0:
            race = race.head(int(top_n))
        else:
            race = race.iloc[0:0]  # no bets

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

            # outcome from finish_position
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

    # after loop
    st.caption(f"DEBUG: equity_rows={len(equity_rows)} bet_rows={len(bet_rows)}")

    df_equity = pd.DataFrame(equity_rows)
    df_ledger = pd.DataFrame(bet_rows)

    st.subheader("Equity curve")

    if not df_equity.empty: 
        equity = df_equity["bankroll"].astype(float)

        st.line_chart(df_equity.set_index("meeting_key")["bankroll"])


        # ---- Performance metrics ----
        start_bankroll = float(equity.iloc[0])
        end_bankroll = float(equity.iloc[-1])
        total_return = (end_bankroll / start_bankroll) - 1.0

        step_returns = equity.pct_change().fillna(0.0)
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1.0
        max_drawdown = float(drawdown.min())

        win_rate = float((step_returns > 0).mean())
        ret_vol = float(step_returns.std())
        sharpe_like = float(step_returns.mean() / ret_vol) if ret_vol > 0 else 0.0

        best_step = float(step_returns.max())
        worst_step = float(step_returns.min())

        st.subheader("Bankroll performance")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Start bankroll", f"${start_bankroll:,.0f}")
        m2.metric("End bankroll", f"${end_bankroll:,.0f}")
        m3.metric("Total return", f"{total_return*100:.1f}%")
        m4.metric("Max drawdown", f"{max_drawdown*100:.1f}%")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Win rate (step)", f"{win_rate*100:.1f}%")
        m6.metric("Volatility (step)", f"{ret_vol*100:.2f}%")
        m7.metric("Sharpe (step)", f"{sharpe_like:.2f}")
        m8.metric("Best / Worst step", f"{best_step*100:.1f}% / {worst_step*100:.1f}%")

    else:
    	st.info("No equity rows produced.")

    st.subheader("Bet ledger")
    st.dataframe(df_ledger, use_container_width=True)

metric = st.selectbox("Chart metric", ["p_win", "p_podium", "p_top10"], index=0)
chart_df = df_compare[["full_name", metric]].dropna().head(20).set_index("full_name")
st.bar_chart(chart_df)
st.caption(f"Table: `{table}` | Column: `{prob_col}` | session_key: **{session_key}**")

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
