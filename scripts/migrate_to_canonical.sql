-- Canonical predictions table
DROP TABLE IF EXISTS predictions_all;

CREATE TABLE predictions_all AS
WITH base AS (
  SELECT
    2023::INTEGER AS season,
    meeting_key,
    session_key,
    driver_number,
    full_name,
    team_name,
    grid_position,
    roll_finish_3,
    finish_position,
    p_podium
  FROM predictions_podium_2023_full
),
w AS (
  SELECT
    session_key,
    driver_number,
    p_win
  FROM predictions_win_2023_full
),
t AS (
  SELECT
    session_key,
    driver_number,
    p_top10
  FROM predictions_top10_2023_full
)
SELECT
  base.season,
  base.meeting_key,
  base.session_key,
  base.driver_number,
  base.full_name,
  base.team_name,
  base.grid_position,
  base.roll_finish_3,
  base.finish_position,
  w.p_win,
  base.p_podium,
  t.p_top10
FROM base
LEFT JOIN w ON base.session_key = w.session_key AND base.driver_number = w.driver_number
LEFT JOIN t ON base.session_key = t.session_key AND base.driver_number = t.driver_number;
