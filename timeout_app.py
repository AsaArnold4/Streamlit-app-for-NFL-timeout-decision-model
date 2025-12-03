# timeout_app.py

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from xgboost import XGBRegressor

# =========================================================
# 1. Load trained model + feature columns
# =========================================================
MODEL_PATH = "timeout_wp_xgb.pkl"
FEATURES_PATH = "feature_columns.json"

xgb_model: XGBRegressor = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)

# These MUST match your training script
feature_cols_numeric = [
    "score_differential",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "clock_stop",
    "offense_trailing",
]

feature_cols_categorical = [
    "qtr",
    "play_type",
]

# =========================================================
# 2. Prescriptive helpers (same logic as your script)
# =========================================================

def _build_design_matrix_from_situations(
    decisions_df: pd.DataFrame,
    call_timeout: bool,
    feature_cols_numeric,
    feature_cols_categorical,
    full_feature_columns,
) -> pd.DataFrame:
    """
    Internal helper to convert a decisions_df into the X design matrix
    for a given action (call timeout vs not).

    Key design choices:

        - Pure WP model, no timeout flags as features.

        - No-timeout branch:
              Use state exactly as entered.

        - Timeout branch:
              Same state, but clock_stop = 1 (timeout stops the clock),
              timeouts unchanged (local effect of clock vs running).
    """

    df = decisions_df.copy()

    # Ensure trailing flag is set based on score_differential
    df["offense_trailing"] = (df["score_differential"] < 0).astype(int)

    # Normalize clock_stop as integer 0/1
    df["clock_stop"] = df["clock_stop"].astype(int).clip(lower=0, upper=1)

    if call_timeout:
        # Local timeout effect: stop the clock, keep timeouts the same
        df["clock_stop"] = 1
    # else: leave as entered

    # Separate numeric and categorical as in training
    X_num_new = df[feature_cols_numeric].copy()
    X_cat_new = df[feature_cols_categorical].copy()

    # One-hot encode categoricals
    X_cat_dummies_new = pd.get_dummies(X_cat_new, drop_first=True)

    # Combine numeric + categoricals
    X_new = pd.concat(
        [X_num_new.reset_index(drop=True), X_cat_dummies_new.reset_index(drop=True)],
        axis=1,
    )

    # Align columns to the training design matrix (FEATURE_COLUMNS)
    X_new = X_new.reindex(columns=full_feature_columns, fill_value=0)

    return X_new


def add_timeout_probs(
    decisions_df: pd.DataFrame,
    model: XGBRegressor,
    full_feature_columns=None,
) -> pd.DataFrame:
    """
    Compute nfl4th-style timeout decision probabilities:

        - wp_timeout: predicted WP if offense calls timeout now
        - wp_no_timeout: predicted WP if offense does NOT call timeout
        - timeout_boost: wp_timeout - wp_no_timeout
    """
    if full_feature_columns is None:
        full_feature_columns = FEATURE_COLUMNS

    required_for_decisions = [
        "qtr",
        "game_seconds_remaining",
        "half_seconds_remaining",
        "score_differential",
        "wp",  # for display only
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "posteam",
        "defteam",
        "play_type",
        "clock_stop",
    ]

    missing_decision_cols = [c for c in required_for_decisions if c not in decisions_df.columns]
    if missing_decision_cols:
        raise ValueError(
            f"decisions_df is missing required columns: {missing_decision_cols}"
        )

    # Build design matrices for both decisions: call timeout vs no timeout
    X_timeout = _build_design_matrix_from_situations(
        decisions_df,
        call_timeout=True,
        feature_cols_numeric=feature_cols_numeric,
        feature_cols_categorical=feature_cols_categorical,
        full_feature_columns=full_feature_columns,
    )

    X_no_timeout = _build_design_matrix_from_situations(
        decisions_df,
        call_timeout=False,
        feature_cols_numeric=feature_cols_numeric,
        feature_cols_categorical=feature_cols_categorical,
        full_feature_columns=full_feature_columns,
    )

    # Predict WP for each action
    wp_timeout = model.predict(X_timeout)
    wp_no_timeout = model.predict(X_no_timeout)

    # Clip to [0, 1] just in case
    wp_timeout = np.clip(wp_timeout, 0, 1)
    wp_no_timeout = np.clip(wp_no_timeout, 0, 1)

    probs = decisions_df.copy()
    probs["wp_timeout"] = wp_timeout
    probs["wp_no_timeout"] = wp_no_timeout
    probs["timeout_boost"] = probs["wp_timeout"] - probs["wp_no_timeout"]

    return probs


def make_timeout_table(probs: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table for timeout vs no-timeout.

    Adds recommendation based on timeout_boost:
        - > 0.02   -> "Recommendation: Call a timeout"
        - <= 0.00  -> "Recommendation: Do not call a timeout"
        - (0,0.02] -> "Recommendation: Toss Up"
    """
    table_cols = [
        "posteam",
        "defteam",
        "qtr",
        "game_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "score_differential",
        "clock_stop",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "wp_no_timeout",
        "wp_timeout",
        "timeout_boost",
    ]

    existing = [c for c in table_cols if c in probs.columns]
    table = probs[existing].copy()

    def _recommend(boost: float) -> str:
        if boost > 0.02:
            return "Recommendation: Call a timeout"
        elif boost <= 0.0:
            return "Recommendation: Do not call a timeout"
        else:
            return "Recommendation: Toss Up"

    if "timeout_boost" in table.columns:
        table["recommendation"] = table["timeout_boost"].apply(_recommend)
        table = table.sort_values("timeout_boost", ascending=False)

    return table

# ---- Delay of game vs timeout helpers ----

def _build_design_matrix_delay_vs_timeout(
    decisions_df: pd.DataFrame,
    accept_penalty: bool,
    feature_cols_numeric,
    feature_cols_categorical,
    full_feature_columns,
) -> pd.DataFrame:
    """
    Construct X for either:
        - accept_penalty = True   (delay-of-game)
        - accept_penalty = False  (use timeout)
    """
    df = decisions_df.copy()

    df["offense_trailing"] = (df["score_differential"] < 0).astype(int)
    df["clock_stop"] = df["clock_stop"].astype(int).clip(lower=0, upper=1)

    if accept_penalty:
        # Accept delay-of-game penalty:
        df["yardline_100"] = (df["yardline_100"] + 5).clip(upper=100)
        df["ydstogo"] = df["ydstogo"] + 5
        # Timeouts unchanged, clock_stop unchanged
    else:
        # Use timeout to avoid penalty:
        df["posteam_timeouts_remaining"] = (
            df["posteam_timeouts_remaining"] - 1
        ).clip(lower=0)
        df["clock_stop"] = 1

    X_num_new = df[feature_cols_numeric].copy()
    X_cat_new = df[feature_cols_categorical].copy()

    X_cat_dummies_new = pd.get_dummies(X_cat_new, drop_first=True)

    X_new = pd.concat(
        [X_num_new.reset_index(drop=True), X_cat_dummies_new.reset_index(drop=True)],
        axis=1,
    )

    X_new = X_new.reindex(columns=full_feature_columns, fill_value=0)
    return X_new


def add_delay_vs_timeout_probs(
    decisions_df: pd.DataFrame,
    model: XGBRegressor,
    full_feature_columns=None,
) -> pd.DataFrame:
    """
    For each row, compute:
        - wp_penalty: accept delay-of-game
        - wp_timeout: use timeout to avoid penalty
        - dog_timeout_boost = wp_timeout - wp_penalty
    """
    if full_feature_columns is None:
        full_feature_columns = FEATURE_COLUMNS

    required_for_decisions = [
        "qtr",
        "game_seconds_remaining",
        "half_seconds_remaining",
        "score_differential",
        "wp",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "posteam",
        "defteam",
        "play_type",
        "clock_stop",
    ]

    missing_decision_cols = [c for c in required_for_decisions if c not in decisions_df.columns]
    if missing_decision_cols:
        raise ValueError(
            f"decisions_df is missing required columns: {missing_decision_cols}"
        )

    X_penalty = _build_design_matrix_delay_vs_timeout(
        decisions_df,
        accept_penalty=True,
        feature_cols_numeric=feature_cols_numeric,
        feature_cols_categorical=feature_cols_categorical,
        full_feature_columns=full_feature_columns,
    )

    X_timeout = _build_design_matrix_delay_vs_timeout(
        decisions_df,
        accept_penalty=False,
        feature_cols_numeric=feature_cols_numeric,
        feature_cols_categorical=feature_cols_categorical,
        full_feature_columns=full_feature_columns,
    )

    wp_penalty = model.predict(X_penalty)
    wp_timeout = model.predict(X_timeout)

    wp_penalty = np.clip(wp_penalty, 0, 1)
    wp_timeout = np.clip(wp_timeout, 0, 1)

    probs = decisions_df.copy()
    probs["wp_penalty"] = wp_penalty
    probs["wp_timeout"] = wp_timeout
    probs["dog_timeout_boost"] = probs["wp_timeout"] - probs["wp_penalty"]

    return probs


def make_delay_vs_timeout_table(probs: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table for delay-of-game vs timeout decisions.

    Recommendation based on dog_timeout_boost:
        - > 0.02   -> "Recommendation: Call a timeout (avoid penalty)"
        - <= 0.00  -> "Recommendation: Accept delay of game penalty"
        - (0,0.02] -> "Recommendation: Toss Up"
    """
    table_cols = [
        "posteam",
        "defteam",
        "qtr",
        "game_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "score_differential",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "wp_penalty",
        "wp_timeout",
        "dog_timeout_boost",
    ]

    existing = [c for c in table_cols if c in probs.columns]
    table = probs[existing].copy()

    def _recommend(boost: float) -> str:
        if boost > 0.02:
            return "Recommendation: Call a timeout (avoid penalty)"
        elif boost <= 0.0:
            return "Recommendation: Accept delay of game penalty"
        else:
            return "Recommendation: Toss Up"

    if "dog_timeout_boost" in table.columns:
        table["recommendation"] = table["dog_timeout_boost"].apply(_recommend)
        table = table.sort_values("dog_timeout_boost", ascending=False)

    return table

# =========================================================
# 3. Streamlit UI
# =========================================================

st.title("NFL Timeout Usage Model (nfl4th-style)")

st.markdown(
    """
This app uses a trained XGBoost win-probability model to:

- Compare **calling a timeout vs not**.
- Compare **using a timeout vs accepting a delay-of-game penalty**.

All recommendations are based on **changes in win probability**.
"""
)

st.sidebar.header("Game situation inputs")

# Basic inputs
posteam = st.sidebar.text_input("Offense team (posteam)", value="MIN")
defteam = st.sidebar.text_input("Defense team (defteam)", value="GB")

qtr = st.sidebar.selectbox("Quarter (qtr)", options=[1, 2, 3, 4], index=3)
minutes = st.sidebar.number_input("Minutes remaining in game", min_value=0, max_value=60, value=2)
seconds = st.sidebar.number_input("Seconds remaining (extra)", min_value=0, max_value=59, value=0)

game_seconds_remaining = minutes * 60 + seconds

half_minutes = st.sidebar.number_input(
    "Minutes remaining in half", min_value=0, max_value=30, value=2
)
half_seconds_extra = st.sidebar.number_input(
    "Seconds remaining in half (extra)", min_value=0, max_value=59, value=0
)
half_seconds_remaining = half_minutes * 60 + half_seconds_extra

score_diff = st.sidebar.number_input(
    "Score differential (posteam_score - defteam_score)", value=-3
)

down = st.sidebar.selectbox("Down", options=[1, 2, 3, 4], index=1)
ydstogo = st.sidebar.number_input("Yards to go", min_value=1, max_value=50, value=7)

yardline_100 = st.sidebar.number_input(
    "Yardline_100 (distance to opponent end zone)", min_value=1, max_value=99, value=40
)

posteam_timeouts = st.sidebar.number_input(
    "Offense timeouts remaining", min_value=0, max_value=3, value=3
)
defteam_timeouts = st.sidebar.number_input(
    "Defense timeouts remaining", min_value=0, max_value=3, value=3
)

play_type = st.sidebar.selectbox("Expected play type", options=["run", "pass", "qb_scramble"], index=1)

clock_running = st.sidebar.checkbox("Is game clock running?", value=True)
clock_stop = 0 if clock_running else 1

# You can either input a WP estimate or leave a default (for display only)
wp_display = st.sidebar.slider(
    "Current win probability estimate (for display only)",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.01,
)

# Build single-row decisions_df
decisions_df = pd.DataFrame(
    {
        "qtr": [qtr],
        "game_seconds_remaining": [game_seconds_remaining],
        "half_seconds_remaining": [half_seconds_remaining],
        "score_differential": [score_diff],
        "wp": [wp_display],
        "down": [down],
        "ydstogo": [ydstogo],
        "yardline_100": [yardline_100],
        "posteam_timeouts_remaining": [posteam_timeouts],
        "defteam_timeouts_remaining": [defteam_timeouts],
        "posteam": [posteam],
        "defteam": [defteam],
        "play_type": [play_type],
        "clock_stop": [clock_stop],
    }
)

st.subheader("Timeout vs No Timeout")

if st.button("Evaluate timeout decision"):
    probs_timeout = add_timeout_probs(decisions_df, xgb_model, FEATURE_COLUMNS)
    table_timeout = make_timeout_table(probs_timeout)

    st.write("**Timeout decision table** (win probabilities):")
    st.dataframe(table_timeout)

    # Show the recommendation prominently
    rec = table_timeout["recommendation"].iloc[0]
    boost = table_timeout["timeout_boost"].iloc[0]
    st.markdown(
        f"### {rec}  \n"
        f"_Estimated change in win probability: {boost:+.3f}_"
    )

st.markdown("---")
st.subheader("Delay of Game vs Timeout (Early-Half Scenario)")

st.markdown(
    "Use the same situation inputs and compare **accepting a delay-of-game penalty** vs **calling timeout**."
)

if st.button("Evaluate delay-of-game vs timeout"):
    probs_dog = add_delay_vs_timeout_probs(decisions_df, xgb_model, FEATURE_COLUMNS)
    table_dog = make_delay_vs_timeout_table(probs_dog)

    st.write("**Delay of game vs timeout table** (win probabilities):")
    st.dataframe(table_dog)

    rec2 = table_dog["recommendation"].iloc[0]
    boost2 = table_dog["dog_timeout_boost"].iloc[0]
    st.markdown(
        f"### {rec2}  \n"
        f"_Estimated change in win probability (timeout - penalty): {boost2:+.3f}_"
    )
