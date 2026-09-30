"""Rossmann promotion success predictor.

Pick a store and a date, and the app estimates the chance that running a
promotion there on that day lifts sales by at least 30% over the store's normal
non-promo sales on the same weekday.

Run locally:  streamlit run app.py
"""
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("sales_model.sav")

CATEGORICAL = ["DayOfWeek", "Month", "StateHoliday", "StoreType", "Assortment"]
NUMERIC_CLF = ["SchoolHoliday", "Promo2", "IsPromo2Month", "LogCompDist",
               "CompOpenMonths", "Day", "LogStoreAvgSales"]
NUMERIC_LIN = ["Promo"] + NUMERIC_CLF

STATE_HOLIDAYS = {
    "None": "0",
    "Public holiday": "a",
    "Easter": "b",
    "Christmas": "c",
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------- features --
def is_promo2_month(row, d):
    """1 if the store's Promo2 is running in the month of date d, else 0."""
    if row["Promo2"] != 1 or pd.isna(row["PromoInterval"]):
        return 0
    months = [m.strip().replace("Sept", "Sep") for m in str(row["PromoInterval"]).split(",")]
    if d.strftime("%b") not in months:
        return 0
    if pd.isna(row["Promo2SinceYear"]) or pd.isna(row["Promo2SinceWeek"]):
        return 0
    try:
        start = datetime.strptime(
            f"{int(row['Promo2SinceYear'])}-W{int(row['Promo2SinceWeek']):02d}-1", "%Y-W%W-%w")
    except ValueError:
        return 0
    return int(datetime(d.year, d.month, d.day) >= start)


def comp_open_months(row, d):
    """Months since the nearest competitor opened (0 if unknown or in the future)."""
    y, m = row["CompetitionOpenSinceYear"], row["CompetitionOpenSinceMonth"]
    if pd.isna(y) or pd.isna(m) or int(y) <= 0 or not (1 <= int(m) <= 12):
        return 0
    return max((d.year - int(y)) * 12 + (d.month - int(m)), 0)


def build_raw(store_row, d, school_holiday, state_holiday, promo):
    """Raw (unencoded) feature values for one store-day."""
    return {
        "Promo": promo,
        "SchoolHoliday": int(school_holiday),
        "Promo2": int(store_row["Promo2"]),
        "IsPromo2Month": is_promo2_month(store_row, d),
        "LogCompDist": float(np.log1p(store_row["CompetitionDistance"])),
        "CompOpenMonths": comp_open_months(store_row, d),
        "Day": d.day,
        "LogStoreAvgSales": float(np.log(store_row["StoreAvgSales"])),
        "DayOfWeek": d.isoweekday(),          # Monday=1 ... Sunday=7
        "Month": d.month,
        "StateHoliday": state_holiday,
        "StoreType": store_row["StoreType"],
        "Assortment": store_row["Assortment"],
    }


def encode(raw, numeric, columns):
    """One-hot encode a single row and align it to the training columns.

    drop_first is not used here on purpose: with one row it would drop the only
    category present. Reindexing to the training columns drops the same
    baseline levels that drop_first removed during training.
    """
    num = pd.DataFrame([{k: raw[k] for k in numeric}])
    cat = pd.DataFrame([{k: str(raw[k]) for k in CATEGORICAL}])
    X = pd.concat([num, pd.get_dummies(cat)], axis=1)
    return X.reindex(columns=columns, fill_value=0).astype(float)


def predict_success(bundle, raw):
    """Probability that a promo on this store-day reaches the lift target."""
    X = encode(raw, NUMERIC_CLF, bundle["clf_feature_columns"])
    Xs = bundle["clf_scaler"].transform(X)
    return float(bundle["log_reg"].predict_proba(Xs)[0, 1])


def predict_sales(bundle, raw, promo):
    """Linear-model sales estimate with the promo switched on or off."""
    raw = {**raw, "Promo": promo}
    X = encode(raw, NUMERIC_LIN, bundle["feature_columns"])
    Xs = bundle["scaler"].transform(X)
    return float(np.expm1(bundle["lin_reg"].predict(Xs))[0])


# --------------------------------------------------------------------- app --
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="Promo success predictor", page_icon="📈")
    st.title("Will a promotion succeed?")
    st.write(
        "Estimate whether running a promotion at a Rossmann store on a given day "
        "will lift sales by at least **30%** over that store's normal non-promo "
        "sales on the same weekday."
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}. Put it next to app.py.")
        st.stop()

    bundle = load_bundle()
    stores = bundle["store_info"].set_index("Store")
    baseline = bundle["baseline"].set_index(["Store", "DayOfWeek"])["NormalSales"]
    lift = bundle["lift_threshold"]

    # ---- inputs
    col1, col2 = st.columns(2)
    with col1:
        store_id = st.selectbox("Store", stores.index.tolist())
        d = st.date_input(
            "Date", value=date(2015, 8, 3),
            min_value=date(2013, 1, 1), max_value=date(2016, 12, 31),
            help="The model was trained on data from 2013 to mid-2015.",
        )
    with col2:
        holiday_label = st.selectbox("State holiday", list(STATE_HOLIDAYS))
        school_holiday = st.checkbox("School holiday")
        threshold = st.slider(
            "Decision threshold", 0.30, 0.80, 0.50, 0.05,
            help="Call the promo a success when the estimated chance is at least this high.",
        )

    store = stores.loc[store_id]
    with st.expander("Store details"):
        st.write(
            f"Type **{store['StoreType']}**, assortment **{store['Assortment']}**, "
            f"nearest competitor **{store['CompetitionDistance']:,.0f} m** away, "
            f"Promo2 {'on' if store['Promo2'] == 1 else 'off'}."
        )

    # ---- prediction
    dow = d.isoweekday()
    normal = baseline.get((store_id, dow), np.nan)
    if np.isnan(normal):
        st.warning(
            f"Store {store_id} has no non-promo sales history on a {WEEKDAYS[dow - 1]}, "
            "so there's no baseline to measure a 30% lift against. Try another date."
        )
        st.stop()

    raw = build_raw(store, d, school_holiday, STATE_HOLIDAYS[holiday_label], promo=1)
    prob = predict_success(bundle, raw)

    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Chance of success", f"{prob:.0%}")
    m2.metric(f"Normal {WEEKDAYS[dow - 1]} sales", f"{normal:,.0f}")
    m3.metric(f"Needed for +{(lift - 1):.0%}", f"{normal * lift:,.0f}")

    if prob >= threshold:
        st.success("Likely to succeed. This store-day looks like a good one for a promotion.")
    else:
        st.error("Unlikely to reach the lift target on this day.")

    with st.expander("Reference: linear model sales estimate"):
        with_promo = predict_sales(bundle, raw, promo=1)
        without_promo = predict_sales(bundle, raw, promo=0)
        c1, c2 = st.columns(2)
        c1.metric("Predicted sales with promo", f"{with_promo:,.0f}")
        c2.metric("Predicted sales without promo", f"{without_promo:,.0f}")
        st.caption("A separate model, shown for context only. It is not used for the success call above.")

    with st.expander("About this model"):
        st.write(
            "Logistic regression trained on promo days only (2013 to mid-June 2015) and "
            "tested on the last six weeks of data (about 71% accuracy, ROC-AUC about 0.71). "
            "\"Normal sales\" is the store's average non-promo sales on the same weekday. "
            "Treat the probability as a guide, not a guarantee."
        )


if __name__ == "__main__":
    main()
