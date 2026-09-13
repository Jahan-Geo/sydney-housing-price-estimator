import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Sydney Housing Price Estimator", page_icon="\U0001F3E0", layout="centered")

# ---------------------------------------------------------------------------
# The saved pipeline is a pickled scikit-learn object whose first step is a
# HousingFeatureEngineer instance. Unpickling requires this class to be
# importable from a module of the same name/path it was pickled from, so
# housing_transformer.py (Section 3.1a) must be deployed alongside app.py in
# the same folder -- see the deployment checklist in Section 6.4.
# ---------------------------------------------------------------------------
from housing_transformer import HousingFeatureEngineer  # noqa: F401 (needed for unpickling)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "sydney_price_model.joblib"))
# Quantile models for the 80% prediction interval
model_p10 = joblib.load(os.path.join(BASE_DIR, "sydney_price_model_p10.joblib"))
model_p90 = joblib.load(os.path.join(BASE_DIR, "sydney_price_model_p90.joblib"))
RAW_INPUT_COLS = joblib.load(os.path.join(BASE_DIR, "raw_input_cols.joblib"))
ref_medians = joblib.load(os.path.join(BASE_DIR, "ref_medians.joblib"))

st.markdown(
    """
    <style>
    .main { background-color: #FAFBFC; }
    h1 { color: #1B2A4A; }
    .stButton>button {
        background-color: #2E86AB; color: white; font-weight: 600;
        border-radius: 6px; padding: 0.6rem 1.2rem; border: none;
    }
    .result-box {
        background-color: #EAF2F8; border-left: 6px solid #2E86AB;
        padding: 1.2rem 1.5rem; border-radius: 8px; margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("\U0001F3E0 Sydney Housing Price Estimator")
st.caption(
    "A decision-support prototype trained on 179 sold properties across Surry Hills, "
    "Strathfield, and Campbelltown. Enter a property's details to get an indicative "
    "sale price estimate and an indicative prediction range (Gradient Boosting regression model, "
    "hyperparameters selected via grid search, Part 3 of this project)."
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    suburb = st.selectbox("Suburb", ["Surry Hills", "Strathfield", "Campbelltown"])
    property_type = st.selectbox("Property type", ["House", "Unit", "Townhouse", "Semi/Terrace"])
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)
    car_spaces = st.number_input("Car spaces", min_value=0, max_value=10, value=1, step=1)
with col2:
    # Keying these on property_type forces Streamlit to treat the checkbox as a
    # brand-new widget whenever the property type changes, so the "known?"
    # default (True for houses re: land, True for units re: floor area) is
    # re-applied instead of being stuck on whatever the user last clicked for
    # a *different* property type. Without the key, Streamlit reuses the
    # widget's prior on-screen state and silently ignores the `value=` default
    # after the first interaction.
    has_land_size = st.checkbox("Land size known?", value=(property_type == "House"),
                                 key=f"land_known_{property_type}")
    land_size = st.number_input("Land size (sqm)", min_value=0.0, max_value=3000.0, value=450.0, step=10.0,
                                 disabled=not has_land_size)
    has_floor_area = st.checkbox("Floor area known?", value=(property_type != "House"),
                                  key=f"floor_known_{property_type}")
    floor_area = st.number_input("Floor area (sqm)", min_value=0.0, max_value=1000.0, value=90.0, step=5.0,
                                  disabled=not has_floor_area)
    sale_quarter = st.selectbox("Expected sale quarter", [1, 2, 3, 4], index=2)

st.divider()

if st.button("Estimate sale price", use_container_width=True):
    row = {
        "suburb": suburb,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "car_spaces": car_spaces,
        "land_size_sqm": land_size if has_land_size else np.nan,
        "floor_area_sqm": floor_area if has_floor_area else np.nan,
        "sale_quarter": sale_quarter,
    }
    X_input = pd.DataFrame([row])[RAW_INPUT_COLS]
    log_pred = model.predict(X_input)[0]
    pred_price = float(np.exp(log_pred))

    # Compute the 80% interval alongside the point estimate
    log_pred_low = model_p10.predict(X_input)[0]
    log_pred_high = model_p90.predict(X_input)[0]
    pred_low = float(np.exp(log_pred_low))
    pred_high = float(np.exp(log_pred_high))
    # Guard against quantile crossing (rare, but possible with independently fit models)
    pred_low, pred_high = min(pred_low, pred_high), max(pred_low, pred_high)

    st.markdown(
        f"""
        <div class="result-box">
            <div style="font-size:0.95rem; color:#555;">Estimated sale price</div>
            <div style="font-size:2.2rem; font-weight:700; color:#1B2A4A;">${pred_price:,.0f}</div>
            <div style="font-size:0.9rem; color:#555; margin-top:0.4rem;">
                Indicative prediction range (10th–90th percentile): ${pred_low:,.0f} &ndash; ${pred_high:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        typical = ref_medians.loc[(suburb, property_type)]
        st.caption(f"For comparison, the median sale price for {property_type}s in {suburb} in the training data was ${typical:,.0f}.")
    except KeyError:
        st.caption(f"No comparable {property_type} sales for {suburb} in the training data -- treat this estimate with extra caution.")
    st.warning(
        "\u26a0\ufe0f This is a prototype estimate from a model trained on only 179 sold properties. "
        "It does not account for property condition, renovations, aspect, views, or unique "
        "features, and should not be used as a substitute for a professional valuation."
    )

st.divider()
with st.expander("About this tool"):
    st.write(
        """
        This app wraps a Gradient Boosting regression model trained as part of a Sydney
        housing price prediction mini-project. The model was trained on manually collected
        sold-property data from domain.com.au and realestate.com.au across three suburbs
        (Surry Hills, Strathfield, Campbelltown), covering 60 sold properties per suburb.

        **Known limitations** (see Part 4 of the accompanying report/notebook for detail):
        - No information on property condition, renovation status, or age.
        - Land size / floor area were missing for a large share of listings and were
          imputed; predictions for atypical properties should be treated cautiously.
        - The training data spans roughly 11 months, so the model cannot account for
          market movements outside that window.
        - The "likely range" shown is an empirical 10th-90th percentile band from two
          quantile-loss Gradient Boosting models, not a statistically guaranteed
          confidence interval -- treat it as indicative, especially for high-value
          Strathfield houses where Part 4 found the largest prediction errors.
        """
    )
