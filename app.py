import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Sydney Housing Price Estimator", page_icon="🏠", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "sydney_price_model.joblib"))
FEATURES_NUM = joblib.load(os.path.join(BASE_DIR, "features_num.joblib"))
FEATURES_CAT = joblib.load(os.path.join(BASE_DIR, "features_cat.joblib"))
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

st.title("🏠 Sydney Housing Price Estimator")
st.caption(
    "A decision-support prototype trained on 179 sold properties across Surry Hills, "
    "Strathfield, and Campbelltown. Enter a property's details to get an indicative "
    "sale price estimate (Gradient Boosting regression model, Part 3 of this project)."
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
    has_land_size = st.checkbox("Land size known?", value=(property_type == "House"))
    land_size = st.number_input("Land size (sqm)", min_value=0.0, max_value=3000.0, value=450.0, step=10.0,
                                 disabled=not has_land_size)
    has_floor_area = st.checkbox("Floor area known?", value=(property_type != "House"))
    floor_area = st.number_input("Floor area (sqm)", min_value=0.0, max_value=1000.0, value=90.0, step=5.0,
                                  disabled=not has_floor_area)
    sale_quarter = st.selectbox("Expected sale quarter", [1, 2, 3, 4], index=2)

st.divider()

if st.button("Estimate sale price", use_container_width=True):
    row = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "car_spaces_filled": car_spaces,
        "land_size_filled": land_size if has_land_size else 0.0,
        "floor_area_filled": floor_area if has_floor_area else np.nan,
        "total_rooms": bedrooms + bathrooms,
        "bed_bath_ratio": bedrooms / bathrooms if bathrooms > 0 else bedrooms,
        "has_car_space": int(car_spaces > 0),
        "is_unit": int(property_type == "Unit"),
        "land_size_missing": int(not has_land_size),
        "floor_area_missing": int(not has_floor_area),
        "car_spaces_missing": 0,
        "sale_quarter": sale_quarter,
        "suburb": suburb,
        "property_type": property_type,
    }
    X_input = pd.DataFrame([row])[FEATURES_NUM + FEATURES_CAT]

    log_pred = model.predict(X_input)[0]
    pred_price = float(np.exp(log_pred))

    st.markdown(
        f"""
        <div class="result-box">
            <div style="font-size:0.95rem; color:#555;">Estimated sale price</div>
            <div style="font-size:2.2rem; font-weight:700; color:#1B2A4A;">${pred_price:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        typical = ref_medians.loc[(suburb, property_type)]
        st.caption(f"For comparison, the median sale price for {property_type}s in {suburb} in the training data was ${typical:,.0f}.")
    except KeyError:
        st.caption(f"No comparable {property_type} sales for {suburb} in the training data — treat this estimate with extra caution.")

    st.warning(
        "⚠️ This is a prototype estimate from a model trained on only 179 sold properties. "
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
        """
    )
