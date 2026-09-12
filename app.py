import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Sydney Housing Price Estimator", page_icon="🏠", layout="centered")

# ---------------------------------------------------------------------------
# HousingFeatureEngineer must be defined here, in app.py's own module namespace,
# because sydney_price_model.joblib is a pickled scikit-learn Pipeline whose first
# step is an instance of this exact class. Pickle/joblib do not store class code,
# only a reference to "the class named HousingFeatureEngineer in module X" -- so
# unpickling fails with an AttributeError unless that class already exists, under
# the same name, in whatever module is doing the loading. This must stay byte-for-
# byte identical to the class in the notebook's Section 3.1 (same OUTPUT_NUM /
# OUTPUT_CAT, same fit()/transform() logic), or predictions from this app will not
# match the notebook's results.
# ---------------------------------------------------------------------------
class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Turns the raw collected columns into the modelling-ready feature block.

    fit() learns three group-median statistics (land size by suburb x property type,
    floor area by property type, and a bathrooms median) from whatever rows it is
    given; transform() applies those learned statistics to impute missing values and
    derive total_rooms, bed_bath_ratio, has_car_space, is_unit, and the three
    missingness flags.
    """

    OUTPUT_NUM = ['bedrooms', 'bathrooms', 'car_spaces_filled', 'land_size_filled',
                  'floor_area_filled', 'total_rooms', 'bed_bath_ratio', 'has_car_space',
                  'is_unit', 'land_size_missing', 'floor_area_missing',
                  'car_spaces_missing', 'sale_quarter']
    OUTPUT_CAT = ['suburb', 'property_type']

    def fit(self, X, y=None):
        df = X.copy()
        self.land_medians_ = df.groupby(['suburb', 'property_type'])['land_size_sqm'].median()
        self.floor_medians_ = df.groupby('property_type')['floor_area_sqm'].median()
        self.floor_median_global_ = df['floor_area_sqm'].median()
        self.bathrooms_median_ = df['bathrooms'].median()
        return self

    def _impute_land(self, row):
        if pd.notna(row['land_size_sqm']):
            return row['land_size_sqm']
        key = (row['suburb'], row['property_type'])
        val = self.land_medians_.get(key, np.nan)
        return val if pd.notna(val) else 0.0  # e.g. units -> land not applicable

    def _impute_floor(self, row):
        if pd.notna(row['floor_area_sqm']):
            return row['floor_area_sqm']
        val = self.floor_medians_.get(row['property_type'], np.nan)
        return val if pd.notna(val) else self.floor_median_global_

    def transform(self, X):
        df = X.copy()
        out = pd.DataFrame(index=df.index)
        out['land_size_missing'] = df['land_size_sqm'].isna().astype(int)
        out['floor_area_missing'] = df['floor_area_sqm'].isna().astype(int)
        out['car_spaces_missing'] = df['car_spaces'].isna().astype(int)
        out['car_spaces_filled'] = df['car_spaces'].fillna(0)
        out['land_size_filled'] = df.apply(self._impute_land, axis=1)
        out['floor_area_filled'] = df.apply(self._impute_floor, axis=1)

        bathrooms_filled = df['bathrooms'].fillna(self.bathrooms_median_)
        out['bedrooms'] = df['bedrooms']
        out['bathrooms'] = df['bathrooms']  # raw; any remaining NaNs are handled by the
                                             # SimpleImputer inside the ColumnTransformer below
        out['total_rooms'] = df['bedrooms'] + bathrooms_filled
        out['bed_bath_ratio'] = df['bedrooms'] / bathrooms_filled.replace(0, 1)
        out['has_car_space'] = (out['car_spaces_filled'] > 0).astype(int)
        out['is_unit'] = (df['property_type'] == 'Unit').astype(int)
        out['sale_quarter'] = df['sale_quarter']
        out['suburb'] = df['suburb']
        out['property_type'] = df['property_type']
        return out[self.OUTPUT_NUM + self.OUTPUT_CAT]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "sydney_price_model.joblib"))
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
        st.caption(f"No comparable {property_type} sales for {suburb} in the training data -- treat this estimate with extra caution.")
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
