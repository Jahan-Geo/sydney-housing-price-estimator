"""Shared feature-engineering transformer.

Defined once here and imported by both this notebook (Section 3.1 onward) and
app.py, so the training pipeline and the deployed app always use an identical
class -- previously the same class was copy-pasted into both places, which is
fragile (a class fixed in one copy but not the other would silently produce a
model/app mismatch, and joblib unpickling depends on the class living in a
module of this exact name at load time).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Turns the raw collected columns into the modelling-ready feature block.

    This is the leakage-safe replacement for manual, pre-split feature engineering.
    fit() learns three group-median statistics (land size by suburb x property type,
    floor area by property type, and a bathrooms median) from whatever rows it is
    given; transform() applies those learned statistics to impute missing values and
    derive total_rooms, bed_bath_ratio, has_car_space, is_unit, and the three
    missingness flags.

    Because this lives as the FIRST step of the modelling pipeline, scikit-learn
    refits it automatically on every cross-validation fold's training partition,
    on X_train alone before the held-out test check, and on the full dataset only
    at final deployment (Section 6.2). No median used to impute a row is ever
    computed using a row that is being predicted at that stage.
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
