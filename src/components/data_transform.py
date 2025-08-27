import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys
import networkx as nx

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components import data_ingestion


@dataclass
class DataTransformationConfig:
    preprocessed_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.train_set_path, self.test_set_path = data_ingestion.DataIngestion().initiate_data_ingestion()

    # Step: cyclical encoding
    @staticmethod
    def cyclical_encode(series, max_val):
        try:
            series = series.values  # ensure numpy
            sin_vals = np.sin(2 * np.pi * series / max_val)
            cos_vals = np.cos(2 * np.pi * series / max_val)
            return np.c_[sin_vals, cos_vals]  # always returns 2D numpy array
        except Exception as e:
            raise CustomException(e, sys)

    # Add delta features
    def add_delta_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['deltaOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig']
            df['deltaDest'] = df['oldbalanceDest'] - df['newbalanceDest']
            df['deltaOrg_diff'] = df['deltaOrg'] - df['amount']
            df['deltaDest_diff'] = df['deltaDest'] - df['amount']
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # Add time features
    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['hour'] = df['step'] % 24
            df['dow'] = (df['step'] // 24) % 7
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # Graph features
    def _graph_maps_from_train(self, train_df: pd.DataFrame):
        try:
            G = nx.from_pandas_edgelist(train_df, source='nameOrig', target='nameDest', create_using=nx.DiGraph())
            out_deg = dict(G.out_degree())
            in_deg = dict(G.in_degree())
            pr = nx.pagerank(G) if len(G) > 0 else {}
            return out_deg, in_deg, pr
        except Exception as e:
            raise CustomException(e, sys)

    def add_graph_features_with_map(self, df: pd.DataFrame, out_deg, in_deg, pr) -> pd.DataFrame:
        try:
            df['degree_org'] = df['nameOrig'].map(out_deg).fillna(0)
            df['degree_in_dest'] = df['nameDest'].map(in_deg).fillna(0)
            df['pagerank_org'] = df['nameOrig'].map(pr).fillna(0)
            df['pagerank_dest'] = df['nameDest'].map(pr).fillna(0)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # Preprocessor object
    def get_data_transformer_object(self):
        categorical_features = ['type']
        numerical_features = [
            "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
            "deltaOrg", "deltaDest", "deltaOrg_diff", "deltaDest_diff",
            "degree_org", "pagerank_org",
            "degree_in_dest", "pagerank_dest"
        ]
        time_features = ["hour", "dow"]

        time_pipeline = Pipeline(steps=[
            ("cyclical", FunctionTransformer(lambda x: np.c_[np.sin(2*np.pi*x/np.array([24 if col=="hour" else 7 for col in time_features])), 
                                                              np.cos(2*np.pi*x/np.array([24 if col=="hour" else 7 for col in time_features]))],
                                             validate=False)),
            ("scaler", StandardScaler())
        ])

        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features),
                ("time", time_pipeline, time_features)
            ],
            remainder='drop'
        )
        return preprocessor

    # Main transformation
    def initiate_data_transform(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Drop flagged fraud if exists
            for d in (train_df, test_df):
                if 'isFlaggedFraud' in d.columns:
                    d.drop(columns=['isFlaggedFraud'], inplace=True)

            # Add delta + time features
            train_df = self.add_delta_feature(train_df)
            test_df = self.add_delta_feature(test_df)
            train_df = self.add_time_features(train_df)
            test_df = self.add_time_features(test_df)

            # Graph features
            out_deg, in_deg, pr = self._graph_maps_from_train(train_df)
            train_df = self.add_graph_features_with_map(train_df, out_deg, in_deg, pr)
            test_df = self.add_graph_features_with_map(test_df, out_deg, in_deg, pr)

            # Drop identifiers
            id_cols = [c for c in ['nameOrig', 'nameDest', 'step'] if c in train_df.columns]
            train_df.drop(columns=id_cols, inplace=True, errors='ignore')
            test_df.drop(columns=id_cols, inplace=True, errors='ignore')

            target_feature_name = 'isFraud'  # your actual target
            X_train = train_df.drop(columns=[target_feature_name], axis=1)
            y_train = train_df[target_feature_name].values.ravel()
            X_test = test_df.drop(columns=[target_feature_name], axis=1)
            y_test = test_df[target_feature_name].values.ravel()

            preprocessor_obj = self.get_data_transformer_object()
            X_train_arr = preprocessor_obj.fit_transform(X_train)
            X_test_arr = preprocessor_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train]
            test_arr = np.c_[X_test_arr, y_test]

            save_object(file_path=self.data_transformation_config.preprocessed_obj_file_path, obj=preprocessor_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessed_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logging.info("Starting Data Transformation module test")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transform(
            data_transformation.train_set_path,
            data_transformation.test_set_path
        )
        print("✅ Train shape:", train_arr.shape)
        print("✅ Test shape:", test_arr.shape)
        print("✅ Preprocessor saved at:", preprocessor_path)
    except Exception as e:
        logging.exception("Error while testing Data Transformation")
        print("❌ Error:", e)
