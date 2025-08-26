# import some necessory libraries
import os
from dataclasses import dataclass # for management class and manipulated
import numpy as np
import pandas as pd
import sys
# customException and logging
from src.exception import CustomException
from src.logger import logging
import networkx as nx
# for saving the python object
from src.utils import save_object
# import the train and test dataset
from src.components import data_ingestion


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    preprocessed_obj_file_path = os.path.join("artifacts","preprocessor.pkl") # specify the path where to save the preprocessed data
    
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.train_set_path,self.test_set_path = data_ingestion() # In this two variable i store traina and test set path
    
    ## Step  feature 
    def add_time_feature(self,df :pd.DataFrame) -> pd.DataFrame:
        try:
           logging.info("adding the time time feature derivated from the step feature")
                # step in 1...744 keep raw plus derived 
           df['hour'] = df['step'] % 24 
           df['day'] = (df['step']//24).astype(int)
           df['dow'] = df['day'] % 7
           df['is_weekend']  = (df['dow'] >= 5).astype(int)
            
           return df.drop(columns=['day','is_weekend'],axis=1)     
        except Exception as e:
            raise CustomException(e,sys)          
        
               
        
    def add_delta_feature(self,df:pd.DataFrame)-> pd.DataFrame:
        try:
            logging.info("Adding the delta features")
            # add the delta feature
            df['deltaOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig'] # deltaOrg === amount
            df['deltaDest'] = df['oldbalanceDest'] - df['newbalanceDest'] # same 
            df['deltaOrg_diff'] = df['deltaOrg'] - df['amount'] # checking the difference 
            df['deltaDest_diff'] = df['deltaDest'] - df['amount'] 
            logging.info("Successfully added the delta  features")
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
    # graph features from training the graph only to avoid leakage    
    def _graph_maps_from_train(self,train_df:pd.DataFrame)->pd.DataFrame:
        try:
            logging.info("making the instance of graph,networdx from the train")
            G = nx.from_pandas_edgelist(train_df,
                                        source='nameOrg',
                                        target='nameDest',
                                        create_using=nx.DiGraph())
            out_deg = dict(G.out_degree()) # sender activaty
            in_deg = dict(G.in_degree()) # reciever activaty
            pr = nx.pagerank(G) if len(G) > 0 else {}
            return out_deg,in_deg,pr
        except Exception as e:
            raise CustomException(e,sys)
    def add_graph_features_with_map(self,df:pd.DataFrame,out_deg,in_deg,pr)->pd.DataFrame:
        try:
            logging.info("adding the graph features in dataframe")
            df['degree_org'] = df['nameOrg'].map(out_deg).fillna(0)
            df['degree_in_dest'] = df['nameDest'].map(in_deg).fillna(0)
            df['pagerank_org'] = df['nameOrg'].map(pr).fillna(0)
            df['pagerank_dest'] = df['nameDest'].map(pr).fillna(0)
            return df
        except Exception as e:
            raise CustomException(e,sys)    
    def get_data_transformer_object(self):
        """  this function return the object for preprocessor"""
        categorical_features = ['type']
        numerical_features = [
            "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
            "deltaOrg", "deltaDest","deltaOrg_diff","deltaDest_diff",
            "degree_org",  "pagerank_org",
            "degree_dest", "pagerank_dest"
        ]
        time_features = ["hour","dow"]
        

        #cyclical pipeline
        time_pipeline = Pipeline(steps=[
            ("cyclical",FunctionTransformer(cyclical_encode, validate=False)),
            ("scaler",StandardScaler())
        ])
        # pipeline for the numerical features
        num_pipeline = Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())
            ]
        )
        
        # categorical pipeline
        cat_pipeline = Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(drop='first',handle_unknown='ignore'))
        ])
        
        # here we will combine all of this 
        preprocessor = ColumnTransformer(
            transformers=[
                ("num",num_pipeline,numerical_features),
                ("cat",cat_pipeline,categorical_features),
                ("time",time_pipeline,time_features)
            ]
        )
        return preprocessor
        
    def initiate_data_transform(self,train_path:str,test_path:str):
        logging.info("Data Transformation start")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            for d in (train_df,test_df):
                if 'isFlaggedFraud' in d.columns:
                    d.drop(columns=['isFlaggedFraud'],inplace=True)
            logging.info("Read the train and test dataset as Dataframe")
            preprocessor_obj = self.get_data_transformer_object()
            # add the delta + time feature
            train_df = self.add_delta_feature(train_df)
            test_df = self.add_delta_feature(test_df)
            
            train_df = self.add_time_feature(train_df)
            test_df = self.add_time_feature(test_df)
            # 4) graph features (maps from TRAIN only)
            out_deg, in_deg, pr = self._graph_maps_from_train(train_df)
            train_df = self.add_graph_features_with_maps(train_df, out_deg, in_deg, pr)
            test_df  = self.add_graph_features_with_maps(test_df,  out_deg, in_deg, pr)
            
            # 6) drop identifiers
            id_cols = [c for c in ['nameOrig','nameDest','step'] if c in train_df.columns]
            train_df.drop(columns=id_cols, inplace=True, errors='ignore')
            test_df.drop(columns=id_cols,  inplace=True, errors='ignore')
            target_feature_name = 'isFraud'
            # input and target features for the train dataset
            input_feature_train_df = train_df.drop(columns=[target_feature_name],axis=1)
            target_feature_train_df = train_df[target_feature_name]
            # input and target for the test dataset
            input_feature_test_df = test_df.drop(columns=[target_feature_name],axis=1)
            target_feature_test_df = test_df[target_feature_name]
            
            logging.info("Apply the preprocessing object on the training dataframe and test dataframe")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = [
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("Saved Preprocessed onject")
            save_object(
                file_path=self.data_transformation_config.preprocessed_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessed_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__ == "__main__":
    try:
        logging.info("Starting Data Transformation module test")

        # Create object of DataTransformation
        data_transformation = DataTransformation()

        # Call the transformation
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transform(
            data_transformation.train_set_path,
            data_transformation.test_set_path
        )

        logging.info("Data Transformation completed successfully.")
        print("✅ Train array shape:", np.shape(train_arr))
        print("✅ Test array shape:", np.shape(test_arr))
        print("✅ Preprocessor object saved at:", preprocessor_path)

    except Exception as e:
        logging.exception("Error while testing Data Transformation")
        print("❌ Error:", e)
              