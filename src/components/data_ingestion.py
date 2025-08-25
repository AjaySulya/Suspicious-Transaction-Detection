# Some basic libraries for the 
from multiprocessing.util import info
import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

# model selection for split the training and testing dataset
from sklearn.model_selection import train_test_split

# dataclasses for the create the class for data management and manipulation 
from dataclasses import dataclass

@dataclass
class DataIngestionCpnfig:
    # In this class we specify path for storing the dataset after the sliting and making this centric to access an modifieds if required
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","raw.csv")
    
class DataIngestion:
    
    """ In this class we will take the path of the dataingestionconfig above we specify,
    seconf method for the data reading and spliting the dataset from the train and test .
    then we will return the there path for the use throughout the project , like data transform .
    """
    
    def __init__(self):
        self.ingestion_config = DataIngestionCpnfig() # here ingestion_config contain the 3 path of the above specify
    
    def initiate_data_ingestion(self,time_based_split:bool=True,test_frac:bool=0.3):
        # generate logging message
        logging.info("Enter the data ingestion method or compenent")
        # use try for the handle the exception okkay  , using this i can prevent the python programm to crash.
        try:
            df = pd.read_csv('Data\Raw.csv') # read the dataset
            logging.info("Read the data as DataFrame")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            # 
            if time_based_split and 'step' in df.columns:
                df = df.sort_values('step')
                steps = sorted(df['step'].unique())
                cutoff_idx = max(1,int((1-test_frac)*len(steps))-1)
                cutoff_step = steps[cutoff_idx]
                
                train_set = df[df['step'] <= cutoff_step]
                test_set = df[df['step'] > cutoff_step]
                
                # for the safety if extreamly skewed , then fallback to the stratified split
                if len(test_set) == 0 or len(train_set) == 0:
                    logging.info("split the dataset for test and train set is 0")
                    train_set,test_set = train_test_split(df,test_size=test_frac,random_state=42,
                                                          stratify=df['isFraud'] if 'isFraud' in df.columns else None)
                    
            else:
                logging.info("spliting the dataset for train and tests normal case")
                train_set,test_set = train_test_split(df,test_size=test_frac,random_state=42,
                                                      stratify=df['idFraud'] if 'isFraud' in df.columns else None)
            # for the save the train and test data 
            
            logging.info("Split the traing and test set.")
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)
            logging.info("Ingestion of data Completed.")
            
            # return the path where the dataset i have save
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            ) 
            
            
        except Exception as e:
            raise CustomException(e,sys) 
            