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
    def __init__(self):
        self.ingestion_config = DataIngestionCpnfig() # here ingestion_config contain the 3 path of the above specify
    
    def initiate_data_ingestion(self):
        # generate logging message
        logging.info("Enter the data ingestion method or compenent")
        # use try for the handle the exception okkay  , using this i can prevent the python programm to crash.
        try:
            df = pd.read_csv('Data\Raw.csv') # read the dataset
            logging.info("Read the data as DataFrame")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 
            # for the save the train and test data 
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42) #split the training and testing data not dependent and independent okya
            
        except Exception as e:
            raise CustomException(e,sys) 
            