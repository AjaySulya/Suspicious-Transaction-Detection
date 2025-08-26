## comman functionalies which use by entire project
import os
import sys 
import dill

# For the exception handling  and logging
from src.exception import CustomException
from src.logger import logging

# function for the python object save
def save_object(file_path,obj): # file_path : where i want to save , object to save 
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=1)
        
        with open(file_path,'wb') as file_obj: # mention here this i open the file_path parameter as write binary and dump the obj .
            dill.dump(obj,file_obj)
            
    except Exception as e: # if there is exception then handle this 
        raise CustomException(e,sys)
    
    