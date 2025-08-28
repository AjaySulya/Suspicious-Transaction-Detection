## comman functionalies which use by entire project
import os
import sys 
import dill

# For the exception handling  and logging
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
# function for the python object save
def save_object(file_path,obj): # file_path : where i want to save , object to save 
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=1)
        
        with open(file_path,'wb') as file_obj: # mention here this i open the file_path parameter as write binary and dump the obj .
            dill.dump(obj,file_obj)
            
    except Exception as e: # if there is exception then handle this 
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_test_pred = model.predict(X_test)
            
            test_model_score = accuracy_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)        