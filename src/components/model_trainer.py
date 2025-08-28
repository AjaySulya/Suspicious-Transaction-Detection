import os
import sys
from dataclasses import dataclass

# algorithms for training
from sklearn.tree import DecisionTreeClassifier
# ensemble methods
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from  xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from src.components import data_transform 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_set,test_set):
        try:
            logging.info("split the data into dependent and independent features ")
            X_train,y_train,X_test,y_test = (
                train_set[:,:-1],
                train_set[:,-1],
                test_set[:,:-1],
                test_set[:,-1]
            )
            models = {
              
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "XGBClassifier": XGBClassifier()
              
            }
            parameters={
                
                "Random Forest":{
                    'n_estimators':[50,100,200],
                    'criterion':['gini','entropy'],
                    'max_depth':[3,5,10,None],
                    'min_samples_split':[2,5,10],
                    'min_samples_leaf':[1,2,4]
                },
                "Gradient Boosting":{
                    'learning_rate':[0.01,0.1,0.2],
                    'n_estimators':[50,100,200],
                    'subsample':[0.6,0.8,1.0],
                    'max_depth':[3,5,10]
                },
                "AdaBoost":{
                    'n_estimators':[50,100,200],
                    'learning_rate':[0.01,0.1,0.2]
                },
                "XGBClassifier":{
                    'learning_rate':[0.01,0.1,0.2],
                    'n_estimators':[50,100,200],
                    'subsample':[0.6,0.8,1.0],
                    'max_depth':[3,5,10]
                }
             
            }
            
            model_report:dict = evaluate_models(X_train = X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test,
                                                models=models,
                                                params = parameters)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best model found , model name is :{best_model_name} .")
            if best_model_score < 0.6 :
                raise CustomException("No best Model found")
            logging.info("Best model found on both training and test dataset")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model)    
            predicted = best_model.predict(X_test) 
            accuracy_score = accuracy_score(y_test,predicted) 
            f1_score_ = f1_score(y_test,predicted)
            return accuracy_score , f1_score_  
        except Exception as e:
            raise CustomException(e, sys)



from src.components.model_trainer import ModelTrainer
from src.components.data_transform import DataTransformation

if __name__ == "__main__":
    try:
        logging.info("Starting Model Trainer module test")

        # First run DataTransformation to get processed arrays
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transform(
            data_transformation.train_set_path,
            data_transformation.test_set_path
        )

        # Train the model
        model_trainer = ModelTrainer()
        acc, f1 = model_trainer.initiate_model_trainer(train_arr, test_arr)

        print("✅ Accuracy:", acc)
        print("✅ F1 Score:", f1)
        print("✅ Model saved at:", model_trainer.model_trainer_config.trained_model_file_path)

    except Exception as e:
        logging.exception("Error while testing Model Trainer")
        print("❌ Error:", e)