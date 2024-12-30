import os
import sys
from pathlib import Path
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('Artifacts', "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting the data into training and testing")
            
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
            )
            models={
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'SVR': SVR(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'Ada Boost Regressor': AdaBoostRegressor(),
                'KNeigbors Regressor': KNeighborsRegressor()
            }
            
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            print(model_report)
            
            print("\n================================================================\n")
            
            logging.info(f"Model Report : {model_report}")
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            print(f"Best Model Found, Model Name : {best_model_name} and R2 Score : {best_model_score}")
            
            print('\n==================================================================\n')
            
            logging.info(f"Best Model Found, Model Name : {best_model_name} and R2 Score : {best_model_score}")
            
            
            
            
            logging.info("Best Model Found in both training and testing data")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logging.info("Exception occured while model training")
            raise CustomException(e,sys)