import os 
import sys

from src.logger import logging
from src.exception import CustomException

from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()
    data_transformtion=DataTransformation()
    train_arr, test_arr=data_transformtion.initiate_data_transformation(train_data, test_data) 
    model_trainer=ModelTrainer()
    model_trainer.initate_model_trainer(train_array=train_arr, test_array=test_arr) 