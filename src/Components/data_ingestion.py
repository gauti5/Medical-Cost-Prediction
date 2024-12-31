import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from dataclasses import dataclass
sys.path.append('/Users/siddanthapusandeep/Medical-Cost-Prediction')

import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

from src.Components.data_transformation import Data_Transformation_Config, DataTransformation
from src.Components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join('Artifacts',"raw_data.csv")
    train_data_path=os.path.join('Artifacts', "train_data.csv")
    test_data_path=os.path.join('Artifacts', "test_data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started!!")
        try:
            data=pd.read_csv('Notebook/Data/data.csv')
            logging.info("Reading the data from CSV File")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Splitting the data into train and test')
            train_data, test_data=train_test_split(data, test_size=0.3, random_state=23)
            
            logging.info('Data Splitting Done!!')
            
            train_data.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, header=True, index=False)
            
            logging.info("Training and Testing Data Completed!!")
            logging.info("Data Ingestion Completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Error occured while data ingestion")
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()
    data_transformtion=DataTransformation()
    train_arr, test_arr=data_transformtion.initiate_data_transformation(train_data, test_data) 
    model_trainer=ModelTrainer()
    print(model_trainer.initate_model_trainer(train_array=train_arr, test_array=test_arr))  
            
            
            
            
    