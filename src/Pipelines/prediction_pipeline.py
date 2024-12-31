import os
import sys
import pandas as pd

from pathlib import Path

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class predict_pipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            preprocessor_path=os.path.join('Artifacts', 'preprocessor.pkl')
            model_path=os.path.join('Artifacts', 'model.pkl')
            
            model=load_object(filepath=model_path)
            preprocessor=load_object(filepath=preprocessor_path)
            
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            
            return pred
        except Exception as e:
            logging.info("Exception occured while prediction pipeline")
            
            
 
       
class CustomData:
    def __init__(self,
                 sex:str,
                 bmi:float,
                 children:int,
                 smoker:str,
                 region:str):
        self.sex=sex
        self.bmi=bmi
        self.children=children
        self.smoker=smoker
        self.region=region
        
    def get_data_as_a_fram(self):
        try:
            custom_data_input_dict={
                'sex':[self.sex],
                'bmi':[self.bmi],
                'children':[self.children],
                'smoker':[self.smoker],
                'region':[self.region]
            }
            df=pd.DataFrame(custom_data_input_dict)
            logging.info("Data Frame gathered!!")
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    