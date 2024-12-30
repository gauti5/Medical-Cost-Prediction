import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class Data_Transformation_Config:
    preprocessor_file_path=os.path.join('Artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=Data_Transformation_Config()
        
    def get_data_transformation(self):
        try:
            
            logging.info("Data Transformation Started!!")
            
            numerical_columns=['age', 'bmi', 'children']
            categ_columns=['sex', 'smoker', 'region']
            
            sex_categories=['male' 'female']
            smoker_categories=['yes' 'no']
            region_categories=['southwest' 'southeast' 'northwest' 'northeast']
            
            logging.info("PipeLine Started !!!")
            
            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('standard scaler', StandardScaler(with_mean=False))
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHot Encode', OneHotEncoder(categories=[sex_categories, smoker_categories, region_categories], handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor=ColumnTransformer([
                ('num pipeline', num_pipeline, numerical_columns),
                ('cat pipeline', cat_pipeline, categ_columns)
            ])
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured while data transformation")
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            preprocessor_obj=self.get_data_transformation()
            
            logging.info("read train and test data completed!!")
            logging.info(f"Train DataFrame : \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame : \n{test_df.head().to_string()}")
            
            
            input_features_train_df=train_df.drop('charges', axis=1)
            target_feature_train_df=train_df['charges']
            
            input_featutes_test_df=test_df.drop('charges', axis=1)
            target_feature_test_df=test_df['charges']
            
            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessor_obj.transform(input_featutes_test_df)
            
            train_arr=np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_features_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor_obj
            )
            logging.info('Preprocessor pickle file saved')
            
            return(
                train_arr, test_arr
            )
            
        except Exception as e:
            logging.info('Exception Occured while data transformation')
            return CustomException(e, sys)
    
                           