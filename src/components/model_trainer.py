# Basic Import
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
from dataclasses import dataclass
import os
import sys
# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and test input data")
            X_train,Y_train,X_test,Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decission Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbours classifer" : KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Classifer" : AdaBoostRegressor(),
            }

            model_report:dict = evaluate_models(X_train=X_train, Y_train=Y_train,X_test = X_test, 
                                               Y_test=Y_test, models = models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                raise CustomException("No best model found")
            
            logging.info(" Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(Y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)







