import pandas as pd
from scripts.config import INPUT_TRAINING_DATA_FOLDER
from scripts.config import INPUT_FORECAST_DATA_FOLDER
import os


def import_train_data():

    try:
        # Check if the file exists in the input folder
        if 'water_potability.csv' not in os.listdir(INPUT_TRAINING_DATA_FOLDER):
            raise FileNotFoundError("water_potability.csv file not found in the input folder.")
        
        data = pd.read_csv(f"{INPUT_TRAINING_DATA_FOLDER}/water_potability.csv", sep = ',')
        #mock code to simulate new prevision input
        last_15_rows = data.tail(2000)
        last_15_rows.to_csv(f"{INPUT_FORECAST_DATA_FOLDER}/prevision_X_mock.csv", index=False)
        return data
        
    except FileNotFoundError as e:
     print("Error:", e)

def import_forecast_data():

    try:
        # Check if the file exists in the input folder
        if len (os.listdir(INPUT_FORECAST_DATA_FOLDER)) == 0:
            raise FileNotFoundError(f"File not found in the path {INPUT_FORECAST_DATA_FOLDER}.")

        filename = list(os.listdir(INPUT_FORECAST_DATA_FOLDER))[0]
        data = pd.read_csv(f"{INPUT_FORECAST_DATA_FOLDER}{filename}", sep = ',')
        return data
        
    except FileNotFoundError as e:
     print("Error:", e)
