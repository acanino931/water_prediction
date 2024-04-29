from scripts.Import_data import import_train_data
from scripts.Import_data import import_forecast_data
from scripts.load_model import load_model
from scripts.preprocessing_functions import preprocess_data
import logging
import pandas as pd


if __name__ == "__main__":

    data = import_train_data()
    data.columns
    target_var = 'Potability'
    model,model_df = load_model(data, target_var = target_var )


    prevision_x = import_forecast_data()
    prevision_x= preprocess_data(prevision_x, target_var = target_var, forecast_data = True)
    
    if target_var in prevision_x.columns:
        prevision_x = prevision_x.drop(target_var, axis=1)
    prediction = model.predict(prevision_x)

    y_prev_prob = model.predict_proba(prevision_x)[:, 1]

    model_prediction = pd.DataFrame({'Potability_Prediction': prediction, 'Potability_Probability': y_prev_prob})

    model_prediction.to_excel('output/prediction.xlsx')
    print(prediction)
    logging.info("Prediction done code excecuted successfully")


